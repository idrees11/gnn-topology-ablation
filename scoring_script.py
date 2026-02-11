import os
import sys
import pandas as pd
from sklearn.metrics import f1_score
import base64
import io
import json
from datetime import datetime

# ----------------------------
# Constants
# ----------------------------
PRIVATE_LABELS_ENV = "TEST_LABELS_B64"
SUBMISSIONS_FOLDER = "submissions"
LEADERBOARD_FILE = "leaderboard.csv"
SCORES_JSON_FILE = "scores.json"

print("Running scoring_script.py from:", sys.argv[0])

# ----------------------------
# Load private labels (BASE64)
# ----------------------------
truth = None
labels_b64 = os.getenv(PRIVATE_LABELS_ENV)

if labels_b64:
    try:
        decoded = base64.b64decode(labels_b64)
        truth = pd.read_csv(io.BytesIO(decoded))
        truth.columns = truth.columns.str.strip().str.lower()
        print("Private labels loaded")
    except Exception as e:
        print("ERROR decoding private labels:", e)
        truth = None
else:
    print("Private labels not available → scoring skipped")

# ----------------------------
# Detect truth column
# ----------------------------
truth_col = None
if truth is not None:
    if "label" in truth.columns:
        truth_col = "label"
    elif "target" in truth.columns:
        truth_col = "target"
    else:
        print("ERROR: truth must contain label or target")
        truth = None

# ----------------------------
# Score helper
# ----------------------------
def compute_score(submission, truth):
    submission.columns = submission.columns.str.strip().str.lower()

    if "label" in submission.columns:
        pred_col = "label"
    elif "target" in submission.columns:
        pred_col = "target"
    else:
        return None

    if "graph_index" in truth.columns and "graph_index" in submission.columns:
        id_col = "graph_index"
    elif "id" in truth.columns and "id" in submission.columns:
        id_col = "id"
    else:
        return None

    truth_clean = truth.copy()
    submission_clean = submission.copy()

    truth_clean[id_col] = pd.to_numeric(truth_clean[id_col], errors="coerce")
    submission_clean[id_col] = pd.to_numeric(submission_clean[id_col], errors="coerce")

    truth_clean = truth_clean.dropna(subset=[id_col])
    submission_clean = submission_clean.dropna(subset=[id_col])

    truth_clean[id_col] = truth_clean[id_col].astype(int)
    submission_clean[id_col] = submission_clean[id_col].astype(int)

    merged = truth_clean.merge(
        submission_clean,
        on=id_col,
        suffixes=("_true", "_pred"),
        how="inner"
    )

    if merged.empty:
        return None

    y_true = merged[f"{truth_col}_true"]
    y_pred = merged[f"{pred_col}_pred"]

    return f1_score(y_true, y_pred, average="macro")

# ----------------------------
# Evaluate paired submission
# ----------------------------
scores = []

ideal_path = os.path.join(SUBMISSIONS_FOLDER, "ideal_submission.csv")
perturbed_path = os.path.join(SUBMISSIONS_FOLDER, "perturbed_submission.csv")

if os.path.exists(ideal_path) and os.path.exists(perturbed_path):

    print("Paired submission detected")

    if truth is not None:
        ideal_df = pd.read_csv(ideal_path)
        perturbed_df = pd.read_csv(perturbed_path)

        f1_ideal = compute_score(ideal_df, truth)
        f1_perturbed = compute_score(perturbed_df, truth)

        if f1_ideal is not None and f1_perturbed is not None:
            robustness_gap = f1_ideal - f1_perturbed
            final_score = (f1_ideal + f1_perturbed) / 2

            scores.append({
                "rank": 1,
                "participant": "latest_submission",
                "f1_ideal": round(f1_ideal, 6),
                "f1_perturbed": round(f1_perturbed, 6),
                "robustness_gap": round(robustness_gap, 6),
                "timestamp": datetime.utcnow().isoformat()
            })

            print("Scoring complete")
        else:
            print("ERROR computing scores")

else:
    print("Paired submission files not found")

# ----------------------------
# Save leaderboard
# ----------------------------
leaderboard = pd.DataFrame(scores)

if not leaderboard.empty:
    leaderboard = leaderboard.sort_values(by="f1_perturbed", ascending=False)
else:
    leaderboard = pd.DataFrame(columns=[
        "rank", "participant", "f1_ideal",
        "f1_perturbed", "robustness_gap", "timestamp"
    ])

leaderboard.to_csv(LEADERBOARD_FILE, index=False)
print("Leaderboard updated")

# ----------------------------
# Save JSON for workflow
# ----------------------------
with open(SCORES_JSON_FILE, "w") as f:
    json.dump(scores, f, indent=2)

print("Scores JSON saved")
