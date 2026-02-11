import os
import sys
import pandas as pd
from sklearn.metrics import f1_score
import base64
import io
import json

# ----------------------------
# Constants
# ----------------------------
PRIVATE_LABELS_ENV = "TEST_LABELS_B64"
SUBMISSIONS_FOLDER = "submissions"
LEADERBOARD_FILE = "leaderboard.csv"
SCORES_JSON_FILE = "scores.json"

EXPECTED_FILES = ["ideal_submission.csv", "perturbed_submission.csv"]

print("Running scoring_script.py...")

# ----------------------------
# Load private labels
# ----------------------------
truth = None
labels_b64 = os.getenv(PRIVATE_LABELS_ENV)

if labels_b64:
    try:
        decoded = base64.b64decode(labels_b64)
        truth = pd.read_csv(io.BytesIO(decoded))
        truth.columns = truth.columns.str.strip().str.lower()
        print("Loaded private labels successfully")
    except Exception as e:
        print(f"ERROR: Failed to decode TEST_LABELS_B64: {e}")
        truth = None
else:
    print("Private labels unavailable. Scoring skipped for PRs/forks.")

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
        print("ERROR: Private labels must contain 'label' or 'target'.")
        truth = None

# ----------------------------
# Evaluate only the expected submission files
# ----------------------------
scores = []

for fname in EXPECTED_FILES:
    submission_path = os.path.join(SUBMISSIONS_FOLDER, fname)
    if not os.path.exists(submission_path):
        print(f"❌ Missing required submission file: {fname}")
        scores.append({"submission": fname, "f1_score": None})
        continue

    submission = pd.read_csv(submission_path)
    submission.columns = submission.columns.str.strip().str.lower()
    print(f"\nProcessing submission: {fname}")

    # Detect prediction column
    if "label" in submission.columns:
        submission_col = "label"
    elif "target" in submission.columns:
        submission_col = "target"
    else:
        print(f"❌ {fname} missing 'label' or 'target' column")
        scores.append({"submission": fname, "f1_score": None})
        continue

    # ----------------------------
    # Score against private labels
    # ----------------------------
    if truth is not None:
        if "graph_index" in truth.columns and "graph_index" in submission.columns:
            id_col = "graph_index"
        elif "id" in truth.columns and "id" in submission.columns:
            id_col = "id"
        else:
            print(f"❌ {fname} missing required ID column")
            scores.append({"submission": fname, "f1_score": None})
            continue

        truth[id_col] = pd.to_numeric(truth[id_col], errors="coerce")
        submission[id_col] = pd.to_numeric(submission[id_col], errors="coerce")
        truth_clean = truth.dropna(subset=[id_col]).copy()
        submission_clean = submission.dropna(subset=[id_col]).copy()
        truth_clean[id_col] = truth_clean[id_col].astype(int)
        submission_clean[id_col] = submission_clean[id_col].astype(int)

        merged = truth_clean.merge(
            submission_clean, on=id_col, suffixes=("_true", "_pred"), how="inner"
        )

        if merged.empty:
            print(f"❌ {fname}: no matching IDs")
            scores.append({"submission": fname, "f1_score": None})
            continue

        y_true_col = f"{truth_col}_true" if f"{truth_col}_true" in merged.columns else truth_col
        y_pred_col = f"{submission_col}_pred" if f"{submission_col}_pred" in merged.columns else submission_col

        score = f1_score(merged[y_true_col], merged[y_pred_col], average="macro")
        print(f"{fname} -> F1 (macro): {score:.4f}")
        scores.append({"submission": fname, "f1_score": round(score, 6)})

    else:
        print(f"Found {fname} (scoring skipped)")
        scores.append({"submission": fname, "f1_score": None})

# ----------------------------
# Save leaderboard CSV
# ----------------------------
leaderboard = pd.DataFrame(scores)
if not leaderboard.empty and truth is not None:
    leaderboard = leaderboard.sort_values(by="f1_score", ascending=False)

leaderboard.to_csv(LEADERBOARD_FILE, index=False)
print(f"Leaderboard saved to {LEADERBOARD_FILE}")

# ----------------------------
# Save scores JSON for leaderboard step
# ----------------------------
try:
    with open(SCORES_JSON_FILE, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"Scores saved to {SCORES_JSON_FILE} for leaderboard update")
except Exception as e:
    print(f"ERROR: Failed to save {SCORES_JSON_FILE}: {e}")
