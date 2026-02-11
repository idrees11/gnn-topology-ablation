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

IDEAL_FILE = "ideal_submission.csv"
PERTURBED_FILE = "perturbed_submission.csv"

print("Running scoring_script.py from:", sys.argv[0])
print("Working directory:", os.getcwd())

# ----------------------------
# Ensure submissions folder exists
# ----------------------------
if not os.path.exists(SUBMISSIONS_FOLDER):
    print(f"Submissions folder not found: {SUBMISSIONS_FOLDER}")
    os.makedirs(SUBMISSIONS_FOLDER, exist_ok=True)

print("Files inside submissions/:", os.listdir(SUBMISSIONS_FOLDER))

# ----------------------------
# Load private labels
# ----------------------------
truth = None
labels_b64 = os.getenv(PRIVATE_LABELS_ENV)

if not labels_b64:
    print("INFO: Private labels unavailable. Scoring skipped (PR-safe mode).")
else:
    decoded = base64.b64decode(labels_b64)
    truth = pd.read_csv(io.BytesIO(decoded))
    truth.columns = truth.columns.str.strip().str.lower()
    print("Loaded private labels")

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
# Helper function to compute F1
# ----------------------------
def compute_f1(submission_path):
    print("Reading:", submission_path)

    submission = pd.read_csv(submission_path)
    submission.columns = submission.columns.str.strip().str.lower()

    if "label" in submission.columns:
        submission_col = "label"
    elif "target" in submission.columns:
        submission_col = "target"
    else:
        print(f"Skipping {submission_path}: missing label column")
        return None

    if "graph_index" in truth.columns and "graph_index" in submission.columns:
        id_col = "graph_index"
    elif "id" in truth.columns and "id" in submission.columns:
        id_col = "id"
    else:
        print(f"Skipping {submission_path}: missing ID column")
        return None

    truth_clean = truth.dropna(subset=[id_col]).copy()
    submission_clean = submission.dropna(subset=[id_col]).copy()

    truth_clean[id_col] = truth_clean[id_col].astype(int)
    submission_clean[id_col] = submission_clean[id_col].astype(int)

    merged = truth_clean.merge(
        submission_clean,
        on=id_col,
        suffixes=("_true", "_pred"),
        how="inner"
    )

    print("Merged rows:", len(merged))

    if merged.empty:
        print(f"Skipping {submission_path}: no matching IDs")
        return None

    y_true = merged[f"{truth_col}_true"]
    y_pred = merged[f"{submission_col}_pred"]

    return f1_score(y_true, y_pred, average="macro")

# ----------------------------
# Evaluate ideal & perturbed ONLY
# ----------------------------
scores = []

ideal_path = os.path.join(SUBMISSIONS_FOLDER, IDEAL_FILE)
perturbed_path = os.path.join(SUBMISSIONS_FOLDER, PERTURBED_FILE)

participant_name = os.getenv("GITHUB_ACTOR", "unknown")
timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

print("Looking for required files:")
print("Ideal:", ideal_path, os.path.exists(ideal_path))
print("Perturbed:", perturbed_path, os.path.exists(perturbed_path))

if truth is not None and os.path.exists(ideal_path) and os.path.exists(perturbed_path):

    print("\nScoring IDEAL submission...")
    f1_ideal = compute_f1(ideal_path)

    print("Scoring PERTURBED submission...")
    f1_perturbed = compute_f1(perturbed_path)

    if f1_ideal is not None and f1_perturbed is not None:
        gap = f1_ideal - f1_perturbed

        scores.append({
            "participant": participant_name,
            "f1_ideal": round(f1_ideal, 6),
            "f1_perturbed": round(f1_perturbed, 6),
            "robustness_gap": round(gap, 6),
            "timestamp": timestamp
        })

        print(f"\nParticipant: {participant_name}")
        print(f"F1 Ideal: {f1_ideal:.4f}")
        print(f"F1 Perturbed: {f1_perturbed:.4f}")
        print(f"Robustness Gap: {gap:.4f}")

else:
    print("Required submission files missing OR labels unavailable.")
    print("Scoring skipped safely.")

# ----------------------------
# Save leaderboard CSV
# ----------------------------
leaderboard = pd.DataFrame(scores)
leaderboard.to_csv(LEADERBOARD_FILE, index=False)
print(f"Leaderboard saved to {LEADERBOARD_FILE}")

# ----------------------------
# Save scores.json
# ----------------------------
with open(SCORES_JSON_FILE, "w") as f:
    json.dump(scores, f, indent=2)

print(f"Scores saved to {SCORES_JSON_FILE}")
