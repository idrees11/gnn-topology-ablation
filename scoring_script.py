import os
import sys
import pandas as pd
from sklearn.metrics import f1_score

PRIVATE_LABELS_ENV = "TEST_LABELS"
SUBMISSIONS_FOLDER = "submissions"
LEADERBOARD_FILE = "leaderboard.csv"

print("Running scoring_script.py from:", sys.argv[0])

# ----------------------------
# Load private labels (FILE PATH ONLY)
# ----------------------------
truth = None
truth_path = os.getenv(PRIVATE_LABELS_ENV)

if truth_path and os.path.exists(truth_path):
    truth = pd.read_csv(truth_path)
    truth.columns = truth.columns.str.strip().str.lower()
    print(f"Loaded private labels from file: {truth_path}")
else:
    print(
        "WARNING: Private labels not found. "
        "Scoring will be skipped (expected for forks)."
    )

# Detect truth column
truth_col = None
if truth is not None:
    for col in ("label", "target"):
        if col in truth.columns:
            truth_col = col
            break

    if truth_col is None:
        print("ERROR: No 'label' or 'target' column in private labels.")
        print("Found columns:", truth.columns.tolist())
        truth = None

# ----------------------------
# Evaluate submissions
# ----------------------------
scores = []

if not os.path.exists(SUBMISSIONS_FOLDER):
    print(f"No submissions folder: {SUBMISSIONS_FOLDER}")
else:
    for fname in os.listdir(SUBMISSIONS_FOLDER):
        if not fname.endswith(".csv"):
            continue

        submission_path = os.path.join(SUBMISSIONS_FOLDER, fname)
        submission = pd.read_csv(submission_path)
        submission.columns = submission.columns.str.strip().str.lower()

        # Detect submission column
        submission_col = None
        for col in ("label", "target"):
            if col in submission.columns:
                submission_col = col
                break

        if submission_col is None:
            print(f"Skipping {fname}: no label column")
            continue

        if truth is not None:
            if len(submission) != len(truth):
                print(f"Skipping {fname}: length mismatch")
                continue

            score = f1_score(
                truth[truth_col],
                submission[submission_col],
                average="macro"
            )

            print(f"{fname} -> F1: {score:.4f}")
            scores.append({"submission": fname, "f1_score": score})
        else:
            print(f"Found submission (skipping scoring): {fname}")
            scores.append({"submission": fname, "f1_score": None})

# ----------------------------
# Save leaderboard
# ----------------------------
if scores:
    leaderboard = pd.DataFrame(scores)
    if truth is not None:
        leaderboard = leaderboard.sort_values(
            by="f1_score", ascending=False
        )
    leaderboard.to_csv(LEADERBOARD_FILE, index=False)
    print(f"Leaderboard saved to {LEADERBOARD_FILE}")
else:
    print("No valid submissions found.")
