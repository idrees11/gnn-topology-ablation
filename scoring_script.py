import os
import sys
import pandas as pd
from sklearn.metrics import f1_score

# ----------------------------
# Constants
# ----------------------------
PRIVATE_LABELS_ENV = "TEST_LABELS"   # env var holds FILE PATH
SUBMISSIONS_FOLDER = "submissions"
LEADERBOARD_FILE = "leaderboard.csv"

print("Running scoring_script.py from:", sys.argv[0])

# ----------------------------
# Load private labels (ORGANISER ONLY)
# ----------------------------
truth = None
truth_path = os.getenv(PRIVATE_LABELS_ENV)

if truth_path and os.path.exists(truth_path):
    truth = pd.read_csv(truth_path)
    truth.columns = truth.columns.str.strip().str.lower()
    print(f"Loaded private labels from file: {truth_path}")
else:
    print(
        "INFO: Private labels unavailable.\n"
        "Scoring skipped (this is EXPECTED for PRs and forks).\n"
        "Scoring runs only on push to main or manual workflow trigger."
    )

# ----------------------------
# Detect truth column (label preferred)
# ----------------------------
truth_col = None
if truth is not None:
    if "label" in truth.columns:
        truth_col = "label"
    elif "target" in truth.columns:
        truth_col = "target"
    else:
        print("ERROR: Private labels must contain a 'label' column.")
        print("Found columns:", truth.columns.tolist())
        truth = None

# ----------------------------
# Evaluate submissions
# ----------------------------
scores = []

if not os.path.exists(SUBMISSIONS_FOLDER):
    print(f"No submissions folder found: {SUBMISSIONS_FOLDER}")
else:
    for fname in os.listdir(SUBMISSIONS_FOLDER):
        if not fname.endswith(".csv"):
            continue

        submission_path = os.path.join(SUBMISSIONS_FOLDER, fname)
        submission = pd.read_csv(submission_path)
        submission.columns = submission.columns.str.strip().str.lower()

        # Detect submission column
        if "label" in submission.columns:
            submission_col = "label"
        elif "target" in submission.columns:
            submission_col = "target"
        else:
            print(f"Skipping {fname}: missing 'label' column")
            continue

        # ----------------------------
        # Scoring (organiser only)
        # ----------------------------
        if truth is not None:
            if len(submission) != len(truth):
                print(f"Skipping {fname}: length mismatch")
                continue

            score = f1_score(
                truth[truth_col],
                submission[submission_col],
                average="macro"
            )

            print(f"{fname} -> F1 (macro): {score:.4f}")
            scores.append({
                "submission": fname,
                "f1_score": round(score, 6)
            })
        else:
            print(f"Found submission (scoring skipped): {fname}")
            scores.append({
                "submission": fname,
                "f1_score": None
            })

# ----------------------------
# Save leaderboard
# ----------------------------
if scores:
    leaderboard = pd.DataFrame(scores)

    if truth is not None:
        leaderboard = leaderboard.sort_values(
            by="f1_score",
            ascending=False
        )

    leaderboard.to_csv(LEADERBOARD_FILE, index=False)
    print(f"Leaderboard saved to {LEADERBOARD_FILE}")
else:
    print("No valid submissions found.")
