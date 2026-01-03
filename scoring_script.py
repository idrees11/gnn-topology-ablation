# scoring_script.py
import os
import pandas as pd
import io
from sklearn.metrics import f1_score

# ----------------------------
# CONFIG
# ----------------------------

# Environment variable containing private labels CSV (for GitHub Actions)
PRIVATE_LABELS_ENV = "TEST_LABELS"

# Local path to private labels (for local testing on Windows)
LOCAL_PRIVATE_LABEL_FILE = r"C:\Users\Idrees Bhat\Desktop\GNNs\Basira\Islem\private_data\test_labels.csv"

# Path to participant submissions folder
SUBMISSIONS_FOLDER = "submissions"

# Leaderboard CSV (updated automatically)
LEADERBOARD_FILE = "leaderboard.csv"

# ----------------------------
# Load private labels
# ----------------------------

private_labels_csv = os.getenv(PRIVATE_LABELS_ENV)

if private_labels_csv:
    # GitHub Actions: read CSV from secret
    truth = pd.read_csv(io.StringIO(private_labels_csv))
    print("Loaded private labels from GitHub secret.")
elif os.path.exists(LOCAL_PRIVATE_LABEL_FILE):
    # Local testing: read CSV from Windows path
    truth = pd.read_csv(LOCAL_PRIVATE_LABEL_FILE)
    print(f"Loaded private labels from local file: {LOCAL_PRIVATE_LABEL_FILE}")
else:
    raise FileNotFoundError(
        f"Private labels not found. Tried env '{PRIVATE_LABELS_ENV}' and local file '{LOCAL_PRIVATE_LABEL_FILE}'."
    )

truth.columns = truth.columns.str.strip()

# Detect label column
for col in ['label', 'target']:
    if col in truth.columns:
        truth_col = col
        break
else:
    raise ValueError("No 'label' or 'target' column found in private labels.")

# ----------------------------
# Evaluate each submission
# ----------------------------

scores = []

if not os.path.exists(SUBMISSIONS_FOLDER):
    raise FileNotFoundError(f"Submissions folder not found: {SUBMISSIONS_FOLDER}")

for fname in os.listdir(SUBMISSIONS_FOLDER):
    if fname.endswith(".csv"):
        submission_path = os.path.join(SUBMISSIONS_FOLDER, fname)
        submission = pd.read_csv(submission_path)
        submission.columns = submission.columns.str.strip()

        # Detect label column in submission
        for col in ['label', 'target']:
            if col in submission.columns:
                submission_col = col
                break
        else:
            print(f"Skipping {fname}: No 'label' or 'target' column found")
            continue

        # Check length
        if len(submission) != len(truth):
            print(f"Skipping {fname}: Length mismatch with ground truth")
            continue

        # Compute F1 score (macro)
        score = f1_score(
            truth[truth_col],
            submission[submission_col],
            average="macro"
        )

        print(f"{fname} -> F1 Score: {score:.4f}")
        scores.append({
            "submission": fname,
            "f1_score": score
        })

# ----------------------------
# Save leaderboard
# ----------------------------

if scores:
    leaderboard = pd.DataFrame(scores)
    leaderboard = leaderboard.sort_values(by="f1_score", ascending=False)
    leaderboard.to_csv(LEADERBOARD_FILE, index=False)
    print(f"\nLeaderboard saved to: {LEADERBOARD_FILE}")
else:
    print("No valid submissions found to score.")
