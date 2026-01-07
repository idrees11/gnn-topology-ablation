import os
import sys
import pandas as pd
from sklearn.metrics import f1_score
import base64
import io

# ----------------------------
# Constants
# ----------------------------
PRIVATE_LABELS_ENV = "TEST_LABELS_B64"
SUBMISSIONS_FOLDER = "submissions"
LEADERBOARD_FILE = "leaderboard.csv"

print("Running scoring_script.py from:", sys.argv[0])

# ----------------------------
# Load private labels (BASE64)
# ----------------------------
truth = None
labels_b64 = os.getenv(PRIVATE_LABELS_ENV)

if not labels_b64:
    print(
        "INFO: Private labels unavailable.\n"
        "Scoring skipped (EXPECTED for PRs/forks).\n"
        "Scoring runs only on push to main or manual workflow trigger."
    )
else:
    try:
        decoded = base64.b64decode(labels_b64)
        truth = pd.read_csv(io.BytesIO(decoded))
        truth.columns = truth.columns.str.strip().str.lower()
        print("Loaded private labels from TEST_LABELS_B64")
    except Exception as e:
        print(f"ERROR: Failed to decode TEST_LABELS_B64: {e}")
        truth = None

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
# Evaluate submissions
# ----------------------------
scores = []

if not os.path.exists(SUBMISSIONS_FOLDER):
    print(f"No submissions folder found: {SUBMISSIONS_FOLDER}")
else:
    for fname in sorted(os.listdir(SUBMISSIONS_FOLDER)):
        if not fname.endswith(".csv"):
            continue

        submission_path = os.path.join(SUBMISSIONS_FOLDER, fname)
        submission = pd.read_csv(submission_path)
        submission.columns = submission.columns.str.strip().str.lower()

        print(f"\nProcessing submission: {fname}")
        print(f"Submission rows: {len(submission)}")

        # Detect prediction column
        if "label" in submission.columns:
            submission_col = "label"
        elif "target" in submission.columns:
            submission_col = "target"
        else:
            print(f"Skipping {fname}: missing 'label' or 'target'")
            continue

        # ----------------------------
        # Organiser scoring
        # ----------------------------
        if truth is not None:
            # Detect ID column
            if "graph_index" in truth.columns and "graph_index" in submission.columns:
                id_col = "graph_index"
            elif "id" in truth.columns and "id" in submission.columns:
                id_col = "id"
            else:
                print(f"Skipping {fname}: missing ID column")
                continue

            # Force numeric ID dtype
            truth[id_col] = pd.to_numeric(truth[id_col], errors="coerce")
            submission[id_col] = pd.to_numeric(submission[id_col], errors="coerce")

            # Drop invalid IDs
            truth_clean = truth.dropna(subset=[id_col]).copy()
            submission_clean = submission.dropna(subset=[id_col]).copy()

            # Convert to int (safe after dropna)
            truth_clean[id_col] = truth_clean[id_col].astype(int)
            submission_clean[id_col] = submission_clean[id_col].astype(int)

            print("ID dtype (truth):", truth_clean[id_col].dtype)
            print("ID dtype (submission):", submission_clean[id_col].dtype)

            # Merge truth and submission
            merged = truth_clean.merge(
                submission_clean,
                on=id_col,
                suffixes=("_true", "_pred"),
                how="inner"
            )

            print(f"Truth rows: {len(truth_clean)} | Merged rows: {len(merged)}")

            if merged.empty:
                print(f"Skipping {fname}: no matching IDs")
                continue

            # Compute F1 score
            score = f1_score(
                merged[f"{truth_col}_true"],
                merged[f"{submission_col}_pred"],
                average="macro"
            )

            print(f"{fname} -> F1 (macro): {score:.4f}")

            scores.append({
                "submission": fname,
                "f1_score": round(score, 6)
            })

        # ----------------------------
        # Participant / fork mode
        # ----------------------------
        else:
            print(f"Found submission (scoring skipped): {fname}")
            scores.append({
                "submission": fname,
                "f1_score": None
            })

# ----------------------------
# Save leaderboard
# ----------------------------
leaderboard = pd.DataFrame(scores)

if not leaderboard.empty and truth is not None:
    leaderboard = leaderboard.sort_values(by="f1_score", ascending=False)

leaderboard.to_csv(LEADERBOARD_FILE, index=False)
print(f"Leaderboard saved to {LEADERBOARD_FILE}")
