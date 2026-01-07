import os
import sys
import pandas as pd
from sklearn.metrics import f1_score

# ----------------------------
# Constants
# ----------------------------
PRIVATE_LABELS_ENV = "TEST_LABELS"
SUBMISSIONS_FOLDER = "submissions"
LEADERBOARD_FILE = "leaderboard.csv"

print("Running scoring_script.py from:", sys.argv[0])

# ----------------------------
# Load private labels
# ----------------------------
truth = None
truth_path = os.getenv(PRIVATE_LABELS_ENV)

if truth_path and os.path.exists(truth_path) and os.path.getsize(truth_path) > 0:
    try:
        truth = pd.read_csv(truth_path)
        truth.columns = truth.columns.str.strip().str.lower()
        print(f"Loaded private labels from file: {truth_path}")
    except pd.errors.EmptyDataError:
        print(f"ERROR: Private labels file {truth_path} is empty or invalid.")
        truth = None
else:
    print(
        "INFO: Private labels unavailable.\n"
        "Scoring skipped (EXPECTED for PRs/forks).\n"
        "Scoring runs only on push to main or manual workflow trigger."
    )

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
    for fname in os.listdir(SUBMISSIONS_FOLDER):
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
            print(f"Skipping {fname}: missing 'label' or 'target' column")
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
                print(f"Skipping {fname}: missing 'graph_index' or 'id'")
                continue

            # 🔑 FIX: force same dtype for merge key
            try:
                truth[id_col] = pd.to_numeric(truth[id_col], errors="coerce").astype("Int64")
                submission[id_col] = pd.to_numeric(submission[id_col], errors="coerce").astype("Int64")
            except Exception as e:
                print(f"Skipping {fname}: invalid ID column ({e})")
                continue

            # Drop rows with invalid IDs
            truth_valid = truth.dropna(subset=[id_col])
            submission_valid = submission.dropna(subset=[id_col])

            merged = truth_valid.merge(
                submission_valid,
                on=id_col,
                suffixes=("_true", "_pred"),
                how="inner"
            )

            print(f"Truth rows: {len(truth_valid)} | Merged rows: {len(merged)}")

            if merged.empty:
                print(f"Skipping {fname}: no matching IDs")
                continue

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
            scores.append({"submission": fname, "f1_score": None})

# ----------------------------
# Save leaderboard
# ----------------------------
leaderboard = pd.DataFrame(scores)

if not leaderboard.empty and truth is not None:
    leaderboard = leaderboard.sort_values(by="f1_score", ascending=False)

leaderboard.to_csv(LEADERBOARD_FILE, index=False)
print(f"Leaderboard saved to {LEADERBOARD_FILE}")
