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
TEMPLATE_PREFIXES = ["sample", "example"]  # files to skip

print("Running scoring_script.py from:", sys.argv[0])

# ----------------------------
# Load private labels (ORGANISER ONLY)
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
        "Scoring skipped (this is EXPECTED for PRs and forks).\n"
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
        print("ERROR: Private labels must contain 'label' or 'target' column.")
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

        # Skip sample/template files
        if any(fname.lower().startswith(p) for p in TEMPLATE_PREFIXES):
            print(f"Skipping template/sample file: {fname}")
            continue

        submission_path = os.path.join(SUBMISSIONS_FOLDER, fname)
        try:
            submission = pd.read_csv(submission_path)
        except pd.errors.EmptyDataError:
            print(f"Skipping {fname}: file is empty or invalid")
            continue

        submission.columns = submission.columns.str.strip().str.lower()

        print(f"\nProcessing submission: {fname}")
        print(f"Submission rows: {len(submission)}")

        # ----------------------------
        # Detect prediction column
        # ----------------------------
        if "label" in submission.columns:
            submission_col = "label"
        elif "target" in submission.columns:
            submission_col = "target"
        else:
            print(f"Skipping {fname}: missing 'label' or 'target' column")
            continue

        # ----------------------------
        # Organiser scoring (truth available)
        # ----------------------------
        if truth is not None:
            # 🔑 Detect ID column (graph_index preferred, fallback to id)
            if "graph_index" in truth.columns and "graph_index" in submission.columns:
                id_col = "graph_index"
            elif "id" in truth.columns and "id" in submission.columns:
                id_col = "id"
            else:
                print(f"Skipping {fname}: missing 'graph_index' or 'id' column")
                continue

            merged = truth.merge(
                submission,
                on=id_col,
                suffixes=("_true", "_pred"),
                how="inner"
            )

            print(f"Truth rows: {len(truth)} | Merged rows: {len(merged)}")

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
        # Participant / fork PR mode
        # ----------------------------
        else:
            print(f"Found submission (scoring skipped): {fname}")
            scores.append({
                "submission": fname,
                "f1_score": None
            })

# ----------------------------
# Save leaderboard (always)
# ----------------------------
leaderboard = pd.DataFrame(scores)

if not leaderboard.empty and truth is not None:
    leaderboard = leaderboard.sort_values(by="f1_score", ascending=False)

leaderboard.to_csv(LEADERBOARD_FILE, index=False)
print(f"Leaderboard saved to {LEADERBOARD_FILE}")
