"""
update_leaderboard.py
Safely updates leaderboard/leaderboard.md based on participant submissions.
Scoring only runs when TEST_LABELS secret is available (push to main / manual run).
"""

import os
import io
import pandas as pd
from datetime import datetime
from sklearn.metrics import f1_score

SUBMISSIONS_DIR = "submissions"
LEADERBOARD_DIR = "leaderboard"
LEADERBOARD_FILE = os.path.join(LEADERBOARD_DIR, "leaderboard.md")

# ----------------------------
# Helper: score one submission
# ----------------------------
def run_scoring(submission_path):
    """Return F1 score or None if scoring is unavailable"""

    private_labels_csv = os.getenv("TEST_LABELS")
    if not private_labels_csv:
        print("INFO: TEST_LABELS secret not available. Skipping scoring.")
        return None

    # Load submission
    df_sub = pd.read_csv(submission_path)
    df_sub.columns = df_sub.columns.str.strip()

    label_col = "label" if "label" in df_sub.columns else "target"
    if label_col not in df_sub.columns:
        raise ValueError("Submission missing 'label' or 'target' column")

    # Load private labels from secret
    truth = pd.read_csv(io.StringIO(private_labels_csv))
    truth.columns = truth.columns.str.strip()

    if "target" not in truth.columns:
        raise ValueError("Private labels missing 'target' column")

    # Align lengths
    min_len = min(len(truth), len(df_sub))
    y_true = truth["target"].iloc[:min_len]
    y_pred = df_sub[label_col].iloc[:min_len]

    return f1_score(y_true, y_pred, average="macro")


# ----------------------------
# Update leaderboard
# ----------------------------
def update_leaderboard():
    os.makedirs(LEADERBOARD_DIR, exist_ok=True)

    # ---- No submissions folder ----
    if not os.path.exists(SUBMISSIONS_DIR):
        print(f"INFO: '{SUBMISSIONS_DIR}' folder not found. Writing empty leaderboard.")
        write_leaderboard([])
        return

    leaderboard = []

    for file in os.listdir(SUBMISSIONS_DIR):
        if not file.endswith(".csv"):
            continue

        submission_path = os.path.join(SUBMISSIONS_DIR, file)
        participant = file.replace(".csv", "")

        try:
            score = run_scoring(submission_path)
        except Exception as e:
            print(f"ERROR scoring {file}: {e}")
            score = "Error"

        leaderboard.append({
            "participant": participant,
            "f1_score": score if score is not None else "N/A",
            "submission_file": file,
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        })

    write_leaderboard(leaderboard)


# ----------------------------
# Write markdown file
# ----------------------------
def write_leaderboard(entries):
    with open(LEADERBOARD_FILE, "w") as f:
        f.write("# 🏆 Competition Leaderboard\n\n")
        f.write("| Rank | Participant | F1 Score | Submission | Timestamp |\n")
        f.write("|------|------------|----------|------------|-----------|\n")
