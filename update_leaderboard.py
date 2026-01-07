"""
update_leaderboard.py
Safely updates leaderboard/leaderboard.md based on participant submissions.
Uses TEST_LABELS_B64 (base64-encoded CSV) secret for scoring.
"""

import os
import io
import base64
import pandas as pd
from datetime import datetime
from sklearn.metrics import f1_score

SUBMISSIONS_DIR = "submissions"
LEADERBOARD_DIR = "leaderboard"
LEADERBOARD_FILE = os.path.join(LEADERBOARD_DIR, "leaderboard.md")

# -------------------------------------------------
# Helper: load private labels from TEST_LABELS_B64
# -------------------------------------------------
def load_private_labels():
    labels_b64 = os.getenv("TEST_LABELS_B64")

    if not labels_b64:
        print("INFO: TEST_LABELS_B64 secret not available. Scoring will be skipped.")
        return None

    try:
        decoded_csv = base64.b64decode(labels_b64).decode("utf-8")
        truth = pd.read_csv(io.StringIO(decoded_csv))
        truth.columns = truth.columns.str.strip()
        return truth
    except Exception as e:
        raise RuntimeError(f"Failed to decode TEST_LABELS_B64: {e}")

# -------------------------------------------------
# Helper: score one submission
# -------------------------------------------------
def run_scoring(submission_path, truth):
    df_sub = pd.read_csv(submission_path)
    df_sub.columns = df_sub.columns.str.strip()

    label_col = "label" if "label" in df_sub.columns else "target"
    if label_col not in df_sub.columns:
        raise ValueError("Submission missing 'label' or 'target' column")

    if "target" not in truth.columns:
        raise ValueError("Private labels missing 'target' column")

    # Align lengths safely
    min_len = min(len(truth), len(df_sub))
    y_true = truth["target"].iloc[:min_len]
    y_pred = df_sub[label_col].iloc[:min_len]

    return f1_score(y_true, y_pred, average="macro")

# -------------------------------------------------
# Update leaderboard
# -------------------------------------------------
def update_leaderboard():
    os.makedirs(LEADERBOARD_DIR, exist_ok=True)

    # Load private labels (may be None in PRs/forks)
    truth = load_private_labels()

    if not os.path.exists(SUBMISSIONS_DIR):
        print(f"INFO: '{SUBMISSIONS_DIR}' folder not found.")
        write_leaderboard([])
        return

    leaderboard = []

    for file in os.listdir(SUBMISSIONS_DIR):
        if not file.endswith(".csv"):
            continue

        submission_path = os.path.join(SUBMISSIONS_DIR, file)
        participant = file.replace(".csv", "")

        try:
            if truth is None:
                score = "N/A"
            else:
                score = run_scoring(submission_path, truth)
        except Exception as e:
            print(f"ERROR scoring {file}: {e}")
            score = "Error"

        leaderboard.append({
            "participant": participant,
            "f1_score": score,
            "submission_file": file,
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        })

    write_leaderboard(leaderboard)

# -------------------------------------------------
# Write leaderboard markdown
# -------------------------------------------------
def write_leaderboard(entries):
    with open(LEADERBOARD_FILE, "w") as f:
        f.write("# 🏆 Competition Leaderboard\n\n")
        f.write("| Rank | Participant | F1 Score | Submission | Timestamp |\n")
        f.write("|------|------------|----------|------------|-----------|\n")

        if not entries:
            f.write("| - | - | - | - | - |\n")
        else:
            entries_sorted = sorted(
                entries,
                key=lambda x: float(x["f1_score"])
                if isinstance(x["f1_score"], (int, float))
                else -1,
                reverse=True
            )

            for i, e in enumerate(entries_sorted, start=1):
                f.write(
                    f"| {i} | {e['participant']} | {e['f1_score']} | "
                    f"{e['submission_file']} | {e['timestamp']} |\n"
                )

    print(f"Leaderboard updated → {LEADERBOARD_FILE}")

# -------------------------------------------------
# Main
# -------------------------------------------------
if __name__ == "__main__":
    update_leaderboard()
