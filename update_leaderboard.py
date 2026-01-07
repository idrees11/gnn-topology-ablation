"""
update_leaderboard.py
Safely updates leaderboard/leaderboard.md using submissions
Handles PRs/forks where private labels may not be available.
"""

import os
import io
import base64
import pandas as pd
from datetime import datetime
from sklearn.metrics import f1_score

# ----------------------------
# Constants
# ----------------------------
SUBMISSIONS_DIR = "submissions"
LEADERBOARD_DIR = "leaderboard"
LEADERBOARD_FILE = os.path.join(LEADERBOARD_DIR, "leaderboard.md")

# ----------------------------
# Ensure submissions folder exists
# ----------------------------
if not os.path.exists(SUBMISSIONS_DIR):
    print(f"INFO: '{SUBMISSIONS_DIR}' folder not found. Creating it.")
    os.makedirs(SUBMISSIONS_DIR)

# ----------------------------
# Load private labels from TEST_LABELS_B64
# ----------------------------
def load_private_labels():
    labels_b64 = os.getenv("TEST_LABELS_B64")
    if not labels_b64:
        print("INFO: TEST_LABELS_B64 secret not found. Scoring skipped.")
        return None
    try:
        decoded_csv = base64.b64decode(labels_b64).decode("utf-8")
        truth = pd.read_csv(io.StringIO(decoded_csv))
        truth.columns = truth.columns.str.strip()
        # Ensure 'target' column exists
        if "target" not in truth.columns:
            if "label" in truth.columns:
                truth["target"] = truth["label"]
            else:
                raise ValueError("Private labels CSV must contain 'label' or 'target' column")
        return truth
    except Exception as e:
        print(f"ERROR: Failed to decode TEST_LABELS_B64: {e}")
        return None

# ----------------------------
# Score a single submission
# ----------------------------
def score_submission(sub_path, truth):
    df = pd.read_csv(sub_path, sep=None, engine="python")  # auto-detect CSV/TSV
    df.columns = df.columns.str.strip()
    pred_col = "label" if "label" in df.columns else "target"

    if pred_col not in df.columns:
        raise ValueError(f"Submission {sub_path} missing prediction column")

    # Align lengths (in case submission is longer than truth)
    min_len = min(len(df), len(truth))
    y_true = truth["target"].iloc[:min_len]
    y_pred = df[pred_col].iloc[:min_len]

    return round(f1_score(y_true, y_pred, average="macro"), 4)

# ----------------------------
# Write leaderboard markdown
# ----------------------------
def write_leaderboard(entries):
    os.makedirs(LEADERBOARD_DIR, exist_ok=True)
    with open(LEADERBOARD_FILE, "w") as f:
        f.write("# 🏆 GNN Challenge Leaderboard\n\n")
        f.write("| Rank | Participant | F1 Score | Submission | Timestamp |\n")
        f.write("|------|------------|----------|------------|-----------|\n")

        if not entries:
            f.write("| - | - | - | - | - |\n")
            print("Leaderboard is empty (no submissions).")
            return

        # Sort entries by F1 descending; put N/A/errors last
        def sort_key(x):
            try:
                return float(x["f1_score"])
            except:
                return -1  # N/A or Error at bottom

        entries_sorted = sorted(entries, key=sort_key, reverse=True)

        for i, e in enumerate(entries_sorted, start=1):
            f.write(
                f"| {i} | {e['participant']} | {e['f1_score']} | "
                f"{e['submission']} | {e['timestamp']} |\n"
            )

    print(f"Leaderboard updated → {LEADERBOARD_FILE}")

# ----------------------------
# Main function
# ----------------------------
def update_leaderboard():
    leaderboard = []

    # Load truth labels (None if PR/fork)
    truth = load_private_labels()

    # Iterate submissions
    for file in os.listdir(SUBMISSIONS_DIR):
        if not file.endswith(".csv"):
            continue

        path = os.path.join(SUBMISSIONS_DIR, file)
        participant = file.replace(".csv", "")

        try:
            score = "N/A" if truth is None else score_submission(path, truth)
        except Exception as e:
            print(f"ERROR scoring {file}: {e}")
            score = "Error"

        leaderboard.append({
            "participant": participant,
            "f1_score": score,
            "submission": file,
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        })

    write_leaderboard(leaderboard)

# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    update_leaderboard()
