"""
update_leaderboard.py
Safely updates leaderboard/leaderboard.md
Uses TEST_LABELS_B64 if available
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
# Load private labels
# -------------------------------------------------
def load_private_labels():
    labels_b64 = os.getenv("TEST_LABELS_B64")
    if not labels_b64:
        print("INFO: TEST_LABELS_B64 not available. Scoring skipped.")
        return None

    decoded = base64.b64decode(labels_b64).decode("utf-8")
    truth = pd.read_csv(io.StringIO(decoded))
    truth.columns = truth.columns.str.strip()
    return truth

# -------------------------------------------------
# Score one submission
# -------------------------------------------------
def score_submission(sub_path, truth):
    df = pd.read_csv(sub_path)
    df.columns = df.columns.str.strip()

    pred_col = "label" if "label" in df.columns else "target"
    if pred_col not in df.columns:
        raise ValueError("Missing prediction column")

    min_len = min(len(df), len(truth))
    return f1_score(
        truth["target"].iloc[:min_len],
        df[pred_col].iloc[:min_len],
        average="macro"
    )

# -------------------------------------------------
# Update leaderboard
# -------------------------------------------------
def update_leaderboard():
    os.makedirs(LEADERBOARD_DIR, exist_ok=True)

    truth = load_private_labels()
    leaderboard = []

    # ✅ CRITICAL FIX
    if not os.path.exists(SUBMISSIONS_DIR):
        print(f"INFO: '{SUBMISSIONS_DIR}' folder not found. Writing empty leaderboard.")
        write_leaderboard([])
        return

    for file in os.listdir(SUBMISSIONS_DIR):
        if not file.endswith(".csv"):
            continue

        path = os.path.join(SUBMISSIONS_DIR, file)
        name = file.replace(".csv", "")

        try:
            score = "N/A" if truth is None else score_submission(path, truth)
        except Exception as e:
            print(f"ERROR scoring {file}: {e}")
            score = "Error"

        leaderboard.append({
            "participant": name,
            "f1_score": score,
            "submission": file,
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        })

    write_leaderboard(leaderboard)

# -------------------------------------------------
# Write markdown
# -------------------------------------------------
def write_leaderboard(entries):
    with open(LEADERBOARD_FILE, "w") as f:
        f.write("# 🏆 Competition Leaderboard\n\n")
        f.write("| Rank | Participant | F1 Score | Submission | Timestamp |\n")
        f.write("|------|------------|----------|------------|-----------|\n")

        if not entries:
            f.write("| - | - | - | - | - |\n")
        else:
            entries = sorted(
                entries,
                key=lambda x: float(x["f1_score"])
                if isinstance(x["f1_score"], (int, float))
                else -1,
                reverse=True
            )

            for i, e in enumerate(entries, start=1):
                f.write(
                    f"| {i} | {e['participant']} | {e['f1_score']} | "
                    f"{e['submission']} | {e['timestamp']} |\n"
                )

    print(f"Leaderboard written → {LEADERBOARD_FILE}")

# -------------------------------------------------
if __name__ == "__main__":
    update_leaderboard()
