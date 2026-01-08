# leaderboard_system.py

import os
import io
import base64
import pandas as pd
from datetime import datetime
from sklearn.metrics import f1_score
import argparse
import json

# ----------------------------
# Constants: ensure folder exists
# ----------------------------
SUBMISSIONS_DIR = "submissions"
LEADERBOARD_DIR = "leaderboard"
LEADERBOARD_FILE = os.path.join(LEADERBOARD_DIR, "leaderboard.md")

# Make sure the folder exists
os.makedirs(LEADERBOARD_DIR, exist_ok=True)
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)

# ----------------------------
# Load private labels
# ----------------------------
def load_private_labels():
    labels_b64 = os.getenv("TEST_LABELS_B64")
    if not labels_b64:
        print("INFO: TEST_LABELS_B64 secret not found. Scores will be N/A.")
        return None
    try:
        decoded_csv = base64.b64decode(labels_b64).decode("utf-8")
        truth = pd.read_csv(io.StringIO(decoded_csv))
        truth.columns = truth.columns.str.strip()
        if "target" not in truth.columns:
            if "label" in truth.columns:
                truth["target"] = truth["label"]
            else:
                raise ValueError("Private labels CSV must contain 'label' or 'target'")
        return truth
    except Exception as e:
        print(f"ERROR decoding TEST_LABELS_B64: {e}")
        return None

# ----------------------------
# Score a single submission
# ----------------------------
def score_submission(sub_path, truth):
    df = pd.read_csv(sub_path, sep=None, engine="python")
    df.columns = df.columns.str.strip()
    pred_col = "label" if "label" in df.columns else "target"

    min_len = min(len(df), len(truth))
    y_true = truth["target"].iloc[:min_len]
    y_pred = df[pred_col].iloc[:min_len]

    return round(f1_score(y_true, y_pred, average="macro"), 4)

# ----------------------------
# Write leaderboard
# ----------------------------
def write_leaderboard(entries):
    with open(LEADERBOARD_FILE, "w") as f:
        f.write("# 🏆 GNN Challenge Leaderboard\n\n")
        f.write("| Rank | Participant | F1 Score | Submission | Timestamp |\n")
        f.write("|------|------------|----------|------------|-----------|\n")

        if not entries:
            f.write("| - | - | - | - | - |\n")
            return

        entries_sorted = sorted(
            entries, 
            key=lambda x: float(x["f1_score"]) if isinstance(x["f1_score"], (int, float)) else -1, 
            reverse=True
        )
        for i, e in enumerate(entries_sorted, start=1):
            f.write(
                f"| {i} | {e['participant']} | {e['f1_score']} | "
                f"{e['submission']} | {e['timestamp']} |\n"
            )

    print(f"Leaderboard updated → {LEADERBOARD_FILE}")

# ----------------------------
# Main
# ----------------------------
def update_leaderboard(scores_file=None):
    leaderboard = []

    if scores_file and os.path.exists(scores_file):
        # Load scores from JSON (PR-safe)
        with open(scores_file, "r") as f:
            scores = json.load(f)
        for s in scores:
            leaderboard.append({
                "participant": s.get("submission", "unknown").replace(".csv", ""),
                "f1_score": s.get("f1_score", "N/A"),
                "submission": s.get("submission", "N/A"),
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            })
    else:
        # Load submissions and private labels
        truth = load_private_labels()
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
# CLI
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores", type=str, default=None, 
        help="Optional path to scores.json to update leaderboard PR-safe"
    )
    args = parser.parse_args()

    update_leaderboard(scores_file=args.scores)
