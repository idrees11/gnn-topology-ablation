import os
import io
import base64
import pandas as pd
from datetime import datetime
from sklearn.metrics import f1_score
import argparse
import json

SUBMISSIONS_DIR = "submissions"
LEADERBOARD_DIR = "leaderboard"
LEADERBOARD_MD = os.path.join(LEADERBOARD_DIR, "leaderboard.md")
LEADERBOARD_CSV = os.path.join(LEADERBOARD_DIR, "leaderboard_history.csv")

os.makedirs(LEADERBOARD_DIR, exist_ok=True)
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)

# ----------------------------
# Write leaderboard markdown
# ----------------------------
def write_leaderboard_markdown(df):
    with open(LEADERBOARD_MD, "w") as f:
        f.write("# 🏆 GNN (Topology Ablation) Robustness Challenge Leaderboard\n\n")
        f.write("| Rank | Participant | F1 Ideal | F1 Perturbed | Robustness Gap | Timestamp |\n")
        f.write("|------|------------|----------|--------------|----------------|-----------|\n")

        if df.empty:
            f.write("| - | - | - | - | - | - |\n")
            return

        df_sorted = df.sort_values(by="f1_perturbed", ascending=False)

        for i, row in enumerate(df_sorted.itertuples(index=False), start=1):
            f.write(
                f"| {i} | {row.participant} | {row.f1_ideal} | "
                f"{row.f1_perturbed} | {row.robustness_gap} | {row.timestamp} |\n"
            )

    print("Leaderboard markdown updated")

# ----------------------------
# Append new score entry
# ----------------------------
def append_score(entry):
    if os.path.exists(LEADERBOARD_CSV):
        df = pd.read_csv(LEADERBOARD_CSV)
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    else:
        df = pd.DataFrame([entry])

    df.to_csv(LEADERBOARD_CSV, index=False)
    write_leaderboard_markdown(df)

# ----------------------------
# Update leaderboard from scores.json
# ----------------------------
def update_leaderboard(scores_file):
    if not scores_file or not os.path.exists(scores_file):
        print("No scores.json found")
        return

    with open(scores_file, "r") as f:
        scores = json.load(f)

    for s in scores:
        entry = {
            "participant": s.get("participant", "unknown"),
            "f1_ideal": s.get("f1_ideal", "N/A"),
            "f1_perturbed": s.get("f1_perturbed", "N/A"),
            "robustness_gap": s.get("robustness_gap", "N/A"),
            "timestamp": s.get("timestamp", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"))
        }
        append_score(entry)

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=str, default=None)
    parser.add_argument("--participant", type=str, default="unknown")
    args = parser.parse_args()

    # Inject participant name if missing
    if args.scores and os.path.exists(args.scores):
        with open(args.scores, "r") as f:
            data = json.load(f)

        for d in data:
            if "participant" not in d or d["participant"] == "unknown":
                d["participant"] = args.participant

        with open(args.scores, "w") as f:
            json.dump(data, f, indent=2)

    update_leaderboard(args.scores)
