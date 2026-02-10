# leaderboard_system.py

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
LEADERBOARD_FILE = os.path.join(LEADERBOARD_DIR, "leaderboard.md")

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
    decoded_csv = base64.b64decode(labels_b64).decode("utf-8")
    truth = pd.read_csv(io.StringIO(decoded_csv))
    truth.columns = truth.columns.str.strip()
    if "target" not in truth.columns:
        truth["target"] = truth["label"]
    return truth


# ----------------------------
# Score a submission
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
        f.write("# 🏆 GNN Robustness Challenge Leaderboard\n\n")
        f.write("| Rank | Participant | F1 Ideal | F1 Perturbed | Robustness Gap | Timestamp |\n")
        f.write("|------|------------|---------|--------------|----------------|-----------|\n")

        if not entries:
            f.write("| - | - | - | - | - | - |\n")
            return

        entries_sorted = sorted(
            entries,
            key=lambda x: float(x["f1_perturbed"]) if isinstance(x["f1_perturbed"], (int, float)) else -1,
            reverse=True
        )

        for i, e in enumerate(entries_sorted, start=1):
            f.write(
                f"| {i} | {e['participant']} | {e['f1_ideal']} | "
                f"{e['f1_perturbed']} | {e['gap']} | {e['timestamp']} |\n"
            )

    print(f"Leaderboard updated → {LEADERBOARD_FILE}")


# ----------------------------
# Merge ideal + perturbed
# ----------------------------
def group_submissions(files):
    participants = {}

    for file in files:
        if not file.endswith(".csv"):
            continue

        name = file.replace(".csv", "")

        if "ideal" in name:
            participant = name.replace("_ideal", "").replace("ideal_", "")
            participants.setdefault(participant, {})["ideal"] = file

        elif "perturbed" in name:
            participant = name.replace("_perturbed", "").replace("perturbed_", "")
            participants.setdefault(participant, {})["perturbed"] = file

    return participants


# ----------------------------
# Main update logic
# ----------------------------
def update_leaderboard(scores_file=None):
    entries = []

    truth = load_private_labels()
    files = os.listdir(SUBMISSIONS_DIR)
    grouped = group_submissions(files)

    for participant, subs in grouped.items():

        ideal_score = "N/A"
        pert_score = "N/A"
        gap = "N/A"

        if truth is not None:
            try:
                if "ideal" in subs:
                    ideal_score = score_submission(
                        os.path.join(SUBMISSIONS_DIR, subs["ideal"]), truth
                    )

                if "perturbed" in subs:
                    pert_score = score_submission(
                        os.path.join(SUBMISSIONS_DIR, subs["perturbed"]), truth
                    )

                if isinstance(ideal_score, float) and isinstance(pert_score, float):
                    gap = round(ideal_score - pert_score, 4)

            except Exception as e:
                print(f"Error scoring {participant}: {e}")

        entries.append({
            "participant": participant,
            "f1_ideal": ideal_score,
            "f1_perturbed": pert_score,
            "gap": gap,
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        })

    write_leaderboard(entries)


# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=str, default=None)
    args = parser.parse_args()
    update_leaderboard(scores_file=args.scores)
