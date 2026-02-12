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
# Constants
# ----------------------------
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
# Score helper
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
        f.write("# 🏆 GNN (Topology Ablation) Robustness Challenge Leaderboard\n\n")
        f.write("| Rank | Participant | F1 Ideal | F1 Perturbed | Robustness Gap | Timestamp |\n")
        f.write("|------|------------|----------|--------------|----------------|-----------|\n")

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
                f"{e['f1_perturbed']} | {e['robustness_gap']} | {e['timestamp']} |\n"
            )

    print(f"Leaderboard updated → {LEADERBOARD_FILE}")

# ----------------------------
# Main leaderboard update
# ----------------------------
def update_leaderboard(scores_file=None):
    leaderboard = []

    # ----------------------------
    # PR-safe JSON mode (from scoring_script)
    # ----------------------------
    if scores_file and os.path.exists(scores_file):
        with open(scores_file, "r") as f:
            scores = json.load(f)

        for s in scores:
            # Ensure all fields exist
            leaderboard.append({
                "participant": s.get("participant", "unknown"),
                "f1_ideal": s.get("f1_ideal", "N/A"),
                "f1_perturbed": s.get("f1_perturbed", "N/A"),
                "robustness_gap": s.get("robustness_gap", "N/A"),
                "timestamp": s.get("timestamp", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"))
            })

    # ----------------------------
    # Direct scoring mode (fallback)
    # ----------------------------
    else:
        truth = load_private_labels()

        ideal_path = os.path.join(SUBMISSIONS_DIR, "ideal_submission.csv")
        perturbed_path = os.path.join(SUBMISSIONS_DIR, "perturbed_submission.csv")

        if truth is not None and os.path.exists(ideal_path) and os.path.exists(perturbed_path):
            f1_ideal = score_submission(ideal_path, truth)
            f1_perturbed = score_submission(perturbed_path, truth)
            gap = round(f1_ideal - f1_perturbed, 4)

            leaderboard.append({
                "participant": "local",
                "f1_ideal": f1_ideal,
                "f1_perturbed": f1_perturbed,
                "robustness_gap": gap,
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            })

    write_leaderboard(leaderboard)

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores",
        type=str,
        default=None,
        help="Optional path to scores.json to update leaderboard PR-safe"
    )
    parser.add_argument(
        "--participant",
        type=str,
        default="unknown",
        help="Participant name to record in leaderboard"
    )
    args = parser.parse_args()

    # If scores.json exists, inject participant name if missing
    if args.scores and os.path.exists(args.scores):
        with open(args.scores, "r") as f:
            data = json.load(f)
        for d in data:
            if "participant" not in d or d["participant"] == "unknown":
                d["participant"] = args.participant
        with open(args.scores, "w") as f:
            json.dump(data, f, indent=4)

    update_leaderboard(scores_file=args.scores)
