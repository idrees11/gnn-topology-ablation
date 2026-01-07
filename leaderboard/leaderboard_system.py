"""
leaderboard_system.py
Safely updates leaderboard/leaderboard.md using submissions.
Handles PRs/forks where private labels may not be available.
Replaces dummy names and sorts participants by F1-score descending.
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
# Ensure folders exist
# ----------------------------
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
os.makedirs(LEADERBOARD_DIR, exist_ok=True)

# ----------------------------
# Load private labels from TEST_LABELS_B64
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
        # Ensure target column exists
        if "target" not in truth.columns:
            if "label" in truth.columns:
                truth["target"] = truth["label"]
            else:
                raise ValueError("Private labels CSV must contain 'label' or 'target' column")
        # Ensure graph_index column exists
        if "graph_index" not in truth.columns:
            raise ValueError("Private labels CSV must contain 'graph_index' column")
        return truth
    except Exception as e:
        print(f"ERROR decoding TEST_LABELS_B64: {e}")
        return None

# ----------------------------
# Score a single submission
# ----------------------------
def score_submission(sub_path, truth):
    # Auto-detect separator
    df = pd.read_csv(sub_path, sep=None, engine="python")
    df.columns = df.columns.str.strip()

    # Prediction column
    pred_col = "label" if "label" in df.columns else "target"
    if pred_col not in df.columns:
        raise ValueError(f"Submission {sub_path} missing 'label' or 'target' column")

    # ID column
    id_col = "graph_index"
    if id_col not in df.columns:
        raise ValueError(f"Submission {sub_path} missing '{id_col}' column")

    # Convert IDs to int for safe merge
    df[id_col] = pd.to_numeric(df[id_col], errors="coerce")
    truth[id_col] = pd.to_numeric(truth[id_col], errors="coerce")

    df_clean = df.dropna(subset=[id_col]).copy()
    truth_clean = truth.dropna(subset=[id_col]).copy()

    df_clean[id_col] = df_clean[id_col].astype(int)
    truth_clean[id_col] = truth_clean[id_col].astype(int)

    # Merge on graph_index
    merged = truth_clean.merge(df_clean, on=id_col, suffixes=("_true", "_pred"))
    if merged.empty:
        print(f"Skipping {sub_path}: no matching IDs")
        return "Error"

    # Compute macro F1
    return round(f1_score(
        merged["target_true"], 
        merged[f"{pred_col}_pred"], 
        average="macro"
    ), 4)

# ----------------------------
# Write leaderboard markdown
# ----------------------------
def write_leaderboard(entries):
    with open(LEADERBOARD_FILE, "w") as f:
        f.write("# 🏆 GNN Challenge Leaderboard\n\n")
        f.write("| Rank | Participant | F1 Score | Submission | Timestamp |\n")
        f.write("|------|------------|----------|------------|-----------|\n")

        if not entries:
            f.write("| - | - | - | - | - |\n")
            print("Leaderboard is empty (no submissions).")
            return

        # Sort numeric F1 descending; put N/A/errors last
        def sort_key(x):
            try:
                return float(x["f1_score"])
            except:
                return -1  # N/A/Error go last

        entries_sorted = sorted(entries, key=sort_key, reverse=True)

        for i, e in enumerate(entries_sorted, start=1):
            f.write(
                f"| {i} | {e['participant']} | {e['f1_score']} | "
                f"{e['submission']} | {e['timestamp']} |\n"
            )

    print(f"Leaderboard updated → {LEADERBOARD_FILE}")

# ----------------------------
# Main update function
# ----------------------------
def update_leaderboard():
    leaderboard = []

    truth = load_private_labels()

    # Iterate over submissions
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
