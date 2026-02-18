import os
import json
import pandas as pd
from datetime import datetime

SUBMISSIONS_DIR = "submissions"
LEADERBOARD_DIR = "leaderboard"
LEADERBOARD_MD = os.path.join(LEADERBOARD_DIR, "leaderboard.md")
LEADERBOARD_HISTORY = os.path.join(LEADERBOARD_DIR, "leaderboard_history.csv")

os.makedirs(LEADERBOARD_DIR, exist_ok=True)


# -------------------------------------------------
# Load full history safely
# -------------------------------------------------
def load_history():
    if os.path.exists(LEADERBOARD_HISTORY):
        return pd.read_csv(LEADERBOARD_HISTORY)
    return pd.DataFrame(columns=[
        "participant",
        "f1_ideal",
        "f1_perturbed",
        "robustness_gap",
        "timestamp"
    ])


# -------------------------------------------------
# Append new score entries safely
# -------------------------------------------------
def append_scores(entries):
    history_df = load_history()
    new_df = pd.DataFrame(entries)
    history_df = pd.concat([history_df, new_df], ignore_index=True)
    history_df.to_csv(LEADERBOARD_HISTORY, index=False)
    print("History updated →", LEADERBOARD_HISTORY)
    return history_df


# -------------------------------------------------
# Keep BEST score per participant
# Criteria:
# 1) Highest perturbed score
# 2) If tie → latest submission wins
# -------------------------------------------------
def get_best_scores(df):
    if df.empty:
        return df

    df = df.copy()

    for col in ["f1_ideal", "f1_perturbed", "robustness_gap"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df_best = df.sort_values(
        by=["f1_perturbed", "timestamp"],
        ascending=[False, False]
    ).drop_duplicates(
        subset=["participant"],
        keep="first"
    )

    return df_best.sort_values(by="f1_perturbed", ascending=False)


# -------------------------------------------------
# Write leaderboard markdown
# -------------------------------------------------
def write_leaderboard_markdown(history_df):

    # Ensure chronological history
    history_df = history_df.sort_values(by="timestamp")

    best_df = get_best_scores(history_df)

    with open(LEADERBOARD_MD, "w", encoding="utf-8") as f:
        f.write("# 🏆 GNN (Topology Ablation) Robustness Challenge Leaderboard\n\n")
        f.write("Best submission per participant based on perturbed performance.\n\n")

        f.write("| Rank | Participant | F1 Ideal | F1 Perturbed | Robustness Gap | Timestamp |\n")
        f.write("|------|------------|----------|--------------|----------------|-----------|\n")

        if best_df.empty:
            f.write("| - | - | - | - | - | - |\n")
        else:
            for i, row in enumerate(best_df.itertuples(index=False), start=1):
                f.write(
                    f"| {i} | {row.participant} | "
                    f"{row.f1_ideal} | {row.f1_perturbed} | "
                    f"{row.robustness_gap} | {row.timestamp} |\n"
                )

        f.write("\n---\n")
        f.write("### 📜 Submission History\n\n")
        f.write("| Participant | F1 Ideal | F1 Perturbed | Gap | Timestamp |\n")
        f.write("|------------|----------|--------------|-----|-----------|\n")

        for row in history_df.itertuples(index=False):
            f.write(
                f"| {row.participant} | {row.f1_ideal} | "
                f"{row.f1_perturbed} | {row.robustness_gap} | {row.timestamp} |\n"
            )

    print("Leaderboard markdown updated →", LEADERBOARD_MD)


# -------------------------------------------------
# Update leaderboard from scores.json
# -------------------------------------------------
def update_leaderboard(scores_file):

    if not scores_file or not os.path.exists(scores_file):
        print("No scores.json found")
        return

    with open(scores_file, "r") as f:
        scores = json.load(f)

    new_entries = []

    for s in scores:
        entry = {
            "participant": s.get("participant", "unknown"),
            "f1_ideal": s.get("f1_ideal", "N/A"),
            "f1_perturbed": s.get("f1_perturbed", "N/A"),
            "robustness_gap": s.get("robustness_gap", "N/A"),
            "timestamp": s.get(
                "timestamp",
                datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            )
        }
        new_entries.append(entry)

    history_df = append_scores(new_entries)
    write_leaderboard_markdown(history_df)


# -------------------------------------------------
# CLI
# -------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=str, default=None)
    args = parser.parse_args()

    update_leaderboard(args.scores)
