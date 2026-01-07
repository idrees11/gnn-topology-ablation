# leaderboard/leaderboard_system.py
import os
import io
import base64
import pandas as pd
from datetime import datetime
from sklearn.metrics import f1_score

SUBMISSIONS_DIR = "submissions"
LEADERBOARD_FILE = "leaderboard/leaderboard.md"

os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
os.makedirs("leaderboard", exist_ok=True)

def load_private_labels():
    b64 = os.getenv("TEST_LABELS_B64")
    if not b64:
        print("Private labels missing. Scores will be N/A")
        return None
    csv_data = base64.b64decode(b64).decode("utf-8")
    df = pd.read_csv(io.StringIO(csv_data))
    df.columns = df.columns.str.strip()
    if "target" not in df.columns and "label" in df.columns:
        df["target"] = df["label"]
    return df

def score_submission(file, truth):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    pred_col = "label" if "label" in df.columns else "target"
    y_true = truth["target"]
    y_pred = df[pred_col]
    return round(f1_score(y_true, y_pred, average="macro"), 4)

def update_leaderboard():
    truth = load_private_labels()
    leaderboard = []

    for f in os.listdir(SUBMISSIONS_DIR):
        if not f.endswith(".csv"): continue
        participant = f.replace(".csv", "")
        path = os.path.join(SUBMISSIONS_DIR, f)
        try:
            score = score_submission(path, truth) if truth is not None else "N/A"
        except Exception as e:
            print(f"ERROR scoring {f}: {e}")
            score = "Error"
        leaderboard.append({
            "participant": participant,
            "f1_score": score,
            "submission": f,
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        })

    leaderboard_sorted = sorted(
        leaderboard,
        key=lambda x: float(x["f1_score"]) if isinstance(x["f1_score"], (int, float)) else -1,
        reverse=True
    )

    with open(LEADERBOARD_FILE, "w") as f:
        f.write("# 🏆 GNN Challenge Leaderboard\n\n")
        f.write("| Rank | Participant | F1 Score | Submission | Timestamp |\n")
        f.write("|------|------------|----------|------------|-----------|\n")
        for i, e in enumerate(leaderboard_sorted, start=1):
            f.write(f"| {i} | {e['participant']} | {e['f1_score']} | {e['submission']} | {e['timestamp']} |\n")

    print(f"Leaderboard updated → {LEADERBOARD_FILE}")

if __name__ == "__main__":
    update_leaderboard()
