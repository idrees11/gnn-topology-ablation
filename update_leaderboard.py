import sys
import os
import pandas as pd
from datetime import datetime
from sklearn.metrics import f1_score

"""
Usage:
python update_leaderboard.py submissions/file.csv username
"""

submission_file = sys.argv[1]
username = sys.argv[2]

# Load data
submission = pd.read_csv(submission_file)
truth = pd.read_csv("data/test_labels.csv")

# Compute score
score = f1_score(truth["target"], submission["target"], average="macro")

# Prepare entry
entry = {
    "User": username,
    "Macro F1": round(score, 4),
    "Submission": os.path.basename(submission_file),
    "Timestamp (UTC)": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
}

leaderboard_file = "leaderboard.md"

# Initialize leaderboard if missing
if not os.path.exists(leaderboard_file):
    with open(leaderboard_file, "w") as f:
        f.write("# 🏆 Leaderboard\n\n")
        f.write("| Rank | User | Macro F1 | Submission | Timestamp (UTC) |\n")
        f.write("|------|------|----------|------------|-----------------|\n")

# Load existing leaderboard
df = pd.read_csv(
    leaderboard_file,
    sep="|",
    skipinitialspace=True,
    engine="python"
)

df = df.dropna(axis=1, how="all")
df.columns = [c.strip() for c in df.columns]

if "User" not in df.columns:
    df = pd.DataFrame(columns=entry.keys())

# Append new entry
df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)

# Sort and re-rank
df = df.sort_values("Macro F1", ascending=False).reset_index(drop=True)
df.insert(0, "Rank", range(1, len(df) + 1))

# Write back as markdown
with open(leaderboard_file, "w") as f:
    f.write("# 🏆 Leaderboard\n\n")
    f.write(df.to_markdown(index=False))

