import os
import pandas as pd
from sklearn.metrics import f1_score
import base64
import io
import json

PRIVATE_LABELS_ENV = "TEST_LABELS_B64"
SUBMISSIONS_FOLDER = "submissions"
LEADERBOARD_FILE = "leaderboard.csv"
SCORES_JSON_FILE = "scores.json"

EXPECTED_FILES = ["ideal_submission.csv", "perturbed_submission.csv"]

print("Running scoring_script.py...")
print("Files in submissions:", os.listdir(SUBMISSIONS_FOLDER) if os.path.exists(SUBMISSIONS_FOLDER) else "No folder")

# -------------------------------------------------
# Load private labels from GitHub Secret
# -------------------------------------------------
truth = None
labels_b64 = os.getenv(PRIVATE_LABELS_ENV)

if labels_b64:
    decoded = base64.b64decode(labels_b64)
    truth = pd.read_csv(io.BytesIO(decoded))
    truth.columns = truth.columns.str.strip().str.lower()
    print("Loaded private labels successfully")
else:
    print("Private labels unavailable. Scoring skipped.")

truth_col = None
if truth is not None:
    truth_col = "label" if "label" in truth.columns else "target"

scores = []

# -------------------------------------------------
# Evaluate required submissions
# -------------------------------------------------
for fname in EXPECTED_FILES:
    path = os.path.join(SUBMISSIONS_FOLDER, fname)

    if not os.path.exists(path):
        print(f"❌ Missing required submission file: {fname}")
        scores.append({"submission": fname, "f1_score": None})
        continue

    sub = pd.read_csv(path)
    sub.columns = sub.columns.str.strip().str.lower()

    id_col = "graph_index" if "graph_index" in sub.columns else "id"
    pred_col = "label" if "label" in sub.columns else "target"

    merged = truth.merge(sub, on=id_col, suffixes=("_true", "_pred"))

    if merged.empty:
        print(f"❌ No matching IDs for {fname}")
        scores.append({"submission": fname, "f1_score": None})
        continue

    score = f1_score(
        merged[f"{truth_col}_true"],
        merged[f"{pred_col}_pred"],
        average="macro"
    )

    print(f"{fname} → F1: {score:.4f}")
    scores.append({"submission": fname, "f1_score": float(score)})

# -------------------------------------------------
# Save outputs
# -------------------------------------------------
pd.DataFrame(scores).to_csv(LEADERBOARD_FILE, index=False)

with open(SCORES_JSON_FILE, "w") as f:
    json.dump(scores, f, indent=2)

print("Scoring complete.")
