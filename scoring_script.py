import os
import pandas as pd
from sklearn.metrics import f1_score
import base64
import io
import json
import argparse
from datetime import datetime

# ----------------------------
# CLI
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--save-scores", type=str, default="scores.json")
parser.add_argument("--participant", type=str, default="unknown")
args = parser.parse_args()
participant_name = args.participant

# ----------------------------
# Constants
# ----------------------------
PRIVATE_LABELS_ENV = "TEST_LABELS_B64"
SUBMISSIONS_FOLDER = "submissions"
LEADERBOARD_FILE = "leaderboard.csv"
SCORES_JSON_FILE = args.save_scores

EXPECTED_FILES = ["ideal_submission.csv", "perturbed_submission.csv"]

print("Running scoring_script.py...")

# ----------------------------
# Load private labels from GitHub Secret
# ----------------------------
truth = None
labels_b64 = os.getenv(PRIVATE_LABELS_ENV)

if labels_b64:
    try:
        decoded = base64.b64decode(labels_b64)
        truth = pd.read_csv(io.BytesIO(decoded))
        truth.columns = truth.columns.str.strip().str.lower()
        print("✅ Loaded private labels successfully")
    except Exception as e:
        print("❌ Failed to decode private labels:", e)
        truth = None
else:
    print("⚠ Private labels unavailable. Scoring skipped.")

# ----------------------------
# Detect truth column
# ----------------------------
truth_col = None
if truth is not None:
    if "label" in truth.columns:
        truth_col = "label"
    elif "target" in truth.columns:
        truth_col = "target"
    else:
        print("❌ Truth file missing label column")
        truth = None

# ----------------------------
# Evaluate submissions
# ----------------------------
scores = []

if not os.path.exists(SUBMISSIONS_FOLDER):
    print("❌ submissions folder not found")
else:
    print("Files in submissions:", os.listdir(SUBMISSIONS_FOLDER))

    for fname in EXPECTED_FILES:
        path = os.path.join(SUBMISSIONS_FOLDER, fname)

        if not os.path.exists(path):
            print(f"❌ Missing submission: {fname}")
            scores.append({"submission": fname, "f1_score": None})
            continue

        sub = pd.read_csv(path)
        sub.columns = sub.columns.str.strip().str.lower()

        print(f"\nProcessing {fname} | rows={len(sub)}")

        if truth is None:
            print("Scoring skipped (no private labels)")
            scores.append({"submission": fname, "f1_score": None})
            continue

        # Detect ID column
        if "graph_index" in truth.columns and "graph_index" in sub.columns:
            id_col = "graph_index"
        elif "id" in truth.columns and "id" in sub.columns:
            id_col = "id"
        else:
            print(f"❌ ID column mismatch in {fname}")
            scores.append({"submission": fname, "f1_score": None})
            continue

        pred_col = "label" if "label" in sub.columns else "target"

        # Clean ID dtype
        truth[id_col] = pd.to_numeric(truth[id_col], errors="coerce")
        sub[id_col] = pd.to_numeric(sub[id_col], errors="coerce")

        truth_clean = truth.dropna(subset=[id_col]).copy()
        sub_clean = sub.dropna(subset=[id_col]).copy()

        truth_clean[id_col] = truth_clean[id_col].astype(int)
        sub_clean[id_col] = sub_clean[id_col].astype(int)

        merged = truth_clean.merge(
            sub_clean,
            on=id_col,
            suffixes=("_true", "_pred"),
            how="inner"
        )

        print("Merged rows:", len(merged))

        if merged.empty:
            print(f"❌ No matching IDs for {fname}")
            scores.append({"submission": fname, "f1_score": None})
            continue

        y_true_col = f"{truth_col}_true"
        y_pred_col = f"{pred_col}_pred"

        score = f1_score(
            merged[y_true_col],
            merged[y_pred_col],
            average="macro"
        )

        print(f"{fname} → F1: {score:.6f}")

        scores.append({
            "submission": fname,
            "f1_score": round(float(score), 6)
        })

# ----------------------------
# Save leaderboard CSV
# ----------------------------
leaderboard = []

timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

f1_ideal = next((s["f1_score"] for s in scores if s["submission"] == "ideal_submission.csv"), "N/A")
f1_perturbed = next((s["f1_score"] for s in scores if s["submission"] == "perturbed_submission.csv"), "N/A")
robustness_gap = round(f1_ideal - f1_perturbed, 6) if isinstance(f1_ideal, float) and isinstance(f1_perturbed, float) else "N/A"

leaderboard.append({
    "participant": participant_name,
    "f1_ideal": f1_ideal,
    "f1_perturbed": f1_perturbed,
    "robustness_gap": robustness_gap,
    "timestamp": timestamp
})

df_leaderboard = pd.DataFrame(leaderboard)
df_leaderboard.to_csv(LEADERBOARD_FILE, index=False)
print("Leaderboard saved.")

# ----------------------------
# Save scores JSON (for leaderboard_system.py)
# ----------------------------
scores_json = [
    {
        "participant": participant_name,
        "f1_ideal": f1_ideal,
        "f1_perturbed": f1_perturbed,
        "robustness_gap": robustness_gap,
        "timestamp": timestamp
    }
]

with open(SCORES_JSON_FILE, "w") as f:
    json.dump(scores_json, f, indent=2)

print("Scores JSON saved →", SCORES_JSON_FILE)
print("Scoring complete.")
