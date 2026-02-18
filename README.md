🧠 GNN Topology Robustness Challenge

This repository hosts the evaluation framework for the GNN robustness challenge.
Participants submit predictions for ideal and perturbed topology settings.
Submissions are evaluated automatically and ranked on a public leaderboard.

All submissions must be encrypted to ensure fair evaluation.

Repository hosted on GitHub.

🎯 Objective

Participants must generate predictions for:

Ideal graph topology

Perturbed graph topology

Evaluation metrics:

F1 Score (Ideal)

F1 Score (Perturbed)

Robustness Gap = |Ideal − Perturbed|

Ranking priority:

1️⃣ Highest perturbed F1
2️⃣ Lowest robustness gap
3️⃣ Latest submission

📂 Repository Structure
.github/workflows        → automated scoring pipeline
keys/public_key.pem      → organiser public RSA key
submissions/             → participant encrypted submissions
leaderboard/             → leaderboard outputs
data/                    → evaluation data
scoring_script.py        → evaluation logic
leaderboard_system.py    → leaderboard update system
scores.json              → temporary scoring output

⚙️ Getting Started

Clone the repository:

git clone https://github.com/idrees11/gnn-topology-ablation.git
cd gnn-topology-ablation


Install dependencies:

pip install -r requirements.txt


Generate predictions using your model and save:

submissions/ideal_submission.csv
submissions/perturbed_submission.csv

🔐 Secure Submission Format (AES + RSA)

All prediction files must be encrypted before submission.
This prevents prediction leakage and ensures blind evaluation.

Encryption uses:

AES-256 for prediction files

RSA public key for AES key protection

The organiser provides the public key:

keys/public_key.pem


The private key is stored securely and never shared.

📦 Files to Submit

Your Pull Request must contain ONLY:

submissions/ideal_submission.enc
submissions/perturbed_submission.enc
submissions/aes_key.enc


Do NOT upload:

❌ CSV files
❌ AES key .hex files
❌ Unencrypted predictions

🧩 Encryption Steps (Run Exactly)
Step 1 — Generate AES key
openssl rand -hex 32 > submissions\aes_key.hex

Step 2 — Encrypt CSV files using AES key
openssl enc -aes-256-cbc -pbkdf2 -in submissions\ideal_submission.csv -out submissions\ideal_submission.enc -pass file:submissions\aes_key.hex

openssl enc -aes-256-cbc -pbkdf2 -in submissions\perturbed_submission.csv -out submissions\perturbed_submission.enc -pass file:submissions\aes_key.hex

Step 3 — Encrypt AES key using organiser public key
openssl pkeyutl -encrypt -pubin -inkey keys\public_key.pem -in submissions\aes_key.hex -out submissions\aes_key.enc


If multiple AES keys are used:

openssl pkeyutl -encrypt -pubin -inkey keys\public_key.pem -in submissions\aes_key_perturbed.hex -ou
