---------------------------------------
🧠 GNN Topology Robustness Challenge
---------------------------------------

This repository hosts the official evaluation system for the GNN robustness challenge.
Participants submit predictions for ideal and perturbed topology settings.

All submissions are encrypted, automatically evaluated, and ranked on a public leaderboard.

Repository hosted on GitHub.

---------------
🎯 Objective
---------------

Participants must generate predictions for two settings: 
```
✅ Ideal graph topology
✅ Perturbed graph topology
```
--------------------
Evaluation metrics:
-------------------

F1 Score (Ideal)

F1 Score (Perturbed)

Robustness Gap = |Ideal − Perturbed|

🏁 Ranking Priority
```
1️⃣ Highest Perturbed F1 Score
2️⃣ Lowest Robustness Gap
3️⃣ Most recent submission
```
--------------------------
📂 Repository Structure
--------------------------
```
gnn-topology-ablation/
│
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── scoring_script.py              # Computes F1 scores and robustness gap
├── leaderboard_system.py          # Leaderboard update engine
├── scores.json                    # Temporary scoring output (auto-generated)
│
├── submissions/                   # Participant encrypted submissions
│
├── starter_code/                  # Starter implementation for participants
│
├── data/                          # Evaluation dataset
│   └── TUDataset/
│       └── MUTAG/
│
├── leaderboard/                   # Public leaderboard outputs
│   ├── leaderboard.md
│   └── leaderboard_history.csv
│
├── keys/                          # Encryption keys
│   └── public_key.pem             # Organiser RSA public key
│
├── .github/
│   └── workflows/
│       └── score_submission.yml   # Automated scoring pipeline
│
├── readme                         # Additional documentation
├── train.csv                      # Training data reference
└── test.csv                       # Test data reference
```
--------------------
⚙️ Getting Started
--------------------

Clone the repository:

git clone https://github.com/idrees11/gnn-topology-ablation.git
cd gnn-topology-ablation


Install dependencies:

pip install -r requirements.txt


Generate prediction files:

submissions/ideal_submission.csv
submissions/perturbed_submission.csv

---------------------------------------------------
🔐 Secure Submission Format (AES + RSA Encryption)
---------------------------------------------------

All prediction files must be encrypted before submission.

Encryption design:

✔ Prediction files encrypted using AES-256
✔ AES key encrypted using RSA public key
✔ Only organiser can decrypt submissions

Public key provided in:

keys/public_key.pem


Private key is securely stored by organiser and never shared.

----------------------
📦 Files to Submit
---------------------

Your Pull Request must contain ONLY:

submissions/ideal_submission.enc
submissions/perturbed_submission.enc
submissions/aes_key.enc

❌ Do NOT upload

Raw CSV files

AES key .hex files

Unencrypted predictions

-----------------------------------
🧩 Encryption Steps (Run Exactly)
-----------------------------------

**🔹 Step 1 — Generate AES key**

openssl rand -hex 32 > submissions\aes_key.hex

**🔹 Step 2 — Encrypt CSV files using AES key**

**Encrypt ideal predictions:**

openssl enc -aes-256-cbc -pbkdf2 -in submissions\ideal_submission.csv -out submissions\ideal_submission.enc -pass file:submissions\aes_key.hex


**Encrypt perturbed predictions:**

openssl enc -aes-256-cbc -pbkdf2 -in submissions\perturbed_submission.csv -out submissions\perturbed_submission.enc -pass file:submissions\aes_key.hex

**🔹 Step 3 — Encrypt AES key using organiser RSA public key**

openssl pkeyutl -encrypt -pubin -inkey keys\public_key.pem -in submissions\aes_key.hex -out submissions\aes_key.enc


**If multiple AES keys are used:**

openssl pkeyutl -encrypt -pubin -inkey keys\public_key.pem -in submissions\aes_key_perturbed.hex -out submissions\aes_key_perturbed.enc

-------------------------
🚀 Submission Procedure
-------------------------

1️⃣ Fork the repository
2️⃣ Place encrypted files inside submissions/
3️⃣ Create a new branch
4️⃣ Commit ONLY .enc files
5️⃣ Open a Pull Request

Submissions are evaluated automatically.

----------------------------------
🤖 Automated Evaluation Pipeline
----------------------------------

When a Pull Request is opened:

1️⃣ AES key is decrypted using organiser private RSA key
2️⃣ Prediction files are decrypted
3️⃣ Evaluation metrics are computed
4️⃣ Scores are written to scores.json
5️⃣ Leaderboard is updated automatically

Participants never see decrypted predictions.

-----------------------
🏆 Leaderboard System
-----------------------

Leaderboard is generated by:

leaderboard_system.py


It maintains:

✔ Full submission history
✔ Best score per participant
✔ Public ranking

Generated outputs:

leaderboard/leaderboard.md
leaderboard/leaderboard.json
leaderboard/leaderboard_history.csv

📊 Leaderboard Ranking Logic

For each submission the system records:

Participant name

F1 Ideal

F1 Perturbed

Robustness Gap

Timestamp

Best submission per participant is selected using:

Sort priority:
1) Highest perturbed score
2) Lowest robustness gap
3) Latest timestamp

----------------------
🔒 Security Guarantee
---------------------

✔ Predictions encrypted locally
✔ AES key encrypted using RSA public key
✔ Only organiser can decrypt
✔ Files visible but unreadable
✔ Ensures blind evaluation

----------------
📜 License
----------------

Released under the MIT License.
