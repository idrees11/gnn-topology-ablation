🧠 GNN Challenge: Graph Classification with Topological Features
🎯 Challenge Overview

Welcome to the Graph Neural Networks (GNN) Graph Classification Challenge!

This competition focuses on graph-level classification using message-passing neural networks (MPNNs) with an emphasis on topological (structural) feature augmentation.

Participants are expected to design models that combine:

Node features

Graph structure

Structural / topological descriptors

to improve classification performance.

The challenge uses the MUTAG dataset, which is small but non-trivial, requiring careful use of topological features for strong results.

🧩 Problem Description

Predict a graph-level class label for each molecular graph.

You are given:

The graph topology (nodes and edges)

Basic node features

Your goal is to build a GNN that effectively leverages graph topology, especially via structural/topological node feature augmentation.

🔹 Evaluation Conditions

The test set is evaluated under two conditions:

Ideal Condition

Graphs are clean, with no perturbations.

Node features are exactly as computed from the original graph topology.

Submission file:

submissions/ideal_submission.csv


Perturbed Condition

Graphs have modified node features to simulate realistic distribution shifts:

Feature shift: all node features are increased by a constant (+0.3)

Gaussian noise: additive noise applied to node features (N(0, 0.05^2))

Node topology (edges) remains unchanged.

Submission file:

submissions/perturbed_submission.csv


⚠️ Participants must submit predictions for both conditions.
Failing to submit either file will result in a score of N/A for that condition.

Robustness gap is computed as:

robustness_gap = F1_ideal - F1_perturbed


Smaller gaps indicate more robust models.

🧠 Problem Type

Graph Classification

Supervised Learning

Binary Classification

📚 Relevant GNN Concepts

Participants can use concepts from DGL Lectures 1.1–4.6:

Message Passing Neural Networks (MPNNs)

Graph Isomorphism Networks (GIN)

Neighborhood aggregation

Graph-level readout (global mean pooling)

Structural / Topological Node Features

Topological descriptors to experiment with:

Node degree

Clustering coefficient

Betweenness centrality

PageRank

k-core number

📦 Dataset

Dataset: MUTAG (from TUDataset)

Graphs: 188 molecular graphs

Classes: 2 (binary)

Average nodes per graph: ~17

Edges: Undirected

Source: Automatically downloaded from TUDataset

🗂️ Data Splits
Split	Percentage
Train	70%
Validation	10%
Test	20%

Files in data/:

train.csv → graph indices + labels

test.csv → graph indices only (labels hidden)

⚠️ Test labels are hidden and used only by the organisers for evaluation.

📊 Evaluation Metric

Primary Metric: Macro F1-score

f1_score(y_true, y_pred, average="macro")


Why Macro F1?

Sensitive to class imbalance

Encourages balanced performance across classes

Difficult to optimize directly

Official leaderboard metric

⚙️ Constraints

❌ No external datasets

❌ No pretraining

❌ No handcrafted features beyond allowed topology features

✅ Only methods covered in DGL Lectures 1.1–4.6

⏱ Models must run within 10 minutes on CPU

✅ Any GNN architecture allowed (GIN, GCN, GraphSAGE, etc.)

🚀 Getting Started

1️⃣ Install Dependencies

pip install -r starter_code/requirements.txt


2️⃣ Run the Baseline Model

cd starter_code
python baseline.py


This will:

Train a simple GIN model on ideal training data

Generate predictions for both ideal and perturbed test sets

Save submission files to submissions/

📤 Submission Format

Participants must submit two CSV files:

Ideal predictions (ideal_submission.csv)

graph_index,target
0,1
1,0
2,1
...


Perturbed predictions (perturbed_submission.csv)

graph_index,target
0,1
1,1
2,0
...


Columns:

graph_index → Index of the graph in the dataset

target → Predicted class label (0 or 1)

🏆 Leaderboard & Submission Notes

Submissions are collected when you upload your CSV files.

Leaderboard updates are controlled by the organisers and use private test labels stored securely as GitHub Secrets.

Participants will only see a “Submission successful” message; scores and ranks are not displayed immediately.

Example message:

✅ Submission successful


Leaderboard file: leaderboard/leaderboard.md

Scores are computed automatically using macro F1-score for both ideal and perturbed conditions.

Robustness gap = F1_ideal - F1_perturbed

💡 Tips for Success

Structural features are crucial for performance

Experiment with different combinations of topological descriptors

Regularization is important for small datasets

Simpler models often generalize better

GIN + structural features is a strong baseline

📁 Repository Structure
gnn-challenge/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── starter_code/
│   ├── dataset.py
│   ├── model.py
│   ├── baseline.py
│   └── requirements.txt
│
├── submissions/
│   └── sample_submission.csv
│
├── scoring_script.py
│
├── leaderboard/
│   └── leaderboard.md
│
└── README.md

🏁 Step-by-Step Commands
# 1. Enter starter code
cd starter_code

# 2. Run baseline model
python baseline.py

# 3. Return to repository root
cd ..

# 4. Verify submission files
ls submissions

# 5. (Optional) Local scoring (organisers only)
python scoring_script.py --participant "YourTeamName"

📬 Contact

For questions or clarifications, open a GitHub Issue in this repository.

📜 License

This project is released under the MIT License.
See the LICENSE file for details.

✅ This README now clearly covers:

Ideal vs perturbed conditions

Exact perturbations applied

Two required submission files

Organiser-controlled leaderboard workflow

Robustness gap explanation

If you want, I can also add a small diagram showing the baseline pipeline → ideal & perturbed predictions → submission flow, which usually helps participants understand the workflow visually.

Do you want me to create that diagram too?
