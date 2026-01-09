🧠 GNN Challenge: Graph Classification with Topological Features
Overview

Welcome to the Graph Neural Networks (GNN) Challenge!

This competition focuses on graph-level classification using message-passing neural networks (MPNNs), with a strong emphasis on topological (structural) feature augmentation. Participants are expected to design models that effectively combine node features, graph structure, and structural descriptors to improve classification performance.

The challenge is small, fast, and non-trivial, and can be fully solved using concepts covered in DGL Lectures 1.1–4.6:
👉 https://www.youtube.com/watch?v=gQRV_jUyaDw&list=PLug43ldmRSo14Y_vt7S6vanPGh-JpHR7T

🎯 Problem Statement

Given a graph

𝐺
=
(
𝑉
,
𝐸
)
G=(V,E)

predict its graph-level class label.

Each graph represents a molecular structure from the MUTAG dataset

Basic node features are provided

The main challenge is to leverage graph topology effectively

🧩 Problem Type

Graph Classification

Supervised Learning

Binary Classification

📚 Relevant GNN Concepts (DGL 1.1–4.6)

This challenge can be solved using:

Message Passing Neural Networks (MPNNs)

Graph Isomorphism Networks (GIN)

Neighborhood aggregation

Graph-level readout (e.g., global mean pooling)

Structural / Topological Node Features

You are encouraged to experiment with:

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

The dataset is small enough for fast experimentation, yet rich enough to benefit from structural features.

🗂️ Data Splits

A fixed random seed is used to ensure fair comparison.

Split	Percentage
Train	70%
Validation	10%
Test	20%

Files in data/:

train.csv → graph indices + labels

test.csv → graph indices only (labels hidden)

⚠️ Test labels are hidden and used only by the organisers for scoring.

📊 Evaluation Metric
Macro F1-score
f1_score(y_true, y_pred, average="macro")


Why Macro F1?

Sensitive to class imbalance

Encourages balanced performance across classes

Difficult to optimize directly

Used as the official leaderboard metric

⚙️ Constraints

To keep the competition fair and focused:

❌ No external datasets

❌ No pretraining

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

Train a simple GIN baseline

Generate predictions on the test set

Save a submission file to:

submissions/sample_submission.csv

📤 Submission Format

Submissions must be CSV files with the following format:

graph_index,target
0,1
1,0
2,1
...


graph_index: Index of the graph in the dataset

target: Predicted class label (0 or 1)

🧪 Scoring

Submissions are evaluated using hidden test labels:

f1_score(y_true, y_pred, average="macro")


Scores are computed automatically by the organiser’s scoring pipeline.

🏆 Leaderboard

Ranked by Macro F1-score (higher is better)

Ties are broken by submission time

Leaderboard is maintained in:

leaderboard/leaderboard.md

💡 Tips for Success

Structural features matter more than you think

Try different combinations of topological descriptors

Regularization is crucial for small datasets

Simpler models often generalize better

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
├── leaderboard/
│   └── leaderboard.md
└── README.md

🏁 Step-by-Step Commands
# 1️⃣ Enter starter code directory
cd starter_code

# 2️⃣ Run baseline model
python baseline.py

# 3️⃣ Return to repository root
cd ..

# 4️⃣ Verify submission file
dir submissions

# 5️⃣ (Optional) Local scoring (organisers only)
python scoring_script.py submissions/sample_submission.csv

📬 Contact

For questions or clarifications, please open a GitHub Issue.

📜 License

This project is released under the MIT License.
