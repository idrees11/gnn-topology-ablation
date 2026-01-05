🧠 GNN Challenge: Graph Classification with Topological Features
Overview
    
Welcome to the Graph Neural Networks (GNN) Challenge!
This competition focuses on graph classification using message-passing neural networks, with an emphasis on topological (structural) feature augmentation. Participants are challenged to design models that effectively combine node features, graph structure, and topological descriptors to improve classification performance.
    
The challenge is small, fast, yet non-trivial, and can be fully solved using methods covered in DGL Lectures 1.1–4.6 (https://www.youtube.com/watch?v=gQRV_jUyaDw&list=PLug43ldmRSo14Y_vt7S6vanPGh-JpHR7T).

🎯 Problem Statement
       
Given a graph 
𝐺
=
(
𝑉
,
𝐸
)
G=(V,E), predict its graph-level class label.

Each graph represents a molecular structure from the MUTAG dataset.

Basic node features are provided, but the main challenge is to leverage graph topology effectively.

🧩 Problem Type

Graph Classification

Supervised Learning

Binary Classification

📚 Relevant GNN Concepts (DGL 1.1–4.6)

This challenge can be solved using:

Message Passing Neural Networks (MPNNs)

Graph Isomorphism Networks (GIN)

Neighborhood aggregation

Graph-level readout (global mean pooling)

Structural / Topological Node Features:

Degree

Clustering Coefficient

Betweenness Centrality

PageRank

k-core number

📦 Dataset

Dataset: MUTAG (TUDataset)

Graphs: 188 molecular graphs

Classes: 2 (binary classification)

Nodes per graph: ~17 (average)

Edges: Undirected

Source: Automatically downloaded from TUDataset

Small enough for quick experimentation, but rich enough to benefit from structural features.

🗂️ Data Splits

The dataset is split once using a fixed random seed to ensure fair comparison:

Split	Percentage
Train	70%
Validation	10%
Test	20%

Files provided in data/:

train.csv → graph indices + labels

test.csv → graph indices only (labels hidden)

⚠️ Test labels are hidden and used only for scoring by organizers.

📊 Objective Metric

Macro F1-score

Why Macro F1?

Sensitive to class imbalance

Encourages balanced performance across classes

Difficult to optimize directly

This is the official ranking metric.

⚙️ Constraints

To keep the competition fair and focused:

❌ No external datasets

❌ No pretraining

✅ Only methods covered in DGL Lectures 1.1–4.6

⏱ Models must run within 10 minutes on CPU

✅ Any GNN architecture is allowed (GIN, GCN, GraphSAGE, etc.)

🚀 Getting Started
1️⃣ Install Dependencies
pip install -r starter_code/requirements.txt

2️⃣ Run the Baseline Model
cd starter_code
python baseline.py


This will:

Train a simple GIN model

Generate predictions on the test set

Save a submission file to submissions/sample_submission.csv

📤 Submission Format

Your submission must be a CSV file with the following format:

graph_index,target
0,1
1,0
2,1
...


graph_index: Index of the graph in the dataset

target: Predicted class label (0 or 1)

🧪 Scoring

Submissions are evaluated using:

f1_score(y_true, y_pred, average="macro")


Scores are computed using a hidden test label file.

🏆 Leaderboard

Submissions are ranked by Macro F1-score (higher is better)

Ties are broken by submission time

Leaderboard is maintained in: leaderboard.md

💡 Tips for Success

Structural features matter more than you think

Experiment with different combinations of topological features

Regularization is important for small datasets

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
├── leaderboard.md
└── README.md

🏁 Step-by-Step Commands
# 1️⃣ Go to starter_code folder
cd starter_code

# 2️⃣ Run the baseline to generate submission
python baseline.py

# 3️⃣ Go back to repo root
cd ..

# 4️⃣ Check that submission exists
dir submissions

# 5️⃣ Score the submission
python scoring_script.py submissions\sample_submission.csv


What Each Step Does:

cd starter_code → enters folder with baseline.py

python baseline.py → trains the model and saves submission CSV

cd .. → returns to repo root

dir submissions → verifies CSV presence

python scoring_script.py ... → computes and prints F1-score

📬 Contact

For questions or clarifications, please open a GitHub Issue.

Good luck — and happy graph learning! 🧠📊

📜 License

This project is released under the MIT License.
