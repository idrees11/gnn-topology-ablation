--------------------------------------------------------------------------------------
**🧠 GNN Challenge: Graph Classification with Topological Features (Ablation Study)**
--------------------------------------------------------------------------------------
------------------------------------
****🎯 Challenge Overview****
-----------------------------------

Welcome to the Graph Neural Networks (GNN) Graph Classification Challenge!

This competition focuses on graph-level classification using message-passing neural networks (MPNNs) with an emphasis on topological (structural) feature augmentation.

Participants are expected to design models that combine:

Node features

Graph structure

Structural / topological descriptors

to improve classification performance.

The challenge uses the MUTAG dataset, which is small but non-trivial, requiring careful use of topological features for strong results.

-----------------------
**🧩 Problem Description**
-----------------------

Predict a graph-level class label for each molecular graph.

You are given:

The graph topology (nodes & edges)

Basic node features

Your goal:

Build a GNN that effectively leverages graph topology, especially via structural/topological node feature augmentation.
-------------------------
**Dataset Description**
------------------------
This competition uses the MUTAG molecular graph dataset for binary graph classification.
Each sample is a graph representing a molecule:

Nodes → atoms

Edges → chemical bonds

Label → class of the molecule

The task is to predict the correct class for each molecular graph using Graph Neural Networks.
Dataset Availability

The dataset is provided directly in the competition repository.

**Participants receive:**

Graph structure (edge connections)

Node features (baseline atom features)

Files for evaluation on clean and perturbed graphs

No external data or pretrained models are allowed to ensure fair comparison of methods.
Standard Node Features (Baseline)

**Each node initially contains chemical identity information only:**

One-hot encoding of atom type

Carbon

Nitrogen

Oxygen

Fluorine

Chlorine

Bromine

Iodine

**Topological Features (Structural Augmentation)**

Examples include:

Node degree

Clustering coefficient

Centrality measures

PageRank

Other structural properties derived from graph connectivity
-------------------------------
**Key Idea of the Competition**
-----------------------------
The study evaluates:
👉 How structural (topological) information improves graph classification
👉 How robust models remain when features are slightly perturbed
👉 The performance gap between standard features vs structure-aware representations


**🔹 Evaluation Conditions**
------------------------

The test set is evaluated under two conditions:

Condition	Description	Submission File
```
🟢 Ideal	Graphs are clean, with no perturbations. Node features exactly as computed.	submissions/ideal_submission.csv
🔴 Perturbed	Graphs have modified node features to simulate realistic distribution shifts:
```
• Feature shift: +0.3
• Gaussian noise: N(0,0.05^2)

Edges remain unchanged.	submissions/perturbed_submission.csv

⚠️ Participants must submit predictions for both conditions.
Failing to submit either file will result in a score of N/A for that condition.

Robustness gap:

robustness_gap = F1_ideal - F1_perturbed


Smaller gaps indicate more robust models.

----------------
**🧠 Problem Type**
----------------

Graph Classification

Supervised Learning

Binary Classification

---------------------------
**📚 Relevant GNN Concepts**
---------------------------

Use concepts from DGL Lectures 1.1–4.6:

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

-----------
**📦 Dataset**
-----------

Dataset: MUTAG (from TUDataset)

Graphs: 188 molecular graphs

Classes: 2 (binary)

Average nodes per graph: ~17

Edges: Undirected

Source: Automatically downloaded from TUDataset

----------------
**🗂️ Data Splits**
----------------
Split	Percentage
Train	70%
Validation	10%
Test	20%

Files in data/:

train.csv → graph indices + labels

test.csv → graph indices only (labels hidden)

⚠️ Test labels are hidden and used only by the organisers for evaluation.

-----------------------
**📊 Evaluation Metric**
-----------------------

Primary Metric: Macro F1-score

f1_score(y_true, y_pred, average="macro")


Why Macro F1?

Sensitive to class imbalance

Encourages balanced performance across classes

Difficult to optimize directly

Official leaderboard metric

-----------------
**⚙️ Constraints**
-----------------

❌ No external datasets

❌ No pretraining

❌ No handcrafted features beyond allowed topology features

✅ Only methods covered in DGL Lectures 1.1–4.6

⏱ Models must run within 10 minutes on CPU

✅ Any GNN architecture allowed (GIN, GCN, GraphSAGE, etc.)

--------------------
**🚀 Getting Started**
--------------------

1️⃣ Install Dependencies

**pip install -r requirements.txt**


2️⃣ Run the Baseline Model

**cd starter_code**

**python baseline.py**


This will:

Train a GIN baseline on ideal training data

Generate predictions for both ideal & perturbed test sets

Save submission files to submissions/

```
**Important:**  
- Participants should **develop or modify their GNN model only in `model.py`**.  
- **Do not modify** `baseline.py`, `dataset.py`, or `scoring_script.py`.  
- The `baseline.py` script will handle:  
  - Training the model on **ideal data**  
  - Generating predictions for **ideal** and **perturbed** test sets  
  - Saving submission files in `submissions/`  
- Your model must accept the inputs as defined in `baseline.py` and output predictions compatible with the pipeline.

> In short: **Develop your model in `model.py` only. Everything else is handled by the baseline and organiser pipeline.**
```
---------------------
**📤 Submission Format**
---------------------

Participants must submit two CSV files:

Ideal predictions (ideal_submission.csv)

```
**graph_index,target**
0,1
1,0
2,1
...
```

Perturbed predictions (perturbed_submission.csv)

```
**graph_index,target**
0,1
1,1
2,0
...
```

Columns:

graph_index → Index of the graph

target → Predicted class label (0 or 1)

-----------------------------------
**🏆 Leaderboard & Submission Notes**
-----------------------------------

Submissions are collected, but scores are NOT displayed immediately.

Leaderboard updates are controlled by organisers using hidden test labels stored securely via GitHub Secrets.

After submission, you will only see:

✅ Submission successful

Scores, robustness gap, and rank are updated later by organisers.

Leaderboard file: leaderboard/leaderboard.md

Score calculation:

Macro F1-score for ideal and perturbed

Robustness gap = F1_ideal - F1_perturbed

-----------------------------
**💡 Tips for Success**
-----------------------------

Structural features are crucial

Experiment with different topological descriptors

Regularization is important for small datasets

Simpler models often generalize better

GIN + structural features is a strong baseline
-------------------------
**📁 Repository Structure**
--------------------------
```
gnn-topology-ablation/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── starter_code/
│   ├── dataset.py
│   ├── model.py
│   ├── baseline.py
│   
│
├── submissions/
│   └── sample_submission.csv
│
├── scoring_script.py
│
├── leaderboard/
│   └── leaderboard.md
└── requirements.txt
│
└── README.md
```

--------------------------
🏁 Step-by-Step Commands
-------------------------

1. **Enter starter code**

    cd starter_code

2. **Run baseline model**

    python baseline.py

3. **Return to repository root**

    cd ..

4. **Verify submission files**

    ls submissions or Dir submissions

5. (Optional) Local scoring (organisers only)

    python scoring_script.py --participant "YourTeamName"

----------------
📬 Contact
----------------

For questions or clarifications, open a GitHub Issue in this repository. Or you can contact me at, idrees11@yahoo.com (+918123434057)

-------------------

**📜 License**

-------------------

This project is released under the MIT License.
See the LICENSE file for details.
