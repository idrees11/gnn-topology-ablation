-------------------------------------------------------------
GNN Challenge: Graph Classification with Topological Features
-------------------------------------------------------------

**Overview**

This competition studies graph classification using GNNs with explicit topological feature augmentation.
Participants design models that combine:

Node features

Graph structure

Structural/topological descriptors

to improve accuracy and robustness.

**Dataset**

We use the **MUTAG molecular** graph dataset.

188 molecular graphs

Binary classification task

Nodes = atoms, edges = chemical bonds

**Official source:**
https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/MUTAG.zip

The dataset is directly accessible via **PyTorch Geometric** using the TUDataset interface.
Baseline code provides loaders and splits.

**Node Representation**

Standard Features (Baseline)

Each node contains a one-hot encoding of atom type:
Carbon, Nitrogen, Oxygen, Fluorine, Chlorine, Bromine, Iodine.

These features capture chemical identity only.

Topological Features (Augmentation)

Nodes may be enriched with structural descriptors such as:

Node degree

Clustering coefficient

Betweenness centrality

PageRank

k-core number

These features encode connectivity and structural role.

**Goal**: learn both chemical identity and structural context.

**Perturbation-Based Robustness Evaluation**

Models are evaluated on two versions of the same graphs:

**Ideal Condition**

Original node features

Unmodified graphs

**Perturbed Condition**

Graph structure unchanged

Node features modified using:

**Feature shift**: +0.3

**Gaussian noise**: N(0, 0.05²)

This simulates realistic feature noise and tests model stability.

**Robustness metric:**

robustness_gap = F1_ideal − F1_perturbed


Smaller gaps indicate more robust models.

Participants must submit predictions for both conditions:

**submissions/ideal_submission.csv**

**submissions/perturbed_submission.csv**
-------------
**Problem Type**
-------------

Graph Classification

Supervised Learning

Binary Classification

---------------------------
**Relevant GNN Concepts**
---------------------------

Use concepts from DGL Lectures 1.1–4.6:

Message Passing Neural Networks (MPNNs)

Graph Isomorphism Networks (GIN)

Neighborhood aggregation

Graph-level readout (global mean pooling)

Structural / Topological Node Features

----------------
**🗂️ Data Splits**
----------------
Split	Percentage
Train	70%
Validation	10%
Test	20%

Files in data/:

**train.csv **→ graph indices + labels

**test.csv** → graph indices only (labels hidden)

⚠️ Test labels are hidden and used only by the organisers for evaluation.

-----------------
**⚙️ Constraints**
-----------------

No external datasets

No pretraining

No handcrafted features beyond allowed topology features

Only methods covered in DGL Lectures 1.1–4.6

Models must run within 10 minutes on CPU

Any GNN architecture allowed (GIN, GCN, GraphSAGE, etc.)

--------------------
**Getting Started**
--------------------

1️  Install Dependencies

**pip install -r requirements.txt**


2️ Run the Baseline Model

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
** Submission Format**
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
** Leaderboard & Submission Notes**
-----------------------------------

Submissions are collected, but scores are NOT displayed immediately.

Leaderboard updates are controlled by organisers using hidden test labels stored securely via GitHub Secrets.

After submission, you will only see:

Submission successful

Scores, robustness gap, and rank are updated later by organisers.

**Leaderboard file**: leaderboard/leaderboard.md

Score calculation:

Macro F1-score for ideal and perturbed

Robustness gap = F1_ideal - F1_perturbed

-----------------------------
** Tips for Success**
-----------------------------

Structural features are crucial

Experiment with different topological descriptors

Regularization is important for small datasets

Simpler models often generalize better

GIN + structural features is a strong baseline
-------------------------
** Repository Structure**
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
 Step-by-Step Commands
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
  Contact
----------------

For questions or clarifications, open a GitHub Issue in this repository. Or you can contact me at, idrees11@yahoo.com (+918123434057)

-------------------

** License**

-------------------

This project is released under the MIT License.
See the LICENSE file for details.
