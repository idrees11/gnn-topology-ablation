import torch
import pandas as pd
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import os

from dataset import TopologicalDataset
from model import GINModel

# ----------------------------
# Paths (root of repo)
# ----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DATA_DIR = os.path.join(REPO_ROOT, "data")
SUBMISSIONS_DIR = os.path.join(REPO_ROOT, "submissions")

os.makedirs(SUBMISSIONS_DIR, exist_ok=True)

# ----------------------------
# Load data splits
# ----------------------------
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

# ----------------------------
# Dataset instances
# ----------------------------
# Training → ideal condition
train_dataset = TopologicalDataset(
    "MUTAG",
    topo_config="degree",
    mode="ideal"
)

# Evaluation → two conditions
ideal_test_dataset = TopologicalDataset(
    "MUTAG",
    topo_config="degree",
    mode="ideal"
)

perturbed_test_dataset = TopologicalDataset(
    "MUTAG",
    topo_config="degree",
    mode="perturbed"
)

# ----------------------------
# Prepare graph lists
# ----------------------------
train_graphs = [train_dataset[i] for i in train_df.graph_index]
ideal_test_graphs = [ideal_test_dataset[i] for i in test_df.graph_index]
perturbed_test_graphs = [perturbed_test_dataset[i] for i in test_df.graph_index]

train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
ideal_test_loader = DataLoader(ideal_test_graphs, batch_size=32)
perturbed_test_loader = DataLoader(perturbed_test_graphs, batch_size=32)

# ----------------------------
# Model
# ----------------------------
model = GINModel(
    input_dim=train_dataset.num_features,
    output_dim=train_dataset.num_classes
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ----------------------------
# Training (Ideal Condition)
# ----------------------------
print("Training on IDEAL data...")
for epoch in range(50):
    model.train()
    total_loss = 0

    for data in train_loader:
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")

# ----------------------------
# Prediction function
# ----------------------------
def predict(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for data in loader:
            out = model(data)
            preds.extend(out.argmax(dim=1).tolist())
    return preds

# ----------------------------
# Evaluate in both conditions
# ----------------------------
print("Generating IDEAL predictions...")
ideal_predictions = predict(model, ideal_test_loader)

print("Generating PERTURBED predictions...")
perturbed_predictions = predict(model, perturbed_test_loader)

# ----------------------------
# Save submissions
# ----------------------------
ideal_submission_path = os.path.join(SUBMISSIONS_DIR, "ideal_submission.csv")
perturbed_submission_path = os.path.join(SUBMISSIONS_DIR, "perturbed_submission.csv")

pd.DataFrame({
    "graph_index": test_df.graph_index,
    "target": ideal_predictions
}).to_csv(ideal_submission_path, index=False)

pd.DataFrame({
    "graph_index": test_df.graph_index,
    "target": perturbed_predictions
}).to_csv(perturbed_submission_path, index=False)

print(f"Saved ideal submission to: {ideal_submission_path}")
print(f"Saved perturbed submission to: {perturbed_submission_path}")
