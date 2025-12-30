"""
Graph Classification Challenge Starter Code
Challenge: Improve GIN model performance using topological feature engineering
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GINConv, global_mean_pool
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
import csv

# Set seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

class ChallengeDataset:
    """Challenge Dataset with hidden complexities"""
    def __init__(self, use_simple=False):
        """
        use_simple: If True, use simpler dataset (for testing)
                   If False, use the full challenge dataset
        """
        self.use_simple = use_simple
        
        if use_simple:
            # Simple version (for initial testing)
            self.dataset = TUDataset(root='data/TUDataset', name='MUTAG')
        else:
            # Challenge version - mixes different graph types
            # This creates a more complex dataset
            dataset1 = TUDataset(root='data/TUDataset', name='MUTAG')
            dataset2 = TUDataset(root='data/TUDataset', name='PROTEINS')
            
            # Combine and shuffle to create complexity
            self.dataset = []
            for i in range(max(len(dataset1), len(dataset2))):
                if i < len(dataset1):
                    self.dataset.append(dataset1[i])
                if i < len(dataset2):
                    # Add with modified labels to create complexity
                    data = dataset2[i].clone()
                    data.y = torch.tensor([1 - data.y.item()])  # Flip labels
                    self.dataset.append(data)
            
            random.shuffle(self.dataset)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    
    @property
    def num_features(self):
        sample = self[0]
        if sample.x is not None:
            return sample.x.shape[1]
        return 7  # Default feature dimension
    
    @property
    def num_classes(self):
        return 2

# Baseline topological feature computation
def compute_baseline_features(edge_index, num_nodes):
    """BASELINE IMPLEMENTATION - Participants should improve this!"""
    G = nx.Graph()
    edge_index_np = edge_index.cpu().numpy()
    G.add_edges_from(edge_index_np.T)
    
    # Add isolated nodes
    for i in range(num_nodes):
        if i not in G:
            G.add_node(i)
    
    # Simple features (baseline)
    features = []
    
    # Degree centrality
    degrees = [G.degree(n) for n in range(num_nodes)]
    features.append(torch.tensor(degrees, dtype=torch.float).unsqueeze(1))
    
    # Simple clustering (if graph has edges)
    if G.number_of_edges() > 0:
        clustering = nx.clustering(G)
        clustering_vec = [clustering.get(i, 0.0) for i in range(num_nodes)]
        features.append(torch.tensor(clustering_vec, dtype=torch.float).unsqueeze(1))
    
    return torch.cat(features, dim=1) if features else None

class BaselineGINModel(nn.Module):
    """BASELINE MODEL - Participants should improve this!"""
    def __init__(self, input_dim, hidden_dim=32, output_dim=2, num_layers=3):
        super(BaselineGINModel, self).__init__()
        
        self.convs = nn.ModuleList()
        
        # First GIN layer
        self.convs.append(GINConv(
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        ))
        
        # Additional layers
        for _ in range(num_layers - 1):
            self.convs.append(GINConv(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
            ))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # GIN layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Classification
        x = self.classifier(x)
        
        return F.log_softmax(x, dim=1)

def train_baseline_model(epochs=50):
    """BASELINE TRAINING - Participants should improve this!"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    dataset = ChallengeDataset(use_simple=True)
    
    # Split data
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    
    # Add baseline features
    train_data = []
    for idx in train_dataset.indices:
        data = dataset[idx].clone()
        if data.x is None:
            data.x = torch.ones(data.num_nodes, 1)  # Dummy features
        topo_features = compute_baseline_features(data.edge_index, data.num_nodes)
        if topo_features is not None:
            data.x = torch.cat([data.x, topo_features], dim=1)
        train_data.append(data)
    
    test_data = []
    for idx in test_dataset.indices:
        data = dataset[idx].clone()
        if data.x is None:
            data.x = torch.ones(data.num_nodes, 1)
        topo_features = compute_baseline_features(data.edge_index, data.num_nodes)
        if topo_features is not None:
            data.x = torch.cat([data.x, topo_features], dim=1)
        test_data.append(data)
    
    # Create loaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
    # Initialize model
    input_dim = train_data[0].x.shape[1]
    model = BaselineGINModel(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            out = model(data)
            loss = F.nll_loss(out, data.y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Evaluation
        model.eval()
        correct = 0
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                out = model(data)
                pred = out.argmax(dim=1)
                correct += (pred == data.y).sum().item()
        
        acc = correct / len(test_data)
        if acc > best_acc:
            best_acc = acc
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, '
                  f'Test Acc={acc:.4f}, Best={best_acc:.4f}')
    
    print(f'\nBaseline Results:')
    print(f'Final Test Accuracy: {acc:.4f}')
    print(f'Best Test Accuracy: {best_acc:.4f}')
    print(f'Input Dimension: {input_dim}')
    
    return best_acc

# Challenge starter
def run_challenge_starter():
    """Run the baseline implementation"""
    print("="*60)
    print("GRAPH CLASSIFICATION CHALLENGE - STARTER CODE")
    print("="*60)
    print("\nYOUR TASK:")
    print("1. Improve the topological feature computation in compute_baseline_features()")
    print("2. Enhance the GIN model architecture")
    print("3. Optimize the training process")
    print("4. Submit your improved solution using the scoring script\n")
    
    print("\nRunning baseline implementation...")
    baseline_acc = train_baseline_model(epochs=50)
    
    print("\n" + "="*60)
    print(f"BASELINE SCORE: {baseline_acc:.4f}")
    print("="*60)
    print("\nSubmit your improved solution to see if you can beat this score!")
    
    return baseline_acc

if __name__ == "__main__":
    run_challenge_starter()
