"""
submission.py
Template for participant submissions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, global_mean_pool
import networkx as nx
import numpy as np

def compute_enhanced_features(data):
    """
    IMPROVE THIS FUNCTION!
    Compute enhanced topological features for a graph.
    
    Args:
        data: PyG Data object with edge_index and num_nodes
    
    Returns:
        torch.Tensor: Enhanced features for each node
    """
    # Convert to NetworkX graph
    edge_index_np = data.edge_index.cpu().numpy()
    G = nx.Graph()
    G.add_edges_from(edge_index_np.T)
    
    # Ensure all nodes are present
    for i in range(data.num_nodes):
        if i not in G:
            G.add_node(i)
    
    features = []
    
    # TODO: Implement enhanced topological features
    # Some ideas:
    # - Higher-order centrality measures
    # - Community detection features
    # - Spectral graph features
    # - Motif counts
    # - Persistent homology features
    
    # Example: Add your enhanced features here
    # 1. Degree centrality (baseline)
    degrees = [G.degree(n) for n in range(data.num_nodes)]
    features.append(torch.tensor(degrees, dtype=torch.float).unsqueeze(1))
    
    # 2. Clustering coefficient
    if G.number_of_edges() > 0:
        clustering = nx.clustering(G)
        clustering_vec = [clustering.get(i, 0.0) for i in range(data.num_nodes)]
        features.append(torch.tensor(clustering_vec, dtype=torch.float).unsqueeze(1))
    
    # 3. Add your innovative features here...
    # features.append(your_innovative_feature)
    
    if features:
        return torch.cat(features, dim=1)
    return None

class EnhancedGraphModel(nn.Module):
    """
    IMPROVE THIS CLASS!
    Enhanced GIN model for graph classification.
    """
    
    def __init__(self, input_dim, hidden_dim=64, output_dim=2, num_layers=4):
        super(EnhancedGraphModel,
