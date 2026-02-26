# src/graph/build_graph.py

import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import pandas as pd


FEATURE_COLS = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']


def compute_cosine_similarity(X: np.ndarray) -> np.ndarray:
    """Compute n×n cosine similarity matrix."""
    X_norm = normalize(X, norm='l2')
    sim_matrix = cosine_similarity(X_norm)
    return sim_matrix


def build_knn_edges(sim_matrix: np.ndarray, k: int = 7, threshold: float = 0.8):
    """
    Build edge list from similarity matrix using k-NN.
    Returns edge_index (2, num_edges) and edge_weight (num_edges,)
    """
    n = sim_matrix.shape[0]
    src_list, dst_list, weight_list = [], [], []

    for i in range(n):
        # Get top-k neighbors (excluding self)
        sim_row = sim_matrix[i].copy()
        sim_row[i] = -1  # exclude self-loop
        
        top_k_idx = np.argsort(sim_row)[-k:]  # indices of top-k
        
        for j in top_k_idx:
            if sim_matrix[i][j] >= threshold:
                src_list.append(i)
                dst_list.append(j)
                weight_list.append(sim_matrix[i][j])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_weight = torch.tensor(weight_list, dtype=torch.float)
    
    print(f"Nodes: {n}, Edges: {edge_index.shape[1]}, Avg degree: {edge_index.shape[1]/n:.2f}")
    return edge_index, edge_weight


def make_undirected(edge_index: torch.Tensor, edge_weight: torch.Tensor):
    """Make graph undirected by adding reverse edges."""
    reverse = torch.stack([edge_index[1], edge_index[0]], dim=0)
    edge_index_full = torch.cat([edge_index, reverse], dim=1)
    edge_weight_full = torch.cat([edge_weight, edge_weight], dim=0)
    
    # Remove duplicate edges
    combined = torch.cat([edge_index_full, edge_weight_full.unsqueeze(0)], dim=0).T
    combined_unique = torch.unique(combined, dim=0)
    edge_index_out = combined_unique[:, :2].T.long()
    edge_weight_out = combined_unique[:, 2]
    
    return edge_index_out, edge_weight_out


def build_graph(csv_path: str, k: int = 7, threshold: float = 0.8, save_path: str = None) -> Data:
    """
    Master function: load processed data → build PyG Data object.
    This is what Member 2 imports.
    """
    df = pd.read_csv(csv_path)
    
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df['label_enc'].values.astype(np.int64)
    
    sim_matrix = compute_cosine_similarity(X)
    edge_index, edge_weight = build_knn_edges(sim_matrix, k=k, threshold=threshold)
    edge_index, edge_weight = make_undirected(edge_index, edge_weight)
    
    # Node feature tensor and label tensor
    x = torch.tensor(X, dtype=torch.float)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    # Create train/val/test masks (70/15/15)
    n = x.shape[0]
    indices = torch.randperm(n)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask   = torch.zeros(n, dtype=torch.bool)
    test_mask  = torch.zeros(n, dtype=torch.bool)
    
    train_mask[indices[:train_end]] = True
    val_mask[indices[train_end:val_end]] = True
    test_mask[indices[val_end:]] = True
    
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_weight,
        y=y_tensor,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )
    
    print(f"\nGraph Summary:")
    print(f"  Nodes (farms/samples): {data.num_nodes}")
    print(f"  Edges:                 {data.num_edges}")
    print(f"  Node features:         {data.num_node_features}")
    print(f"  Classes:               {y_tensor.max().item() + 1}")
    print(f"  Train/Val/Test:        {train_mask.sum()}/{val_mask.sum()}/{test_mask.sum()}")
    
    if save_path:
        torch.save(data, save_path)
        print(f"\nGraph saved to {save_path}")
    
    return data


if __name__ == "__main__":
    data = build_graph(
        csv_path="data/processed/crop_processed.csv",
        k=7,
        threshold=0.80,
        save_path="data/graph/crop_graph.pt"
    )