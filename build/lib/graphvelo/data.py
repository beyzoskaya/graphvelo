import scvelo as scv
import scanpy as sc
import torch
from torch_geometric.data import Data
import scipy.sparse
import numpy as np

def load_dataset(name='dentate'):
    """Loads built-in scVelo datasets."""
    if name == 'dentate':
        return scv.datasets.dentategyrus()
    elif name == 'pancreas':
        return scv.datasets.pancreas()
    else:
        raise ValueError(f"Dataset {name} not found. Available: 'dentate', 'pancreas'")

def preprocess_data(adata, n_top_genes=2000, n_pcs=30, n_neighbors=30):
    """
    Ingests an AnnData object and prepares it for GraphVelo.
    """
    print(f"--- GraphVelo: Preprocessing {adata.shape[0]} cells ---")
    
    # 1. Standard scVelo pipeline
    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=n_top_genes)
    scv.pp.moments(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)
    
    # 2. Dynamics
    scv.tl.recover_dynamics(adata, n_jobs=1)
    scv.tl.velocity(adata, mode='dynamical')
    scv.tl.velocity_graph(adata, n_jobs=1)
    
    # 3. Embeddings
    sc.tl.pca(adata, n_comps=n_pcs)
    scv.tl.velocity_embedding(adata, basis='pca')
    
    # 4. Tensor Conversion (Features)
    X = torch.tensor(adata.obsm['X_pca'], dtype=torch.float)
    
    # Check if velocity exists
    if 'velocity_pca' not in adata.obsm:
        raise ValueError("Velocity calculation failed. Ensure dataset has spliced/unspliced counts.")
        
    V_pca = torch.tensor(adata.obsm['velocity_pca'], dtype=torch.float)
    V_pca = torch.nan_to_num(V_pca)

    # 5. Graph Construction
    adj = adata.uns['velocity_graph']
    rows, cols = adj.nonzero()
    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_attr = torch.tensor(adj.data, dtype=torch.float)
    
    # 6. Raw Gene Target (for Generative models)
    if scipy.sparse.issparse(adata.X):
        y_genes = torch.tensor(adata.X.toarray(), dtype=torch.float)
    else:
        y_genes = torch.tensor(adata.X, dtype=torch.float)

    # Create PyG Data Object
    data = Data(x=X, edge_index=edge_index, edge_attr=edge_attr, v=V_pca, y_genes=y_genes)
    
    # Train/Test Split (80/20)
    n_nodes = data.num_nodes
    perm = torch.randperm(n_nodes)
    val_size = int(n_nodes * 0.2)
    data.train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    data.train_mask[perm[val_size:]] = True
    data.val_mask[perm[:val_size]] = True
    
    return data, adata