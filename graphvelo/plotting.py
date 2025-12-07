import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scvelo as scv
from matplotlib.collections import LineCollection
import pandas as pd
import torch

def plot_graph_connectivity(data, adata, subset_size=500, save="visual_1_graph_topology.png"):
    print(f"   -> [Plotting] Graph Topology -> {save}")
    X = data.x.cpu().numpy()
    edge_index = data.edge_index.cpu().numpy()
    if data.num_nodes > subset_size:
        indices = np.random.choice(data.num_nodes, subset_size, replace=False)
    else: indices = np.arange(data.num_nodes)

    mask = np.isin(edge_index[0], indices) & np.isin(edge_index[1], indices)
    subset_edges = edge_index[:, mask]
    
    plt.figure(figsize=(10, 8))
    plt.scatter(X[:,0], X[:,1], c='lightgrey', s=5, alpha=0.3)
    start_pts = X[subset_edges[0]]
    end_pts = X[subset_edges[1]]
    lines = list(zip(start_pts[:, :2], end_pts[:, :2]))
    lc = LineCollection(lines, colors='teal', linewidths=0.1, alpha=0.3)
    plt.gca().add_collection(lc)
    plt.scatter(X[indices,0], X[indices,1], c='black', s=10, zorder=5)
    plt.title(f"Graph Topology ({adata.uns.get('dataset_name', 'dataset')})")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.savefig(save, dpi=200)
    plt.close()

def plot_per_cluster_performance(adata, cosine_scores, save="visual_2_cluster_performance.png"):
    print(f"   -> [Plotting] Cluster Performance -> {save}")
    key = None
    for k in ['clusters', 'louvain', 'cell_type']:
        if k in adata.obs:
            key = k
            break
    if not key: return
    clusters = adata.obs[key].values
    df = pd.DataFrame({'Cell Type': clusters, 'Cosine Similarity': cosine_scores})
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Cell Type', y='Cosine Similarity', palette='viridis')
    plt.axhline(0, color='r', linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save, dpi=150)
    plt.close()

def plot_streamlines(adata, V_pred_pca, save="visual_3_streamlines.png"):
    print(f"   -> [Plotting] Streamlines -> {save}")
    adata_pred = adata.copy()
    adata_pred.obsm['velocity_pca'] = V_pred_pca
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    scv.pl.velocity_embedding_stream(adata, basis='pca', title='Ground Truth', ax=ax[0], show=False)
    scv.pl.velocity_embedding_stream(adata_pred, basis='pca', title='GraphVelo Prediction', ax=ax[1], show=False)
    plt.savefig(save, dpi=150)
    plt.close()

def plot_attention_map(adata, attention_weights, edge_index, save="visual_4_biological_attention.png"):
    print(f"   -> [Plotting] Biological Attention -> {save}")
    if attention_weights is None: return
    num_nodes = adata.shape[0]
    node_score = np.zeros(num_nodes)
    sources = edge_index[0].cpu().numpy()
    weights = attention_weights.detach().cpu().numpy().flatten()
    np.add.at(node_score, sources, weights)
    
    plt.figure(figsize=(10, 8))
    sc_plot = plt.scatter(adata.obsm['X_pca'][:,0], adata.obsm['X_pca'][:,1], 
                     c=node_score, cmap='magma', s=15, alpha=0.9)
    plt.colorbar(sc_plot, label="Influence Score")
    plt.title("Biological Driver Analysis (Attention Weights)")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.savefig(save, dpi=200)
    plt.close()

def plot_velocity_speed(adata, V_pred_pca, save="visual_5_velocity_speed.png"):
    print(f"   -> [Plotting] Velocity Speed Map -> {save}")
    speed = np.linalg.norm(V_pred_pca, axis=1)
    plt.figure(figsize=(10, 8))
    sc_plot = plt.scatter(adata.obsm['X_pca'][:,0], adata.obsm['X_pca'][:,1], 
                     c=speed, cmap='plasma', s=15, alpha=0.9)
    plt.colorbar(sc_plot, label="Differentiation Speed")
    plt.title("Predicted Speed of Differentiation")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.savefig(save, dpi=150)
    plt.close()

def plot_gene_metrics_ranking(y_true, y_pred, gene_names, save="visual_6_gene_metrics.png"):
    print(f"   -> [Plotting] Gene Metrics -> {save}")
    if torch.is_tensor(y_true): y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred): y_pred = y_pred.detach().cpu().numpy()
    
    corrs = []
    for i in range(y_true.shape[1]):
        if np.std(y_true[:, i]) == 0 or np.std(y_pred[:, i]) == 0: corrs.append(0)
        else: corrs.append(np.corrcoef(y_true[:, i], y_pred[:, i])[0,1])
    corrs = np.array(corrs)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(corrs, bins=30, kde=True, color='teal')
    plt.title("Distribution of Reconstruction Accuracy")
    
    top_indices = np.argsort(corrs)[-10:][::-1]
    top_genes = [gene_names[i] for i in top_indices]
    top_scores = corrs[top_indices]
    
    plt.subplot(1, 2, 2)
    sns.barplot(x=top_scores, y=top_genes, palette='viridis')
    plt.title("Top 10 Best Reconstructed Genes")
    plt.tight_layout()
    plt.savefig(save, dpi=150)
    plt.close()
    return top_indices

def plot_gene_spatial_comparison(adata, y_true, y_pred, gene_names, top_indices=None, save="visual_7_gene_spatial.png"):
    print(f"   -> [Plotting] Gene Spatial -> {save}")
    if torch.is_tensor(y_true): y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred): y_pred = y_pred.detach().cpu().numpy()
    if top_indices is None:
        corrs = [np.corrcoef(y_true[:, i], y_pred[:, i])[0,1] for i in range(y_true.shape[1])]
        top_indices = np.argsort(corrs)[-3:]
    else: top_indices = top_indices[:3]
    
    X_pca = adata.obsm['X_pca']
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    for i, gene_idx in enumerate(top_indices):
        gene_name = gene_names[gene_idx]
        axes[i, 0].scatter(X_pca[:,0], X_pca[:,1], c=y_true[:, gene_idx], s=10, cmap='viridis')
        axes[i, 0].set_title(f"{gene_name} (True)")
        axes[i, 1].scatter(X_pca[:,0], X_pca[:,1], c=y_pred[:, gene_idx], s=10, cmap='viridis')
        axes[i, 1].set_title(f"{gene_name} (Pred)")
    plt.tight_layout()
    plt.savefig(save, dpi=150)
    plt.close()

def plot_bio_context(adata, bio_df, save="visual_8_bio_features.png"):
    print(f"   -> [Plotting] Biological Context -> {save}")
    X_pca = adata.obsm['X_pca']
    features = bio_df.columns.tolist()
    n_feats = len(features)
    fig, axes = plt.subplots(1, n_feats, figsize=(5 * n_feats, 5))
    if n_feats == 1: axes = [axes]
    for i, col in enumerate(features):
        ax = axes[i]
        sc_plot = ax.scatter(X_pca[:,0], X_pca[:,1], c=bio_df[col].values, cmap='plasma', s=10, alpha=0.8)
        ax.set_title(col)
        plt.colorbar(sc_plot, ax=ax)
    plt.tight_layout()
    plt.savefig(save, dpi=150)
    plt.close()

# --- BENCHMARK PLOTS ---
def plot_benchmark_streamlines(adata, predictions_dict, save="benchmark_streamlines.png"):
    print(f"   -> [Plotting] Benchmark Streamlines -> {save}")
    n_models = len(predictions_dict)
    fig, ax = plt.subplots(1, n_models + 1, figsize=(5 * (n_models + 1), 5))
    
    scv.pl.velocity_embedding_stream(adata, basis='pca', title='Ground Truth', ax=ax[0], show=False)
    
    for i, (name, v_pred) in enumerate(predictions_dict.items()):
        temp = adata.copy()
        temp.obsm['velocity_pca'] = v_pred
        scv.pl.velocity_embedding_stream(temp, basis='pca', title=name, ax=ax[i+1], show=False)
    plt.tight_layout()
    plt.savefig(save, dpi=150)
    plt.close()

def plot_benchmark_metrics(model_scores_dict, save="benchmark_metrics.png"):
    print(f"   -> [Plotting] Benchmark Metrics -> {save}")
    names = list(model_scores_dict.keys())
    scores = list(model_scores_dict.values())
    plt.figure(figsize=(8, 6))
    bars = plt.bar(names, scores, color=['steelblue', 'orange', 'crimson'])
    plt.ylim(0, 1.05)
    plt.title("Model Accuracy (Cosine Similarity)")
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{bar.get_height():.3f}", ha='center', va='bottom')
    plt.savefig(save, dpi=150)
    plt.close()