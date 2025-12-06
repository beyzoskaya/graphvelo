import os
import graphvelo as gv
import matplotlib
matplotlib.use('Agg')
import torch
import numpy as np

def main():
    # 1. Configuration
    DATASET = 'dentate' # Try 'pancreas' too!
    print(f"=== GraphVelo Demo: Generative Gene Reconstruction ({DATASET}) ===")

    # 2. Load Data
    raw_adata = gv.load_dataset(DATASET)
    data, adata = gv.preprocess_data(raw_adata)

    # 3. Initialize Generative Model
    # Input = 60 (30 State + 30 Velocity)
    # Output = 2000 Genes
    model = gv.GenerativeVelocityGNN(input_dim=60, hidden_dim=128, output_genes=data.y_genes.shape[1])

    # 4. Train
    print("\n--- Training Generative Autoencoder ---")
    trainer = gv.Trainer(model)
    # Note: mode='generative' uses MSE loss against gene counts
    history = trainer.train(data, epochs=100, mode='generative')

    # 5. Predict
    # Generative prediction requires passing the velocity vector
    gene_pred = trainer.predict_generative(data)

    # 6. Biological Metrics
    gene_names = adata.var_names.tolist()

    # A. Rank genes by reconstruction accuracy
    from graphvelo.plotting import plot_gene_metrics_ranking, plot_gene_spatial_comparison

    # This plots the histogram and returns indices of best genes
    top_indices = plot_gene_metrics_ranking(data.y_genes, gene_pred, gene_names)

    # B. Visualize Spatial Manifold for best genes
    plot_gene_spatial_comparison(adata, data.y_genes, gene_pred, gene_names, top_indices)

    print("\n=== GENERATIVE DEMO COMPLETE ===")
    print("Check visual_8_gene_metrics.png to see which genes were modeled best.")

if __name__ == "__main__":
    main()