import os
import graphvelo as gv
import matplotlib
matplotlib.use('Agg')
import numpy as np

def main():
    # 1. Configuration
    DATASET = 'dentate'
    print(f"=== GraphVelo Demo: {DATASET.upper()} Analysis ===")

    # 2. Load & Process Data
    # The library handles everything internally
    raw_adata = gv.load_dataset(DATASET)
    data, adata = gv.preprocess_data(raw_adata)

    # 3. Initialize GAT Model
    # Note: Input is 30 (PCA), Hidden is 64, Output is 30 (Velocity)
    model = gv.VelocityGAT(input_dim=30, hidden_dim=64, output_dim=30, heads=4)

    # 4. Train
    print("\n--- Training Graph Attention Network ---")
    trainer = gv.Trainer(model)
    # We use the built-in training loop
    history = trainer.train(data, epochs=120, mode='velocity')

    # 5. Predict
    v_pred, (att_edge_index, att_weights) = trainer.predict(data)

    # Calculate Cosine Similarity for metrics
    v_true = data.v.cpu().numpy()
    norms_p = np.linalg.norm(v_pred, axis=1)
    norms_t = np.linalg.norm(v_true, axis=1)
    norms_p[norms_p==0] = 1e-8; norms_t[norms_t==0] = 1e-8
    cosine_scores = np.sum(v_pred * v_true, axis=1) / (norms_p * norms_t)
    print(f"Final Mean Cosine Similarity: {np.mean(cosine_scores):.4f}")

    # 6. Generate Portfolio Plots
    print("\n--- Generating Plots ---")
    gv.plot_graph_connectivity(data, adata)
    gv.plot_streamlines(adata, v_pred)
    gv.plot_per_cluster_performance(adata, cosine_scores)
    gv.plot_attention_map(adata, att_weights, att_edge_index)

    # NEW: Plot Velocity Speed
    # We import this specific function if it's not exposed in top-level init, 
    # but based on previous steps it should be available via gv.plot_velocity_speed if added to __init__
    # If you didn't add it to __init__.py, you might need: from graphvelo.plotting import plot_velocity_speed
    try:
        gv.plot_velocity_speed(adata, v_pred)
    except AttributeError:
        from graphvelo.plotting import plot_velocity_speed
        plot_velocity_speed(adata, v_pred)

    # 7. Run Simulation
    gv.run_simulation(model, data, adata)

    print("\n=== DEMO COMPLETE ===")

if __name__ == "__main__":
    # Essential for Mac OS Multiprocessing
    main()