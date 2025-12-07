import graphvelo as gv
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    print("================================================")
    print("   GRAPHVELO: HYBRID MODEL DEMO")
    print("   (Fusing Geometry + Biology)")
    print("================================================")
    
    DATASET = 'pancreas' 
    print(f"\n[1] Loading {DATASET}...")
    raw_adata = gv.load_dataset(DATASET)
    data, adata = gv.preprocess_data(raw_adata)

    print(f"\n[2] Extracting Biological Context...")
    bio_features = gv.extract_biological_features(adata)

    from graphvelo.plotting import plot_bio_context
    plot_bio_context(adata, bio_features)
    
    x_bio = torch.tensor(bio_features.values, dtype=torch.float)
    
    print(f"\n[3] Initializing HybridVelocityGAT...")
    model = gv.HybridVelocityGAT(
        input_dim_pca=30,
        input_dim_bio=x_bio.shape[1], 
        hidden_dim=64,
        output_dim=30,
        heads=4
    )

    print("\n[4] Training Hybrid Network...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn = torch.nn.MSELoss()
    
    model.train()
    epochs = 100
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Pass PCA + BIO + GRAPH
        pred = model(data.x, x_bio, data.edge_index)
        
        # Simple MSE loss against ground truth velocity
        loss = loss_fn(pred[data.train_mask], data.v[data.train_mask])
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"    Epoch {epoch:03d} | Loss: {loss.item():.4f}")

    # Evaluation & Visualization
    print("\n[5] Evaluating Results...")
    model.eval()
    with torch.no_grad():
        v_pred = model(data.x, x_bio, data.edge_index).cpu().numpy()
    
    gv.plot_streamlines(adata, v_pred)
    
    print("\n=== HYBRID DEMO COMPLETE ===")
    print("1. 'visual_8_bio_features.png': Shows the biological inputs.")
    print("2. 'visual_3_streamlines.png': Shows the resulting Hybrid vector field.")

if __name__ == "__main__":
    main()