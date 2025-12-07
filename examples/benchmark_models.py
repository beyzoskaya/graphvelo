import graphvelo as gv
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')

def evaluate_model(model_name, data, adata, x_bio):
    print(f"   ... Training {model_name} ...")
    input_dim = 30
    hidden_dim = 64
    output_dim = 30
    
    if model_name == 'GCN':
        model = gv.VelocityGNN(input_dim, 128, output_dim)
    elif model_name == 'GAT':
        model = gv.VelocityGAT(input_dim, hidden_dim, output_dim, heads=4)
    elif model_name == 'Hybrid':
        model = gv.HybridVelocityGAT(input_dim, x_bio.shape[1], hidden_dim, output_dim)
    
    trainer = gv.Trainer(model)
    # Pass x_bio only if model handles it (Hybrid), logic handled in Trainer
    use_bio = x_bio if model_name == 'Hybrid' else None
    
    trainer.train(data, x_bio=use_bio, epochs=100, verbose=False) 
    v_pred, _ = trainer.predict(data, x_bio=use_bio)
    
    # Score
    v_true = data.v.cpu().numpy()
    norms_p = np.linalg.norm(v_pred, axis=1)
    norms_t = np.linalg.norm(v_true, axis=1)
    norms_p[norms_p==0] = 1e-8; norms_t[norms_t==0] = 1e-8
    score = np.sum(v_pred * v_true, axis=1) / (norms_p * norms_t)
    return v_pred, np.mean(score)

def run_benchmark(dataset_name):
    print(f"\n==========================================")
    print(f" BENCHMARKING: {dataset_name.upper()}")
    print(f"==========================================")
    
    # 1. Load & Prep
    raw_adata = gv.load_dataset(dataset_name)
    data, adata = gv.preprocess_data(raw_adata)
    
    # 2. Bio Features (Required for Hybrid)
    bio_df = gv.extract_biological_features(adata)
    x_bio = torch.tensor(bio_df.values, dtype=torch.float)
    
    # 3. Models to test
    models = ['GCN', 'GAT', 'Hybrid']
    preds = {}
    scores = {}
    
    for m in models:
        pred, score = evaluate_model(m, data, adata, x_bio)
        preds[m] = pred
        scores[m] = score
        print(f"   -> {m} Score: {score:.4f}")

    print(f"\n   [Generating Reports for {dataset_name}]")
    
    gv.plot_benchmark_streamlines(
        adata, preds, 
        save=f"benchmark_{dataset_name}_streamlines.png"
    )
    
    gv.plot_benchmark_metrics(
        scores, 
        save=f"benchmark_{dataset_name}_metrics.png"
    )

def main():
    datasets = ['dentate', 'pancreas']
    for ds in datasets:
        try:
            run_benchmark(ds)
        except Exception as e:
            print(f"FAILED on {ds}: {e}")

if __name__ == "__main__":
    main()