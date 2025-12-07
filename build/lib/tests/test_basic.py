import unittest
import torch
import scanpy as sc
import numpy as np
from graphvelo import preprocess_data, VelocityGAT, Trainer

class TestGraphVelo(unittest.TestCase):
    
    def test_pipeline_on_random_data(self):
        print("\n--- Testing Pipeline on Synthetic Data ---")
        
        # 1. Create Fake AnnData
        # 100 cells, 50 genes
        X = np.random.randint(0, 10, (100, 50))
        adata = sc.AnnData(X)
        
        # Add required layers for scVelo
        adata.layers['spliced'] = X
        adata.layers['unspliced'] = X + np.random.randint(0, 2, (100, 50))
        
        # 2. Test Preprocessing
        try:
            # Low params because data is small
            data, adata_proc = preprocess_data(adata, n_top_genes=20, n_pcs=5, n_neighbors=5)
            print("✔ Preprocessing Success")
        except Exception as e:
            self.fail(f"Preprocessing failed: {e}")
            
        # 3. Test Model Initialization
        try:
            model = VelocityGAT(input_dim=5, hidden_dim=16, output_dim=5)
            print("✔ Model Init Success")
        except Exception as e:
            self.fail(f"Model Init failed: {e}")
            
        # 4. Test Training Loop
        try:
            trainer = Trainer(model)
            hist = trainer.train(data, epochs=5, verbose=False)
            print("✔ Training Success")
        except Exception as e:
            self.fail(f"Training failed: {e}")

if __name__ == '__main__':
    unittest.main()