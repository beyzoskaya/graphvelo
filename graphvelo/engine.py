import torch
import torch.nn as nn

class Trainer:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005, weight_decay=1e-4)
        self.mse_crit = nn.MSELoss()
        
    def _direction_loss(self, pred, target, alpha=1.5):
        """Hybrid loss: MSE (Magnitude) + Cosine (Direction)"""
        loss_mse = self.mse_crit(pred, target)
        cos_sim = torch.nn.functional.cosine_similarity(pred, target, dim=1)
        loss_cos = (1.0 - cos_sim).mean()
        return loss_mse + (alpha * loss_cos)

    def train(self, data, x_bio=None, epochs=100, mode='velocity', verbose=True):
        """
        Universal training loop.
        Args:
            x_bio: Tensor of biological features (required for Hybrid model)
            mode: 'velocity' (default), 'generative'
        """
        self.model.train()
        data = data.to(self.device)
        if x_bio is not None:
            x_bio = x_bio.to(self.device)
            
        history = []
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # --- DYNAMIC FORWARD PASS ---
            if x_bio is not None:
                # Hybrid Case: (x, bio, edge_index)
                pred = self.model(data.x, x_bio, data.edge_index)
                loss = self._direction_loss(pred[data.train_mask], data.v[data.train_mask])
                
            elif mode == 'generative':
                # Generative Case: (x, v, edge_index) -> Genes
                pred = self.model(data.x, data.v, data.edge_index)
                loss = self.mse_crit(pred[data.train_mask], data.y_genes[data.train_mask])
                
            else:
                # Standard GCN/GAT Case: (x, edge_index)
                pred = self.model(data.x, data.edge_index)
                loss = self._direction_loss(pred[data.train_mask], data.v[data.train_mask])
            
            loss.backward()
            self.optimizer.step()
            history.append(loss.item())
            
            if verbose and epoch % 20 == 0:
                print(f"   Epoch {epoch:03d} | Loss: {loss.item():.4f}")
                
        return history

    def predict(self, data, x_bio=None):
        """Universal prediction function."""
        self.model.eval()
        data = data.to(self.device)
        if x_bio is not None:
            x_bio = x_bio.to(self.device)
            
        with torch.no_grad():
            # 1. Hybrid Prediction
            if x_bio is not None:
                return self.model(data.x, x_bio, data.edge_index).cpu().numpy(), None

            # 2. Standard/GAT Prediction (Check for attention weights)
            # Inspect signature or try/except to handle return_attention
            try:
                res = self.model(data.x, data.edge_index, return_attention=True)
                if isinstance(res, tuple):
                    return res[0].cpu().numpy(), res[1] # Pred, Weights
                else:
                    return res.cpu().numpy(), None
            except TypeError:
                # Model doesn't support return_attention (e.g., GCN)
                return self.model(data.x, data.edge_index).cpu().numpy(), None

    def predict_generative(self, data):
        self.model.eval()
        data = data.to(self.device)
        with torch.no_grad():
            return self.model(data.x, data.v, data.edge_index).cpu().numpy()