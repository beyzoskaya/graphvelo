import torch
import torch.nn as nn

class Trainer:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005, weight_decay=1e-4)
        self.mse_crit = nn.MSELoss()
        
    def _direction_loss(self, pred, target, alpha=1.5):
        """Hybrid loss for velocity vectors."""
        loss_mse = self.mse_crit(pred, target)
        cos_sim = torch.nn.functional.cosine_similarity(pred, target, dim=1)
        loss_cos = (1.0 - cos_sim).mean()
        return loss_mse + (alpha * loss_cos)

    def train(self, data, epochs=100, mode='velocity', verbose=True):
        self.model.train()
        data = data.to(self.device)
        history = []
        
        print(f"--- GraphVelo Training: {mode.upper()} Mode ---")
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            if mode == 'generative':
                # Generative requires x + v inputs
                pred = self.model(data.x, data.v, data.edge_index)
                loss = self.mse_crit(pred[data.train_mask], data.y_genes[data.train_mask])
            else:
                # Standard Velocity requires x input
                pred = self.model(data.x, data.edge_index)
                loss = self._direction_loss(pred[data.train_mask], data.v[data.train_mask])
            
            loss.backward()
            self.optimizer.step()
            history.append(loss.item())
            
            if verbose and epoch % 20 == 0:
                print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")
                
        return history

    def predict(self, data):
        self.model.eval()
        data = data.to(self.device)
        with torch.no_grad():
            # Check if model supports attention return
            res = self.model(data.x, data.edge_index, return_attention=True)
            if isinstance(res, tuple):
                return res[0].cpu().numpy(), res[1]
            else:
                return res.cpu().numpy(), None
    
    def predict_generative(self, data):
        self.model.eval()
        data = data.to(self.device)
        with torch.no_grad():
            return self.model(data.x, data.v, data.edge_index).cpu().numpy()