import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv

# --- 1. GCN ---
class VelocityGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VelocityGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.predictor = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, return_attention=False):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        out = self.predictor(x)
        if return_attention: return out, None
        return out

# --- 2. GAT (Fixed Unpacking) ---
class VelocityGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4):
        super(VelocityGAT, self).__init__()
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=heads, concat=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)
        self.conv2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=1, concat=False)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.predictor = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, return_attention=False):
        # Layer 1: No attention return needed
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x) 
        x = F.dropout(x, p=0.3, training=self.training)
        
        # Layer 2: Optional attention return
        if return_attention:
            x, (att_edge_index, att_weights) = self.conv2(x, edge_index, return_attention_weights=True)
        else:
            x = self.conv2(x, edge_index)
            att_edge_index, att_weights = None, None

        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        out = self.predictor(x)
        if return_attention: return out, (att_edge_index, att_weights)
        return out

# --- 3. GENERATIVE (Autoencoder) ---
class GenerativeVelocityGNN(nn.Module):
    def __init__(self, input_dim=60, hidden_dim=128, output_genes=2000, heads=4):
        super(GenerativeVelocityGNN, self).__init__()
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=heads, concat=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)
        self.conv2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=1, concat=False)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1024),
            nn.ELU(),
            nn.Linear(1024, output_genes)
        )

    def forward(self, x_pca, v_pca, edge_index):
        x_in = torch.cat([x_pca, v_pca], dim=1) 
        h = self.conv1(x_in, edge_index) # No unpacking
        h = self.bn1(h)
        h = F.elu(h)
        h = self.conv2(h, edge_index) # No unpacking
        h = self.bn2(h)
        h = F.elu(h)
        return self.decoder(h)