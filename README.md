# GraphVelo 🧬 -> 🕸️

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-ee4c2c.svg)
![scVelo](https://img.shields.io/badge/scVelo-Compatible-green.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

**GraphVelo** is a Geometric Deep Learning library for **denoising RNA velocity**, **discovering biological driver cells**, and performing **generative in-silico lineage tracing**.

It treats single-cell differentiation as a flow on a graph manifold, utilizing **Graph Attention Networks (GATs)** to learn the underlying equations of cell fate dynamics.

---

## 📸 Capabilities at a Glance

### 1. Vector Field Denoising & Regularization
GraphVelo reconstructs a smooth, continuous vector field from noisy RNA velocity counts.
![Streamlines]<img width="4800" height="1200" alt="compare_3_streamlines" src="https://github.com/user-attachments/assets/79990acc-dc58-4f6f-91f9-f51247f45f61" />

*> **Left:** Raw scVelo (Noisy)![Uploading compare_3_streamlines.png…]()
. **Center:** GCN Smoothed. **Right:** GraphVelo (GAT) - Note the sharp decision boundaries at the branching point.*

### 2. Generative Simulation (ODE Solver)
Using the learned manifold, we perform time-forward simulations to predict the future fate of stem cells.
![Simulation](visual_4_simulation.png)
*> **In-Silico Lineage Tracing:** Starting from Radial Glia stem cells (Green), the model correctly predicts the bifurcation into Astrocytes vs. Granule Cells (Red).*

### 3. Unsupervised Driver Discovery
The attention mechanism automatically highlights cells that drive the system dynamics.
![Attention](compare_5_gat_attention.png)
*> **Influence Map:** Bright spots indicate "Driver Cells" (high incoming attention). The model spontaneously identified the Stem Cell niche without supervision.*

---

## 📦 Installation

GraphVelo is designed as a modular Python package.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/GraphVelo.git
    cd GraphVelo
    ```

2.  **Create the environment (Recommended):**
    ```bash
    conda env create -f environment_mac.yml
    conda activate scvelo-gnn
    ```

3.  **Install the library in editable mode:**
    ```bash
    pip install -e .
    ```

---

## 🚀 Quick Start

GraphVelo abstracts the complex GNN logic into a simple API.

```python
import graphvelo as gv
import matplotlib.pyplot as plt

# 1. Load Data (Custom or Built-in)
# Supports 'dentate' (Brain) and 'pancreas'
raw_adata = gv.load_dataset('dentate')

# 2. Preprocess & Construct Graph
data, adata = gv.preprocess_data(raw_adata)

# 3. Initialize & Train Model
model = gv.VelocityGAT(input_dim=30, hidden_dim=64, output_dim=30)
trainer = gv.Trainer(model)
trainer.train(data, epochs=100)

# 4. Predict & Visualize
v_pred, (att_edge, att_weights) = trainer.predict(data)

# Plotting
gv.plot_streamlines(adata, v_pred)
gv.plot_attention_map(adata, att_weights, att_edge)

# 5. Run Generative Simulation
gv.run_simulation(model, data, adata)
