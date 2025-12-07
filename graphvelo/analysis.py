import scanpy as sc
import scvelo as scv
import numpy as np
import pandas as pd
import scipy.sparse
from scipy.stats import entropy

def get_gene_list(adata, gene_names):
    """
    Finds genes in adata.var_names, handling Case Sensitivity.
    e.g., finds 'Mcm5' even if user asks for 'MCM5'.
    """
    found_genes = []
    lookup = {g.lower(): g for g in adata.var_names}
    
    for g in gene_names:
        g_lower = g.lower()
        if g_lower in lookup:
            found_genes.append(lookup[g_lower])
            
    return found_genes

def extract_biological_features(adata):
    """
    Extracts biological context features.
    Returns: DataFrame (N_cells x N_features) --> for each cell, there are N features for describing its biological context.
    """
    print(f"--- GraphVelo Analysis: Extracting Context for {adata.n_obs} cells ---")
    
    features = pd.DataFrame(index=adata.obs_names)

    # 1. Velocity Confidence
    if 'velocity_confidence' not in adata.obs:
        print("   -> Computing velocity confidence...")
        try:
            scv.tl.velocity_confidence(adata) # Adds 'velocity_confidence' to adata.obs
        except Exception:
            print("   ! Warning: scVelo confidence calculation failed. Using zeros.")
            adata.obs['velocity_confidence'] = 0.0

    features['velocity_confidence'] = adata.obs['velocity_confidence'].values

    # 2. Graph Entropy (Differentiation Potential)
    if 'velocity_graph' in adata.uns:
        try:
            T = scv.utils.get_transition_matrix(adata)
            # Calculate entropy row-wise (dense or sparse) for each cell
            if scipy.sparse.issparse(T):
                # Efficient sparse entropy
                features['entropy'] = np.array([entropy(row.data) if len(row.data) > 0 else 0 
                                                for row in T])
            else:
                features['entropy'] = entropy(T, axis=1)
        except Exception as e:
            print(f"   ! Warning: Entropy calculation error ({e}). Using zeros.")
            features['entropy'] = 0.0
    else:
        features['entropy'] = 0.0

    # 3. Cell Cycle Scoring 
    # Standard S-phase and G2M-phase markers
    s_genes_ref = ['Mcm5', 'Pcna', 'Tyms', 'Fen1', 'Mcm2', 'Mcm4', 'Rrm1', 'Ung', 'Gins2', 'Mcm6', 'Cdca7', 'Dtl', 'Prim1', 'Uhrf1', 'Mlf1ip', 'Hells', 'Rfc2', 'Rpa2', 'Nasp', 'Rad51ap1', 'Gmnn', 'Wdr76', 'Slbp', 'Ccne2', 'Ubr7', 'Pold3', 'Msh2', 'Atad2', 'Rad51', 'Rrm2', 'Cdc45', 'Cdc6', 'Exo1', 'Tipin', 'Dscc1', 'Blm', 'Casp8ap2', 'Usp1', 'Clspn', 'Polola1', 'Chaf1b', 'Brip1', 'E2f8']
    g2m_genes_ref = ['Hmgb2', 'Cdk1', 'Nusap1', 'Ube2c', 'Birc5', 'Tpx2', 'Top2a', 'Ndc80', 'Cks2', 'Nuf2', 'Cks1b', 'Mki67', 'Tmpo', 'Cenpf', 'Tacc3', 'Fam64a', 'Smc4', 'Ccnb1', 'Ckap2l', 'Ckap2', 'Aurkb', 'Bub1', 'Kif11', 'Anp32e', 'Tubb4b', 'Gtse1', 'Kif20b', 'Hjurp', 'Cdca3', 'Hn1', 'Cdc20', 'Ttk', 'Cdc25c', 'Kif2c', 'Rangap1', 'Ncapd2', 'Dlgap5', 'Cdca2', 'Cdca8', 'Ect2', 'Kif23', 'Hmmr', 'Aurka', 'Psrc1', 'Anln', 'Lbr', 'Ckap5', 'Cenpe', 'Ctcf', 'Nek2', 'G2e3', 'Gas2l3', 'Cbx5', 'Cenpa']
    
    s_genes = get_gene_list(adata, s_genes_ref)
    g2m_genes = get_gene_list(adata, g2m_genes_ref)
    
    if len(s_genes) > 5 and len(g2m_genes) > 5:
        print(f"   -> Found {len(s_genes)} S-phase and {len(g2m_genes)} G2M genes.")
        sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)
        features['S_score'] = adata.obs['S_score'].values
        features['G2M_score'] = adata.obs['G2M_score'].values
    else:
        print("   ! Warning: Insufficient cell cycle genes found. Skipping scoring.")
        features['S_score'] = 0.0
        features['G2M_score'] = 0.0

    features = features.fillna(0.0)
    
    return features