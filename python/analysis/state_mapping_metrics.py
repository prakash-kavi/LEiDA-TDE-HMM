"""
Streamlined State Mapping Metrics for Meditation Analysis

This script extracts essential activation patterns from trained TDE-HMM models
for manual meditation state assignment.

Outputs a clean tabular format showing:
- Temporal metrics (FO, lifetime, stay probability)
- Z-scored activation for all 8 Yeo networks
- DMN anticorrelation with task networks
"""

import os
import numpy as np
import pickle
import logging
from datetime import datetime

# Import GLHMM modules
from glhmm import utils

# Setup minimal logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup paths
ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
TRAINED_DIR = os.path.join(DATA_DIR, 'trained', 'glhmm_tde')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')

# Yeo networks to analyze
YEO_NETWORKS = ['VIS', 'SMN', 'DAN', 'VAN', 'LIM', 'FPN', 'DMN', 'SUB']

def load_model(group, k):
    """Load trained HMM model."""
    model_path = os.path.join(TRAINED_DIR, group, f'k{k}', 'model.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # Make sure k is included
    if 'k' not in model_data:
        model_data['k'] = k
        
    logger.info(f"Loaded {group} k={k} model")
    return model_data

def load_network_data():
    """Load network mapping data."""
    data_path = os.path.join(DATA_DIR, 'processed', 'tde', 'tde_8networks_persequence.pkl')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Network data not found: {data_path}")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    return data

def cov_to_corr(cov_matrix):
    """Convert covariance matrix to correlation matrix."""
    d = np.sqrt(np.diag(cov_matrix))
    corr_matrix = cov_matrix / np.outer(d, d)
    corr_matrix = np.clip(corr_matrix, -1.0, 1.0)  # Ensure valid range
    return corr_matrix

def extract_metrics(model_data, network_data, group):
    """Extract only essential metrics needed for state mapping."""
    k = model_data['k']
    hmm = model_data['hmm']
    stc = model_data['stc']
    vpath = model_data['vpath']
    indices = model_data['indices']
    P = model_data['P']
    network_fields = network_data['network_fields']
    
    # --- Temporal metrics ---
    # Extract FO
    FO = utils.get_FO(stc, indices)
    FO_mean = np.mean(FO, axis=0)
    
    # Extract lifetimes
    # Convert to 1D if needed
    if len(vpath.shape) > 1 and vpath.shape[1] > 1:
        from glhmm import statistics
        vpath_1d = statistics.generate_vpath_1D(vpath)
        # Create 2D version for lifetime calculation
        vpath_2d = np.zeros((len(vpath_1d), k))
        for i, state in enumerate(vpath_1d):
            if state < k:
                vpath_2d[i, int(state)] = 1
    else:
        vpath_1d = vpath
        # Create 2D version
        vpath_2d = np.zeros((len(vpath_1d), k))
        for i, state in enumerate(vpath_1d):
            if state < k:
                vpath_2d[i, int(state)] = 1
    
    # Calculate lifetimes
    try:
        LTmean, _, _ = utils.get_life_times(vpath_2d, indices)
    except Exception:
        # Use placeholder if calculation fails
        LTmean = np.zeros(k)
    
    # Get self-transition probabilities
    self_trans = np.diag(P)
    
    # --- Network activation metrics ---
    # Get state covariances
    state_covs = []
    for state in range(k):
        state_covs.append(hmm.get_covariance_matrix(state))
    
    # Get network indices
    network_indices = {}
    available_networks = []
    for network in YEO_NETWORKS:
        if network in network_fields:
            network_indices[network] = network_fields.index(network)
            available_networks.append(network)
    
    # Calculate normalization statistics
    all_variances = []
    for cov in state_covs:
        network_size = min(len(network_fields), cov.shape[0])
        for network, idx in network_indices.items():
            if idx < network_size:
                all_variances.append(cov[idx, idx])
    
    # Get mean and std for z-scoring
    mean_var = np.mean(all_variances) if all_variances else 1.0
    std_var = np.std(all_variances) if all_variances and np.std(all_variances) > 0 else 1.0
    
    # Extract z-scored activations and DMN anticorrelation
    state_metrics = []
    for state_idx, cov in enumerate(state_covs):
        network_size = min(len(network_fields), cov.shape[0])
        corr = cov_to_corr(cov)
        
        # Get activations for all networks
        activations = {}
        for network in YEO_NETWORKS:
            if network in network_indices:
                idx = network_indices[network]
                if idx < network_size:
                    raw_var = cov[idx, idx]
                    z_var = (raw_var - mean_var) / std_var
                    activations[network] = z_var
                else:
                    activations[network] = 0.0
            else:
                activations[network] = 0.0
        
        # Calculate DMN anticorrelation
        dmn_anticorr = 0.0
        if 'DMN' in network_indices:
            dmn_idx = network_indices['DMN']
            task_anticorr = []
            for task_net in ['FPN', 'DAN', 'VAN']:
                if task_net in network_indices:
                    task_idx = network_indices[task_net]
                    if dmn_idx < network_size and task_idx < network_size:
                        task_anticorr.append(corr[dmn_idx, task_idx])
            
            if task_anticorr:
                dmn_anticorr = sum(task_anticorr) / len(task_anticorr)
        
        # Combine metrics
        state_metrics.append({
            'state': state_idx + 1,
            'FO': float(FO_mean[state_idx]),
            'lifetime': float(LTmean[state_idx]) if not isinstance(LTmean[state_idx], np.ndarray) 
                       else float(np.mean(LTmean[state_idx])),
            'stay': float(self_trans[state_idx]),
            'activations': activations,
            'dmn_anticorr': dmn_anticorr
        })
    
    return state_metrics

def export_mapping_data(output_dir=METRICS_DIR):
    """Export state mapping data in tabular format."""
    logger.info("Exporting state mapping data for manual assignment...")
    
    mapping_dir = os.path.join(output_dir, 'state_mapping')
    os.makedirs(mapping_dir, exist_ok=True)
    
    # Load network data once
    network_data = load_network_data()
    
    # Process only k=4 and k=5 models
    for k in [4, 5]:
        output_path = os.path.join(mapping_dir, f'k{k}_mapping_data.txt')
        
        with open(output_path, 'w') as f:
            f.write(f"STATE MAPPING DATA (k={k})\n")
            f.write("=======================\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d')}\n\n")
            
            # Process each group
            for group in ['controls', 'meditators']:
                f.write(f"{group.upper()}\n")
                f.write("-" * 8 + "\n\n")
                
                # Load model and extract metrics
                try:
                    model_data = load_model(group, k)
                    state_metrics = extract_metrics(model_data, network_data, group)
                except Exception as e:
                    f.write(f"Error processing {group} k={k}: {str(e)}\n\n")
                    continue
                
                # Write header
                f.write("State | FO    | Life  | Stay  | ")
                f.write(" | ".join(YEO_NETWORKS))
                f.write(" | DMN-Task\n")
                
                f.write("----- | ----- | ----- | ----- | ")
                f.write(" | ".join(["-" * 5] * len(YEO_NETWORKS)))
                f.write(" | --------\n")
                
                # Write data rows
                for metrics in state_metrics:
                    f.write(f"{metrics['state']:5d} | ")
                    f.write(f"{metrics['FO']:.3f} | ")
                    f.write(f"{metrics['lifetime']:.1f} | ")
                    f.write(f"{metrics['stay']:.3f} | ")
                    
                    # Write all network activations
                    for network in YEO_NETWORKS:
                        activation = metrics['activations'].get(network, 0.0)
                        f.write(f"{activation:5.2f} | ")
                    
                    # Write DMN anticorrelation
                    f.write(f"{metrics['dmn_anticorr']:.2f}\n")
                
                f.write("\n\n")
            
            # Simple notes
            f.write("Notes:\n")
            f.write("- FO = Fractional Occupancy (proportion of time spent in state)\n")
            f.write("- Life = Average lifetime in timepoints (TR=2.0s)\n")
            f.write("- Stay = Self-transition probability (stability of state)\n")
            f.write("- Network values are z-scored activations\n")
            f.write("- DMN-Task = Average correlation between DMN and task networks (FPN,DAN,VAN)\n")
        
        logger.info(f"Mapping data for k={k} saved to {output_path}")
    
    return mapping_dir

def main():
    """Main function to extract and export state mapping data."""
    logger.info("=== Extracting State Mapping Data for Meditation Analysis ===")
    
    try:
        # Export mapping data
        mapping_dir = export_mapping_data()
        logger.info(f"State mapping data saved to: {mapping_dir}")
        logger.info("=== State mapping data extraction complete ===")
        
    except Exception as e:
        logger.error(f"Error extracting state mapping data: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()