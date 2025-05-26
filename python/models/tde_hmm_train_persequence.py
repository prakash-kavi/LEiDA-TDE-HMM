"""
TDE-HMM Model Training Script for Meditation Analysis

This script trains Time-Delay Embedding Hidden Markov Models (TDE-HMM)
using the GLHMM library, following the methodology in Vidaurre et al. (2018).

The script:
1. Loads preprocessed TDE data
2. Trains separate HMMs for controls and meditators with per-sequence standardization
3. Tests multiple state numbers (k=4 to k=5)
4. Saves trained models for future analysis
5. Outputs summary metrics for model comparison
"""

import os
import sys
import numpy as np
import pickle
import logging
import time
from datetime import datetime

# Import GLHMM modules
from glhmm import glhmm, preproc, utils, statistics, auxiliary

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup paths
ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
TDE_DIR = os.path.join(PROCESSED_DIR, 'tde')
TRAINED_DIR = os.path.join(DATA_DIR, 'trained', 'persequence')

# Create trained models directory
os.makedirs(TRAINED_DIR, exist_ok=True)

def load_tde_data(networks):
    """Load the preprocessed time-delay embedded data."""
    logger.info(f"Loading {networks}-network TDE data...")
    
    # Define data path based on network configuration
    data_path = os.path.join(TDE_DIR, f'tde_{networks}networks_persequence.pkl')
    
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    try:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        logger.info(f"Loaded data with {len(data['controls_sequences'])} control subjects and "
                   f"{len(data['meditators_sequences'])} meditation subjects")
        logger.info(f"PCA components: {data['pca_components']}")
        logger.info(f"Lags used: {data['lags']}")
        
        return data
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def prepare_hmm_data(sequences):
    """ Prepare data for HMM by concatenating sequences and creating indices. """
    # Concatenate all sequences into one array
    concatenated_data = np.vstack(sequences)
    
    # Create indices for each subject
    indices = statistics.get_indices_from_list(sequences)
    
    return concatenated_data, indices

def train_tde_model(data, group, k):
    """ Build TDE-HMM model using the GLHMM library """
    # Fixed parameters
    L = 10          # Maximum lag
    S = 1           # Step size
    
    # Get sequences for this group
    sequences = data[f'{group}_sequences']
    n_regions = sequences[0].shape[1]  # Number of networks/regions
    
    # Prepare data for HMM
    X_preproc, idx_data = prepare_hmm_data(sequences)
    
    # Standardize data (per sequence)
    X_preproc = preproc.preprocess_data(X_preproc, idx_data)[0]
    
    # Add global network baselines for better group comparisons
    network_baselines = {}
    for net_idx, network in enumerate(data['network_fields']):
        network_baselines[network] = {
            'mean': np.mean(X_preproc[:, net_idx]),
            'std': np.std(X_preproc[:, net_idx])
        }
        print(f"Network {network} baseline: mean={network_baselines[network]['mean']:.4f}, std={network_baselines[network]['std']:.4f}")
        
    # Define TDE lags
    lags = np.arange(-L, L+1, S)
    print(f"Using lags: {lags}")
    
    # Apply time-delay embedding with PCA
    pca_components = n_regions * 2  # Following Vidaurre's approach
    print(f"Applying TDE with PCA components: {pca_components}")

    # Handle the version difference in build_data_tde return values
    try:
        # Try unpacking 3 values (newer GLHMM version)
        X_embedded, idx_tde, pca_model = preproc.build_data_tde(
            X_preproc, idx_data, lags=lags, pca=pca_components
        )
    except ValueError:
        # Fall back to 2 values (older GLHMM version)
        X_embedded, idx_tde = preproc.build_data_tde(
            X_preproc, idx_data, lags=lags, pca=pca_components
        )
        # Create a dummy PCA model
        pca_model = None
        print("Using older GLHMM version without PCA model return")

    print(f"TDE data shape: {X_embedded.shape}, Indices shape: {idx_tde.shape}")
    
    # Create HMM model    
    TDE_hmm = glhmm.glhmm(
        model_beta='no',    # No regression part
        model_mean='state', # State-specific means
        K=k,                # Number of states
        covtype='full'      # Full covariance matrices
    )
    
    # Train the HMM
    print(f"Training {group} TDE-HMM model with k={k} states...")
    start_time = time.time()
    
    # Train model with variable names matching Vidaurre
    stc_tde, xi_tde, fe_tde = TDE_hmm.train(X=None, Y=X_embedded, indices=idx_tde)
    vpath_tde = TDE_hmm.decode(X=None, Y=X_embedded, indices=idx_tde, viterbi=True)

    # Convert free energy from numpy array to scalar
    if isinstance(fe_tde, np.ndarray):
        fe_tde_scalar = fe_tde.item() if fe_tde.size == 1 else float(fe_tde[0])
    else:
        fe_tde_scalar = float(fe_tde)

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds with free energy: {fe_tde_scalar:.2f}")
    
    # Calculate state metrics 
    print("Calculating temporal metrics...")
    FO = utils.get_FO(stc_tde, indices=idx_tde)  # Fractional occupancy
    SR = utils.get_switching_rate(stc_tde, indices=idx_tde)  # Switching rate
    LTmean, LTmed, LTmax = utils.get_life_times(vpath_tde, indices=idx_tde)  # Lifetimes
    
    print("Detailed switching metrics:")
    print(f"  Average switching rate: {np.mean(SR):.6f}")
    print(f"  Min switching rate: {np.min(SR):.6f}")
    print(f"  Max switching rate: {np.max(SR):.6f}")
    print(f"  State lifetimes (mean): {np.mean(LTmean, axis=0)}")
    
    # Calculate state means for network activation profiles
    network_fields = data['network_fields']
    state_means = []
    state_z_scores = []  # Store z-scores for better group comparisons
    state_global_norm_scores = []  # Store global norm values

    # Map vpath_tde back to original time dimensions
    # This accounts for the timing impact of TDE embedding
    T = auxiliary.get_T(idx_data)  # Original time lengths
    options_tde = {'embeddedlags': list(lags)}
    padded_vpath = auxiliary.padGamma(vpath_tde, T, options_tde)  # Align vpath to original time

    print(f"Original data shape: {X_preproc.shape}, Padded vpath shape: {padded_vpath.shape}")

    # Check shapes to ensure alignment
    if padded_vpath.shape[0] > X_preproc.shape[0]:
        # Trim excess padding
        padded_vpath = padded_vpath[:X_preproc.shape[0], :]
    elif padded_vpath.shape[0] < X_preproc.shape[0]:
        # Use only aligned portion
        X_preproc = X_preproc[:padded_vpath.shape[0], :]
        print(f"Warning: Trimmed X_preproc to match padded_vpath length: {X_preproc.shape}")

    for state in range(k):
        # Average network values when this state is decoded as active
        state_mask = np.argmax(padded_vpath, axis=1) == state
        if np.sum(state_mask) > 0:
            state_data = X_preproc[state_mask]
            state_mean = np.mean(state_data, axis=0)
            
            # Ensure we're only using the first n_regions columns if data has more dimensions
            if len(state_mean) > len(network_fields):
                print(f"Warning: State mean has {len(state_mean)} dimensions but there are only {len(network_fields)} networks.")
                print("Using only the first network dimensions for state profiles.")
                state_mean = state_mean[:len(network_fields)]
            
            # Calculate within-state z-score (relative network importance within state)
            state_z = (state_mean - np.mean(state_mean)) / (np.std(state_mean) + 1e-10)
            
            # Calculate globally normalized values (for between-group comparison)
            state_global_norm = []
            for net_idx, network in enumerate(network_fields):
                if net_idx < len(state_mean):  # Safety check
                    baseline = network_baselines[network]
                    # How many standard deviations from global mean?
                    global_z = (state_mean[net_idx] - baseline['mean']) / (baseline['std'] + 1e-10)
                    state_global_norm.append(global_z)
            
            state_means.append(state_mean)
            state_z_scores.append(state_z)
            state_global_norm_scores.append(state_global_norm)
        else:
            print(f"Warning: State {state} has no timepoints in original space")
            state_means.append(np.zeros(len(network_fields)))
            state_z_scores.append(np.zeros(len(network_fields)))
            state_global_norm_scores.append(np.zeros(len(network_fields)))
    
    # Create model data dictionary with all relevant information
    model_data = {
        'hmm': TDE_hmm,
        'free_energy': fe_tde_scalar,
        'training_time': training_time,
        'stc': stc_tde,
        'xi': xi_tde,
        'vpath': vpath_tde,
        'indices': idx_tde,
        'FO': FO,
        'SR': SR,
        'LTmean': LTmean,
        'LTmed': LTmed,
        'LTmax': LTmax,
        'active_states': TDE_hmm.get_active_K(),
        'P': TDE_hmm.get_P(),  # Transition probability matrix
        'network_fields': network_fields,
        'X_preproc': X_preproc,  # Original preprocessed data
        'X_embedded': X_embedded,  # TDE data
        'lags': lags,
        'state_means': state_means,
        'state_z_scores': state_z_scores,
        'state_global_norm': state_global_norm_scores,  
        'network_baselines': network_baselines,  
        'pca_model': pca_model
    }
    
    return model_data

def save_model(model_data, group, k, network_dir):
    """Save trained model to disk."""
    # Create group directory within network-specific directory
    group_dir = os.path.join(network_dir, group)
    os.makedirs(group_dir, exist_ok=True)
    
    # Create k-specific directory
    k_dir = os.path.join(group_dir, f'k{k}')
    os.makedirs(k_dir, exist_ok=True)
    
    # Define output path
    output_path = os.path.join(k_dir, 'model.pkl')
    
    logger.info(f"Saving model to {output_path}")
    
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)
        logger.info("Model saved successfully")
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise
    
def save_summary_metrics(model_data, group, k, timestamp, summary_file):
    """Save key metrics to a summary CSV file for easy model comparison."""
    import csv
    
    file_exists = os.path.isfile(summary_file)
    
    # Calculate additional summary metrics
    fo_mean = np.mean(model_data['FO'], axis=0)
    lt_mean = np.mean(model_data['LTmean'], axis=0)
    sr_mean = np.mean(model_data['SR'])
    
    # Format metrics as strings for CSV
    fo_str = ';'.join([f"{x:.4f}" for x in fo_mean])
    lt_str = ';'.join([f"{x:.2f}" for x in lt_mean])
    
    # Prepare row data
    row_data = {
        'timestamp': timestamp,
        'group': group,
        'k': k,
        'free_energy': f"{model_data['free_energy']:.2f}",
        'active_states': model_data['active_states'],
        'training_time': f"{model_data['training_time']:.2f}",
        'avg_switching_rate': f"{sr_mean:.6f}",
        'fractional_occupancy': fo_str,
        'lifetimes': lt_str
    }
    
    # Write to CSV
    with open(summary_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        
        # Write header if file doesn't exist
        if not file_exists:
            writer.writeheader()
            
        writer.writerow(row_data)
    
    logger.info(f"Summary metrics saved to {summary_file}")

def main():
    """Main function to run the TDE-HMM training pipeline for both network configurations"""
    logger.info("=== Starting TDE-HMM Model Training ===")
    overall_start_time = time.time()
    
    # Loop through network configurations
    for networks in [7, 8]:
        logger.info(f"\n=== Processing {networks}-network configuration ===")
        
        # Create network-specific directory
        network_dir = os.path.join(TRAINED_DIR, f'{networks}networks')
        os.makedirs(network_dir, exist_ok=True)
        
        # Create summary file path (no timestamp)
        summary_file = os.path.join(network_dir, "model_summary.csv")
        
        try:
            # Load preprocessed TDE data for this network configuration
            tde_data = load_tde_data(networks)
            
            # Train models with k=4 to k=5 for each group
            for k in range(4, 6):
                logger.info(f"\n--- Training models with k={k} ---")
                
                for group in ['controls', 'meditators']:
                    logger.info(f"\nProcessing {group} group...")
                    
                    # Train model
                    model_data = train_tde_model(tde_data, group, k)
                    
                    # Save model to network-specific directory
                    save_model(model_data, group, k, network_dir)
                    
                    # Save metrics to summary file (remove timestamp parameter)
                    save_summary_metrics(model_data, group, k, "", summary_file)
                    
                    # Log summary statistics
                    logger.info(f"Model summary for {group} (k={k}):")
                    logger.info(f"  Free energy: {model_data['free_energy']:.2f}")
                    logger.info(f"  Active states: {model_data['active_states']}/{k}")
                    logger.info(f"  Training time: {model_data['training_time']:.2f} seconds")
            
            # Report network configuration completion
            logger.info(f"\n=== {networks}-network configuration completed ===")
            logger.info(f"Trained models saved to: {network_dir}")
            logger.info(f"Summary metrics saved to: {summary_file}")
            
        except Exception as e:
            logger.error(f"Error in {networks}-network TDE-HMM training: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Report overall completion
    elapsed_time = time.time() - overall_start_time
    logger.info(f"\n=== All TDE-HMM training completed in {elapsed_time/60:.2f} minutes ===")
        
if __name__ == "__main__":
    main()