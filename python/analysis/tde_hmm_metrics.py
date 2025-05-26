"""
TDE-HMM Metrics Calculation Script for Meditation Analysis

This script calculates comprehensive metrics from trained TDE-HMM models
using the GLHMM library's utility functions.

The script:
1. Loads trained TDE-HMM models for different k values and standardization approaches
2. Calculates temporal metrics (FO, lifetimes, switching rates)
3. Computes state-specific network activations
4. Organizes results by group, standardization method, and network configuration
5. Saves metrics for further analysis and visualization

Usage:
  python tde_hmm_metrics.py --standardize [global|bygroup|persequence] [--networks [7|8]] [--kmin 4] [--kmax 5]

Important Note on TDE Handling:
    The Time-Delay Embedding (TDE) method expands the original network dimensions
    into a larger space that includes time-lagged copies. Our implementation:
    
    1. Uses 21 lags (-10 to +10) to create a full embedding space of ~168 dimensions
    2. Applies PCA to reduce to 16 dimensions (2× the original network count)
    3. Trains the HMM in this reduced TDE space
    
    For network-level metrics, we must use only the original network dimensions, not
    the TDE-expanded or PCA-reduced spaces.

References:
  Vidaurre et al. (2023). The Gaussian-linear hidden Markov model: A Python package.
  https://direct.mit.edu/imag/article/doi/10.1162/imag_a_00460/127499/The-Gaussian-Linear-Hidden-Markov-model-a-Python
  
  Shine et al., 2016 (DOI: 10.1016/j.neuroimage.2015.11.001): Uses correlation-based metrics for dynamic functional connectivity.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import logging
import time
from datetime import datetime
import argparse  
# Import GLHMM modules
from glhmm import glhmm, utils, auxiliary, statistics

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup paths
ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')

# Create base metrics directory
os.makedirs(METRICS_DIR, exist_ok=True)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='TDE-HMM Metrics Calculation with multiple standardization options')
    
    parser.add_argument('--standardize', type=str, choices=['global', 'bygroup', 'persequence'],
                        default='persequence', 
                        help='Standardization method to use for metrics calculation (default: persequence)')
    
    parser.add_argument('--networks', type=int, nargs='+', default=[7, 8],
                        help='Network configurations to process (default: 7 8)')
    
    parser.add_argument('--kmin', type=int, default=4,
                        help='Minimum number of states to analyze (default: 4)')
    
    parser.add_argument('--kmax', type=int, default=5,
                        help='Maximum number of states to analyze (default: 5)')
    
    args = parser.parse_args()
    
    # Convert single network to list for consistent processing
    if isinstance(args.networks, int):
        args.networks = [args.networks]
        
    return args

def load_trained_model(group, k, networks, standardize_method):
    """Load a trained TDE-HMM model and align X_preproc and vpath dimensions."""
    logger.info(f"Loading {group} model with k={k} for {networks}-network configuration ({standardize_method} standardization)...")
    
    # Define model path with correct directory structure
    model_dir = os.path.join(DATA_DIR, 'trained', standardize_method, f'{networks}networks', group, f'k{k}')
    model_path = os.path.join(model_dir, 'model.pkl')
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Add k to model_data if not present
        if 'k' not in model_data:
            model_data['k'] = k
        
        # Ensure X_preproc and vpath are aligned
        if 'X_preproc' in model_data and 'vpath' in model_data:
            X_preproc = model_data['X_preproc']
            vpath = model_data['vpath']
            
            # Convert vpath to 1D if it's one-hot encoded
            vpath_1d = statistics.generate_vpath_1D(vpath)
            
            # Handle potential mismatch between vpath and X_preproc lengths
            if len(vpath_1d) > X_preproc.shape[0]:
                logger.warning(f"Vpath length ({len(vpath_1d)}) exceeds X_preproc length ({X_preproc.shape[0]}). Trimming vpath at load time.")
                # Trim vpath
                model_data['vpath'] = vpath[:X_preproc.shape[0]]
                
            elif len(vpath_1d) < X_preproc.shape[0]:
                # Calculate how many timepoints will be lost
                lost_points = X_preproc.shape[0] - len(vpath_1d)
                lost_percent = (lost_points / X_preproc.shape[0]) * 100
                
                logger.warning(f"X_preproc length ({X_preproc.shape[0]}) exceeds vpath length ({len(vpath_1d)}). "
                              f"Trimming {lost_points} points ({lost_percent:.1f}% of data) at load time.")
                
                # Trim X_preproc
                model_data['X_preproc'] = X_preproc[:len(vpath_1d)]
            
            logger.info(f"Aligned data dimensions: X_preproc={model_data['X_preproc'].shape}, vpath={model_data['vpath'].shape}")
        else:
            logger.warning("Model data missing X_preproc or vpath. Alignment not performed.")
            
        logger.info(f"Loaded model with {model_data['active_states']}/{k} active states")
        return model_data
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def load_original_data(networks, standardize_method):
    """Load the original data to map back to network space."""
    logger.info(f"Loading original data for {networks}-network mapping ({standardize_method} standardization)...")
    
    # Load the preprocessed TDE data which contains network_fields
    data_path = os.path.join(DATA_DIR, 'processed', 'tde', f'tde_{networks}networks_{standardize_method}.pkl')
    
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    try:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        logger.info(f"Loaded original data with {len(data['network_fields'])} networks")
        return data
    
    except Exception as e:
        logger.error(f"Error loading original data: {str(e)}")
        raise

def cov_to_corr(cov_matrix):
    """Convert covariance matrix to correlation matrix with improved numerical stability."""
    # Add small value to diagonal to prevent division by zero
    epsilon = 1e-10
    diag_values = np.diag(cov_matrix).copy()
    
    # Ensure positive values on diagonal
    if np.any(diag_values <= 0):
        logger.warning(f"Found non-positive diagonal values in covariance matrix. Adding epsilon.")
        np.fill_diagonal(cov_matrix, np.maximum(diag_values, epsilon))
        diag_values = np.diag(cov_matrix)
    
    # Calculate correlation
    d = np.sqrt(diag_values)
    corr_matrix = np.zeros_like(cov_matrix)
    n = cov_matrix.shape[0]
    
    for i in range(n):
        for j in range(n):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                corr_matrix[i, j] = cov_matrix[i, j] / (d[i] * d[j])
    
    # Ensure values are in valid range [-1, 1]
    corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
    return corr_matrix

def calculate_state_means(model_data, network_fields):
    """Calculate mean activation patterns for each state, focusing on network dimensions."""
    logger.info("Calculating state mean activation patterns...")
    
    # Extract model components
    network_count = len(network_fields)
    k = model_data['k']
    
    # PRIMARY APPROACH: Use precomputed means from training (most accurate)
    if 'state_means' in model_data and model_data['state_means'] is not None and len(model_data['state_means']) > 0:
        logger.info("Using precomputed state means from training")
        state_means = []
        for state_mean in model_data['state_means']:
            # Ensure correct dimensions (first network_count values)
            if len(state_mean) >= network_count:
                network_mean = state_mean[:network_count]
            else:
                network_mean = state_mean
            state_means.append(network_mean)
        
        # Verify means were successfully retrieved
        if len(state_means) == k:
            logger.info(f"Retrieved {k} state means, each with {len(state_means[0])} dimensions")
            return state_means
    
    # FALLBACK APPROACH: Get means directly from HMM
    logger.info("Precomputed means not available - using HMM parameters")
    hmm = model_data['hmm']
    
    try:
        means = hmm.get_means()
        
        # Check if we need to transpose (we want states as first dimension)
        if means.shape[1] == k:
            means = means.T  # Convert to (n_states, n_variables)
            
        # Extract network-specific parts for each state
        state_means = []
        for state in range(k):
            state_mean = means[state]
            if len(state_mean) > network_count:
                logger.info(f"Extracting network means from expanded state mean vector (len={len(state_mean)})")
                network_mean = state_mean[:network_count]
            else:
                network_mean = state_mean
            state_means.append(network_mean)
        
        logger.info(f"Successfully retrieved {len(state_means)} state means from HMM parameters")
        return state_means
    except Exception as e:
        logger.warning(f"Error getting means from HMM: {str(e)}")
    
    # If no means can be found
    logger.error("Could not retrieve state means through any method")
    return None

def calculate_temporal_metrics(model_data, original_data, group):
    """Calculate temporal metrics using GLHMM utility functions with improved vpath handling."""
    logger.info("Calculating temporal metrics...")
    
    # Extract variables from model data
    hmm = model_data['hmm']
    Gamma = model_data['stc']  # State time courses (probabilities)
    vpath = model_data['vpath']
    indices = model_data['indices']
    k = model_data['k']
    
    # Handle vpath: convert to 1D if necessary
    vpath_1d = statistics.generate_vpath_1D(vpath)
    
    # Log unique state indices in vpath
    unique_states = np.unique(vpath_1d)
    logger.info(f"Unique state indices in vpath: {unique_states}")
    
    # Check for invalid state indices (states >= k)
    invalid_idx = np.where(vpath_1d >= k)[0]
    if len(invalid_idx) > 0:
        logger.warning(f"Found {len(invalid_idx)} time points with invalid state indices (>= {k})")
        for t in invalid_idx:
            # Remap to the most likely valid state based on Gamma
            vpath_1d[t] = np.argmax(Gamma[t, :k])
        logger.info("Remapped invalid states using Gamma probabilities")
    
    # Create vpath_2d (one-hot encoded) for lifetime calculation
    logger.info("Creating 2D Viterbi path for lifetime calculation...")
    vpath_2d = statistics.viterbi_path_to_stc(vpath_1d, k)
    
    # Calculate Fractional Occupancy (FO)
    logger.info("Calculating Fractional Occupancy (FO)...")
    FO = utils.get_FO(Gamma, indices)
    FO_mean = np.mean(FO, axis=0)
    FO_std = np.std(FO, axis=0)
    
    # Calculate Switching Rate
    logger.info("Calculating Switching Rate...")
    switching_rate = utils.get_switching_rate(Gamma, indices)
    
    # Calculate State Lifetimes
    logger.info("Calculating State Lifetimes...")
    try:
        LTmean, LTmed, LTmax = utils.get_life_times(vpath_2d, indices)
    except Exception as e:
        logger.warning(f"Error calculating lifetimes: {str(e)}")
        LTmean = np.zeros(k)
        LTmed = np.zeros(k)
        LTmax = np.zeros(k)
    
    # Extract state covariances
    logger.info("Extracting state covariances from model...")
    state_covs = [hmm.get_covariance_matrix(state) for state in range(k)]
    
    # Get network_fields from original_data
    network_fields = original_data['network_fields']
    
    # Compile temporal metrics
    temporal_metrics = {
        'FO': FO,
        'FO_mean': FO_mean,
        'FO_std': FO_std,
        'switching_rate': switching_rate,
        'lifetimes_mean': LTmean,
        'lifetimes_median': LTmed,
        'lifetimes_max': LTmax,
        'state_covs': state_covs,
        'network_fields': network_fields
    }
    
    return temporal_metrics

def calculate_network_interactions(model_data, network_fields):
    """Calculate direct interactions between the brain networks for each state using original data.
    
    Important Note on TDE Handling:
        Although the HMM models are trained on TDE-expanded data, and then PCA reduced data, 
        this function deliberately calculates covariances using only the original 
        network dimensions (~8 dimensions). Validation tests show that extracting 
        network submatrices from TDE covariances produces dramatically different values 
        (up to 30x scale differences) compared to direct calculation from network data.
        
        Direct calculation ensures:
        1. Proper network-level interpretability
        2. Comparable scales between covariance and correlation values
        3. Accurate representation of network interactions without TDE artifacts
        """
    logger.info("Calculating network-to-network interactions from original data...")
    
    interactions = {}
    network_count = len(network_fields)
    k = model_data['k']
    
    # Include all key networks for meditation analysis
    key_networks = ['VIS', 'SMN', 'DAN', 'VAN', 'LIM', 'FPN', 'DMN', 'SUB']

    # Get indices of the networks (only those that exist in data)
    network_indices = {}
    available_networks = []
    for network in key_networks:
        if network in network_fields:
            network_indices[network] = network_fields.index(network)
            available_networks.append(network)
    
    logger.info(f"Available networks for analysis: {', '.join(available_networks)}")
    
    # Get original data and state assignments
    X_preproc = model_data['X_preproc']
    vpath = model_data['vpath']
    
    # Use GLHMM utility to convert vpath to 1D if necessary
    vpath_1d = statistics.generate_vpath_1D(vpath)
    
    # Handle potential mismatch between vpath and X_preproc lengths
    if len(vpath_1d) > X_preproc.shape[0]:
        logger.warning(f"Vpath length ({len(vpath_1d)}) exceeds X_preproc length ({X_preproc.shape[0]}). Trimming vpath.")
        vpath_1d = vpath_1d[:X_preproc.shape[0]]
    elif len(vpath_1d) < X_preproc.shape[0]:
        # Calculate how many timepoints will be lost
        lost_points = X_preproc.shape[0] - len(vpath_1d)
        lost_percent = (lost_points / X_preproc.shape[0]) * 100
        
        logger.warning(f"X_preproc length ({X_preproc.shape[0]}) exceeds vpath length ({len(vpath_1d)}). "
                      f"Trimming {lost_points} points ({lost_percent:.1f}% of data).")
        
        # Only use the aligned portion - trim from the end which is typically where padding issues occur
        X_preproc = X_preproc[:len(vpath_1d), :]
        
    logger.info(f"Final aligned data: X_preproc={X_preproc.shape}, vpath={len(vpath_1d)}")

    # Extract only the first network_count dimensions which correspond to the networks
    if X_preproc.shape[1] > network_count:
        logger.info(f"Extracting first {network_count} dimensions from X_preproc (total dimensions: {X_preproc.shape[1]})")
        X_networks = X_preproc[:, :network_count]
    else:
        X_networks = X_preproc
    
    logger.info(f"Network data shape: {X_networks.shape}, using first {network_count} dimensions")
    
    # Calculate state-specific covariance matrices directly from original data
    state_covs = []
    for state in range(k):
        # Get points assigned to this state
        state_mask = vpath_1d == state
        if np.sum(state_mask) > 50:  # Ensure enough points for stable covariance (minimum 50 timepoints)
            state_data = X_networks[state_mask]
            # Calculate covariance directly from the network dimensions
            # rowvar=False means variables are in columns
            state_cov = np.cov(state_data, rowvar=False)
            state_covs.append(state_cov)
            logger.info(f"State {state+1}: calculated covariance matrix from {np.sum(state_mask)} timepoints")
        else:
            # Not enough points, use identity matrix as fallback
            logger.warning(f"State {state+1} has fewer than 50 points ({np.sum(state_mask)}). Using identity matrix.")
            state_covs.append(np.eye(network_count))
    
    # Calculate global variance metrics across all states for normalization
    all_variances = []
    for cov in state_covs:
        for network, idx in network_indices.items():
            if idx < network_count:
                all_variances.append(cov[idx, idx])
    
    # Calculate statistics for normalization
    if all_variances:
        mean_var = np.mean(all_variances)
        std_var = np.std(all_variances) or 1.0  # Use 1.0 if std is 0 to avoid division by zero
    else:
        mean_var = 1.0
        std_var = 1.0
        
    logger.info(f"Network variance stats - mean: {mean_var:.4f}, std: {std_var:.4f}")
    
    # For each state, calculate network correlations
    for state_idx, cov in enumerate(state_covs):
        # Process state covariance matrix
        logger.info(f"Processing state {state_idx+1} covariance matrix with shape {cov.shape}")
        
        # No need to extract submatrix - we calculated covariance directly on network data
        network_cov = cov
        
        # Convert covariance to correlation
        network_corr = cov_to_corr(network_cov)
        
        # Store all available network-to-network correlations
        state_interactions = {}
        
        # Store activations (variances)
        for network, idx in network_indices.items():
            if idx < network_count:
                raw_var = network_cov[idx, idx]
                # Store both raw and z-scored variance for activation
                state_interactions[f"{network}_activation_raw"] = raw_var
                # Z-score the variance for comparable activation measure
                state_interactions[f"{network}_activation"] = (raw_var - mean_var) / std_var
        
        # Store correlations between networks
        for i, net1 in enumerate(available_networks):
            idx1 = network_indices[net1]
            for j, net2 in enumerate(available_networks):
                idx2 = network_indices[net2]
                if i < j and idx1 < network_count and idx2 < network_count:
                    key = f"{net1}-{net2}"
                    state_interactions[key] = network_corr[idx1, idx2]
        
        # Calculate special meditation-relevant metrics
        
        # DMN anticorrelation with task-positive networks
        if all(net in available_networks for net in ['DMN', 'FPN', 'DAN', 'VAN']):
            dmn_idx = network_indices['DMN']
            fpn_idx = network_indices['FPN']
            dan_idx = network_indices['DAN']
            van_idx = network_indices['VAN']
            
            if all(idx < network_count for idx in [dmn_idx, fpn_idx, dan_idx, van_idx]):
                dmn_anticorr = (
                    network_corr[dmn_idx, fpn_idx] + 
                    network_corr[dmn_idx, dan_idx] + 
                    network_corr[dmn_idx, van_idx]
                ) / 3.0
                state_interactions['DMN_anticorrelation'] = dmn_anticorr
        
        # DMN-LIM correlation (relevant for emotional regulation in meditation)
        if all(net in available_networks for net in ['DMN', 'LIM']):
            dmn_idx = network_indices['DMN']
            limbic_idx = network_indices['LIM']
            
            if all(idx < network_count for idx in [dmn_idx, limbic_idx]):
                state_interactions['DMN-LIM_correlation'] = network_corr[dmn_idx, limbic_idx]
        
        # SMN metrics for breath awareness
        if 'SMN' in available_networks:
            smn_idx = network_indices['SMN']
            
            # Calculate SMN correlations with attention networks
            smn_attention_corr = []
            for att_net in ['DAN', 'VAN']:
                if att_net in available_networks:
                    att_idx = network_indices[att_net]
                    if all(idx < network_count for idx in [smn_idx, att_idx]):
                        smn_attention_corr.append(network_corr[smn_idx, att_idx])
            
            if smn_attention_corr:
                state_interactions['SMN-attention_correlation'] = np.mean(smn_attention_corr)
        
        interactions[state_idx] = state_interactions
    
    return interactions

def calculate_state_transitions(model_data):
    """Calculate state transition probabilities."""
    logger.info("Calculating state transition metrics...")
    
    # Get transition matrix from the model
    P = model_data['P']
    
    # Calculate self-transitions (stays)
    self_transitions = np.diag(P)
    
    transition_metrics = {
        'P': P,  # Raw transition matrix
        'self_transitions': self_transitions
    }
    
    return transition_metrics

def calculate_all_metrics(model_data, original_data, group):
    """Calculate all metrics for a given model."""
    logger.info(f"Calculating metrics for {group} model with k={model_data['k']}...")
    
    # Calculate different types of metrics
    temporal_metrics = calculate_temporal_metrics(model_data, original_data, group)
    
    # Calculate state means specifically for networks
    state_means = calculate_state_means(model_data, temporal_metrics['network_fields'])
    if state_means is not None:
        temporal_metrics['state_means'] = state_means
    else:
        logger.warning("State means not available - some visualizations may be limited")
    
    # Pass model_data directly to calculate_network_interactions
    network_interactions = calculate_network_interactions(model_data, temporal_metrics['network_fields'])
    
    # Calculate state transitions
    transition_metrics = calculate_state_transitions(model_data)
    
    # Compile all metrics
    all_metrics = {
        'temporal_metrics': temporal_metrics,
        'transition_metrics': transition_metrics,
        'network_interactions': network_interactions,
        'k': model_data['k'],
        'group': group,
        'networks': len(temporal_metrics['network_fields']),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return all_metrics

def save_metrics(metrics, group, k, networks, standardize_method):
    """Save metrics to disk."""
    # Define network-specific output directory with standardization method
    std_dir = os.path.join(METRICS_DIR, standardize_method)
    network_metrics_dir = os.path.join(std_dir, f'{networks}networks')
    group_dir = os.path.join(network_metrics_dir, group)
    
    # Create directories if they don't exist
    os.makedirs(group_dir, exist_ok=True)
    
    # Define output path
    output_path = os.path.join(group_dir, f'k{k}_metrics.pkl')
    
    logger.info(f"Saving metrics to {output_path}")
    
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(metrics, f)
        logger.info("Metrics saved successfully")
    except Exception as e:
        logger.error(f"Error saving metrics: {str(e)}")
        raise

def main():
    """Main function to run the TDE-HMM metrics calculation pipeline"""
    # Parse command line arguments
    args = parse_arguments()
    
    logger.info("=== Starting TDE-HMM Metrics Calculation ===")
    logger.info(f"Standardization method: {args.standardize}")
    logger.info(f"Network configurations: {args.networks}")
    logger.info(f"State range: k={args.kmin} to k={args.kmax}")
    
    overall_start_time = time.time()
    
    # Set up base directory for standardization method
    std_dir = os.path.join(METRICS_DIR, args.standardize)
    os.makedirs(std_dir, exist_ok=True)

    # Process specified network configurations
    for networks in args.networks:
        logger.info(f"\n=== Processing {networks}-network configuration ===")
        
        # Create network-specific metrics directory
        network_metrics_dir = os.path.join(std_dir, f'{networks}networks')
        os.makedirs(network_metrics_dir, exist_ok=True)
        os.makedirs(os.path.join(network_metrics_dir, 'controls'), exist_ok=True)
        os.makedirs(os.path.join(network_metrics_dir, 'meditators'), exist_ok=True)
        
        try:
            # Load the original data for network mapping
            original_data = load_original_data(networks, args.standardize)
            
            # Log available networks in the data
            logger.info(f"Available networks in {networks}-network data: {original_data['network_fields']}")
            
            # Calculate metrics for specified k range
            for k in range(args.kmin, args.kmax + 1):
                logger.info(f"\n--- Processing {networks}-network models with k={k} ---")
                
                for group in ['controls', 'meditators']:
                    logger.info(f"\nCalculating metrics for {group} group...")
                    
                    # Load the trained model with standardization method
                    model_data = load_trained_model(group, k, networks, args.standardize)
                    
                    # Calculate all metrics
                    all_metrics = calculate_all_metrics(model_data, original_data, group)
                    
                    # Save metrics with standardization method
                    save_metrics(all_metrics, group, k, networks, args.standardize)
                    
                    # Log key meditation-relevant metrics
                    logger.info(f"Metrics summary for {group} (k={k}):")
                    logger.info(f"  State distribution: {[f'{fo:.3f}' for fo in all_metrics['temporal_metrics']['FO_mean']]}")
                    
                    # Get meditation-relevant network metrics
                    network_interactions = all_metrics['network_interactions']
                    for state_idx in sorted(network_interactions.keys()):
                        interactions = network_interactions[state_idx]
                        
                        # Get key metrics for meditation state inference
                        metrics_of_interest = {
                            'DMN': interactions.get('DMN_activation', 'NA'),
                            'SMN': interactions.get('SMN_activation', 'NA'),
                            'DMN-Task': interactions.get('DMN_anticorrelation', 'NA'),
                            'LIM-DMN': interactions.get('DMN-LIM_correlation', 'NA')
                        }
                        
                        logger.info(f"  State {state_idx+1}: {metrics_of_interest}")
            
            # Report completion for this network configuration
            logger.info(f"\n=== {networks}-network metrics calculation completed ===")
            logger.info(f"Metrics saved to: {network_metrics_dir}")
                    
        except Exception as e:
            logger.error(f"Error in {networks}-network TDE-HMM metrics calculation: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Report overall completion
    elapsed_time = time.time() - overall_start_time
    logger.info(f"\n=== All TDE-HMM metrics calculation completed in {elapsed_time/60:.2f} minutes ===")
    logger.info(f"Metrics saved to: {METRICS_DIR}")
        
if __name__ == "__main__":
    main()