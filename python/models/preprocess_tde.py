"""
Time-Delay Embedding (TDE) Preprocessing for GLHMM Analysis

This script processes leading eigenvector data for HMM analysis using the GLHMM library's
time-delay embedding approach as described in Vidaurre et al. (2018).

Features:
1. Multiple standardization options: global, by-group, or per-sequence
2. Network configuration options: all 8 networks or Yeo7 (excluding subcortical)
3. Optional visualization
4. Consistent output naming for model selection

Usage:
  python preprocess_tde.py --standardize [global|bygroup|persequence] --networks [7|8] [--skip-visualization]
  separate viusualization for tde in visualization/tde_visualization.py
  
References:
- Vidaurre, D., Hunt, L.T., Quinn, A.J. et al. (2018). Spontaneous cortical activity
  transiently organises into frequency specific phase-coupling networks. Nature 
  Communications, 9, 2987. https://doi.org/10.1038/s41467-018-05316-z
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pickle
import logging
import time
import argparse

# Import GLHMM modules for TDE processing
from glhmm import preproc, graphics, statistics

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup paths
ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
TDE_DIR = os.path.join(PROCESSED_DIR, 'tde')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'preprocessed', 'eigenvectors_data_yeo.mat')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results', 'visualizations', 'tde')

# Create necessary directories
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(TDE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_eigenvector_data(data_path):
    """ Load eigenvector data from MAT file. """
    logger.info(f"Loading eigenvector data from {data_path}")
    
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Load MAT file
    try:
        mat_data = loadmat(data_path, squeeze_me=True, struct_as_record=False)
        eigenvectors_data = mat_data['eigenvectors_data_yeo']
        
        # Extract data from MATLAB structure
        data = {
            'controls_sequences': [],
            'meditators_sequences': [],
            'network_fields': list(eigenvectors_data.network_fields),
            'timepoints_per_subject': eigenvectors_data.info.timepoints_per_subject
        }
        
        # Extract controls data (each subject is a separate sequence)
        for i in range(eigenvectors_data.controls.n_subjects):
            data['controls_sequences'].append(eigenvectors_data.controls.subjects[i])
            
        # Extract meditators data (each subject is a separate sequence)
        for i in range(eigenvectors_data.meditators.n_subjects):
            data['meditators_sequences'].append(eigenvectors_data.meditators.subjects[i])
        
        logger.info(f"Loaded data: Controls ({len(data['controls_sequences'])} subjects), "
                   f"Meditators ({len(data['meditators_sequences'])} subjects)")
        logger.info(f"Networks: {data['network_fields']}")
        
        return data
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise
        
def filter_networks(data, num_networks=8):
    """Filter networks to use all 8 or just Yeo7 (excluding subcortical)."""
    logger.info(f"Configuring for {num_networks} networks...")
    
    if num_networks == 8:
        # Use all networks
        logger.info("Using all 8 networks (including subcortical)")
        return data
    
    elif num_networks == 7:
        # Exclude subcortical network
        networks_to_exclude = ['SUB']
        
        # Get indices of networks to keep
        keep_indices = []
        filtered_network_fields = []
        
        for i, network in enumerate(data['network_fields']):
            if network not in networks_to_exclude:
                keep_indices.append(i)
                filtered_network_fields.append(network)
        
        logger.info(f"Using 7 networks (excluding subcortical)")
        logger.info(f"Keeping networks: {filtered_network_fields}")
        
        # Filter data
        filtered_data = {
            'controls_sequences': [],
            'meditators_sequences': [],
            'network_fields': filtered_network_fields,
            'timepoints_per_subject': data['timepoints_per_subject']
        }
        
        # Filter controls data
        for seq in data['controls_sequences']:
            filtered_data['controls_sequences'].append(seq[:, keep_indices])
        
        # Filter meditators data
        for seq in data['meditators_sequences']:
            filtered_data['meditators_sequences'].append(seq[:, keep_indices])
        
        return filtered_data
    else:
        logger.error(f"Invalid network configuration: {num_networks}")
        raise ValueError(f"Network configuration must be 7 or 8, got {num_networks}")

def prepare_data_for_tde(sequences):
    """Prepare data for TDE by concatenating and creating indices"""
    # Concatenate all sequences
    X_concat = np.vstack(sequences)
    
    # Create indices for each sequence
    indices = statistics.get_indices_from_list(sequences)
    
    return X_concat, indices

def standardize_global(X, indices):
    """Apply global standardization to the entire dataset"""
    logger.info("Applying global standardization")
    
    # Calculate global mean and std
    global_mean = np.mean(X, axis=0)
    global_std = np.std(X, axis=0) + 1e-6  # Add small epsilon to avoid division by zero
    
    # Apply standardization
    X_std = (X - global_mean) / global_std
    
    return X_std

def standardize_by_group(X, indices):
    """Apply standardization separately within this group"""
    logger.info("Applying within-group standardization")
    
    # Use the entire data as one group for standardization
    group_mean = np.mean(X, axis=0)
    group_std = np.std(X, axis=0) + 1e-6  # Add small epsilon to avoid division by zero
    
    # Apply standardization
    X_std = (X - group_mean) / group_std
    
    return X_std

def apply_tde_with_glhmm(data, group='controls', standardize_method='persequence', 
                        lags=None, pca_components=None):
    """Apply Time-Delay Embedding using GLHMM with specified standardization method"""
    logger.info(f"Applying TDE to {group} data with standardization: {standardize_method}")
    
    # Get sequences for this group
    sequences = data[f'{group}_sequences']
    
    # Prepare data (concatenate and create indices)
    X, indices = prepare_data_for_tde(sequences)
    
    # Apply appropriate standardization method
    if standardize_method == 'global':
        # Global standardization was applied earlier, use X directly
        X_preproc = X
    elif standardize_method == 'bygroup':
        # Standardize within this group
        X_preproc = standardize_by_group(X, indices)
    else:  # 'persequence'
        # Use GLHMM's per-sequence standardization (default)
        X_preproc = preproc.preprocess_data(X, indices)[0]
    
    # Define lags if not provided (default from the example: (-L to L) with step S)
    if lags is None:
        L = 10  # Maximum lag
        S = 1   # Step size (lag interval)
        lags = np.arange(-L, L+1, S)
    
    logger.info(f"Using lags: {lags}")
    
    # Set PCA components if not specified (default: 2x number of regions)
    if pca_components is None:
        n_regions = sequences[0].shape[1]
        pca_components = n_regions * 2
    
    logger.info(f"Using PCA components: {pca_components}")
    
    # Apply TDE with GLHMM's build_data_tde function
    X_embedded, indices_tde = preproc.build_data_tde(
        X_preproc, indices, lags=lags, pca=pca_components
    )
    
    logger.info(f"TDE applied: Input shape {X_preproc.shape}, Output shape {X_embedded.shape}")
    
    # Extract embedded sequences
    embedded_sequences = []
    for i in range(len(sequences)):
        start_idx, end_idx = indices_tde[i]
        embedded_sequences.append(X_embedded[start_idx:end_idx])
        logger.info(f"  Subject {i+1}: {sequences[i].shape} -> {embedded_sequences[i].shape}")
    
    return {
        'X_preproc': X_preproc,
        'X_embedded': X_embedded,
        'indices': indices,
        'indices_tde': indices_tde,
        'embedded_sequences': embedded_sequences,
        'lags': lags,
        'pca_components': pca_components
    }

def visualize_embedded_data(data, group_data, group, subject_idx=0):
    """Visualize original and TDE-processed data for a subject"""
    logger.info(f"Visualizing {group} subject {subject_idx+1} data...")
    
    # Get original and preprocessed data
    orig_seq = data[f'{group}_sequences'][subject_idx]
    preproc_data = group_data['X_preproc']
    
    # Get indices for this subject
    start_idx, end_idx = group_data['indices'][subject_idx]
    preproc_seq = preproc_data[start_idx:end_idx]
    
    # Get embedded data for this subject
    start_tde_idx, end_tde_idx = group_data['indices_tde'][subject_idx]
    embedded_seq = group_data['X_embedded'][start_tde_idx:end_tde_idx]
    
    # Create figure with 3 plots: original, preprocessed, and embedded
    plt.figure(figsize=(18, 12))
    
    # Plot original data
    plt.subplot(3, 1, 1)
    plt.imshow(orig_seq.T, aspect='auto', cmap='RdBu_r')
    plt.title(f'Original {group.capitalize()} Data (Subject {subject_idx+1})')
    plt.xlabel('Time (TRs)')
    plt.ylabel('Network')
    plt.yticks(range(len(data['network_fields'])), data['network_fields'])
    plt.colorbar(label='Eigenvector Value')
    
    # Plot preprocessed data
    plt.subplot(3, 1, 2)
    plt.imshow(preproc_seq.T, aspect='auto', cmap='RdBu_r')
    plt.title(f'Preprocessed {group.capitalize()} Data (Subject {subject_idx+1})')
    plt.xlabel('Time (TRs)')
    plt.ylabel('Network')
    plt.yticks(range(len(data['network_fields'])), data['network_fields'])
    plt.colorbar(label='Standardized Value')
    
    # Plot embedded data (PCA components)
    plt.subplot(3, 1, 3)
    plt.imshow(embedded_seq.T, aspect='auto', cmap='RdBu_r')
    plt.title(f'TDE-Embedded {group.capitalize()} Data with PCA (Subject {subject_idx+1})')
    plt.xlabel('Time (TRs)')
    plt.ylabel('PCA Component')
    plt.colorbar(label='Component Value')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(RESULTS_DIR, f'{group}_subject_{subject_idx+1}_tde.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    logger.info(f"Visualization saved to: {output_path}")

def visualize_state_time_course(data, group_data, group, subject_idx=0):
    """Plot signal and original data together to show time courses"""
    try:
        logger.info(f"Visualizing time courses for {group} subject {subject_idx+1}...")
        
        # Get original data and indices
        orig_seq = data[f'{group}_sequences'][subject_idx]
        start_idx, end_idx = group_data['indices'][subject_idx]
        
        # Get preprocessed data for this subject
        preproc_seq = group_data['X_preproc'][start_idx:end_idx]
        
        # Plot the first 1000 time points (or fewer if sequence is shorter)
        plot_length = min(1000, preproc_seq.shape[0])
        
        # Create the figure using GLHMM's plotting functionality
        plt.figure(figsize=(15, 8))
        
        # Use the first network/component as the signal
        signal = preproc_seq[:plot_length, 0].copy()
        
        # Generate a dummy state time course (this is just for visualization)
        dummy_states = np.zeros((plot_length, 1))
        for i in range(plot_length):
            dummy_states[i] = i % 4  # Cycle through 4 dummy states
            
        # Plot using GLHMM's graphics
        graphics.plot_vpath(
            dummy_states, 
            signal=signal,
            title=f"{group.capitalize()} - Signal example (Subject {subject_idx+1})"
        )
        
        plt.tight_layout()
        output_path = os.path.join(RESULTS_DIR, f'{group}_subject_{subject_idx+1}_time_course.png')
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        logger.info(f"Time course visualization saved to: {output_path}")
        
    except Exception as e:
        logger.warning(f"Could not create time course visualization: {str(e)}")

def save_processed_data(processed_data, output_path):
    """Save processed data to file"""
    logger.info(f"Saving processed data to {output_path}")
    
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(processed_data, f)
        logger.info("Data saved successfully")
    except Exception as e:
        logger.error(f"Error saving data: {str(e)}")
        raise

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='TDE Preprocessing for GLHMM Analysis')
    
    parser.add_argument('--standardize', type=str, choices=['global', 'bygroup', 'persequence'],
                        default='persequence', help='Standardization method (default: persequence)')
    
    parser.add_argument('--networks', type=int, choices=[7, 8], default=8,
                        help='Number of networks to use: 7 (Yeo networks) or 8 (includes subcortical)')
    
    parser.add_argument('--skip-visualization', action='store_true',
                        help='Skip visualization generation')
    
    args = parser.parse_args()
    return args

def apply_global_standardization(data):
    """Apply global standardization across all subjects and groups"""
    logger.info("Preparing global standardization...")
    
    # Concatenate all sequences from both groups
    all_sequences = data['controls_sequences'] + data['meditators_sequences']
    all_data = np.vstack(all_sequences)
    
    # Calculate global mean and std
    global_mean = np.mean(all_data, axis=0)
    global_std = np.std(all_data, axis=0) + 1e-6
    
    # Apply standardization to each group
    for group in ['controls', 'meditators']:
        for i in range(len(data[f'{group}_sequences'])):
            data[f'{group}_sequences'][i] = (data[f'{group}_sequences'][i] - global_mean) / global_std
    
    logger.info("Global standardization applied to all subjects")
    return data

def main():
    """Main function to run the TDE preprocessing pipeline"""
    # Parse command line arguments
    args = parse_arguments()
    
    logger.info("=== Starting TDE Preprocessing with Configuration ===")
    logger.info(f"Standardization method: {args.standardize}")
    logger.info(f"Network configuration: {args.networks} networks")
    logger.info(f"Visualization: {'skipped' if args.skip_visualization else 'enabled'}")
    
    start_time = time.time()
    
    try:
        # Load raw eigenvector data
        raw_data = load_eigenvector_data(RAW_DATA_PATH)
        
        # Filter networks based on configuration
        data = filter_networks(raw_data, args.networks)
        
        # For global standardization, apply it here across all subjects
        if args.standardize == 'global':
            data = apply_global_standardization(data)
        
        # Define TDE parameters following Vidaurre's method
        L = 10  # Maximum lag window
        S = 1   # Step size
        lags = np.arange(-L, L+1, S)
        
        # Calculate PCA components (2x number of networks)
        n_networks = len(data['network_fields'])
        pca_components = n_networks * 2
        
        logger.info(f"TDE Parameters: Lag window -L to L = -{L} to {L}, step size = {S}")
        logger.info(f"PCA components: {pca_components}")
        
        # Process controls
        controls_tde = apply_tde_with_glhmm(
            data, 'controls', standardize_method=args.standardize,
            lags=lags, pca_components=pca_components
        )
        
        # Process meditators
        meditators_tde = apply_tde_with_glhmm(
            data, 'meditators', standardize_method=args.standardize,
            lags=lags, pca_components=pca_components
        )
        
        # Create combined processed data structure
        processed_data = {
            'controls_sequences': controls_tde['embedded_sequences'],
            'meditators_sequences': meditators_tde['embedded_sequences'],
            'network_fields': data['network_fields'],
            'lags': lags,
            'pca_components': pca_components,
            'standardization_method': args.standardize,
            'n_networks': args.networks,
            'tde_parameters': {
                'controls_indices_original': controls_tde['indices'],
                'controls_indices_tde': controls_tde['indices_tde'],
                'meditators_indices_original': meditators_tde['indices'],
                'meditators_indices_tde': meditators_tde['indices_tde']
            }
        }
        
        # Save the processed data with consistent naming
        output_filename = f'tde_{args.networks}networks_{args.standardize}.pkl'
        output_path = os.path.join(TDE_DIR, output_filename)
        save_processed_data(processed_data, output_path)
        
        # Create visualizations if enabled
        if not args.skip_visualization:
            logger.info("Generating visualizations...")
            for subj_idx in [0, 5, 10, 15]:
                if subj_idx < len(data['controls_sequences']):
                    visualize_embedded_data(data, controls_tde, 'controls', subj_idx)
                    visualize_state_time_course(data, controls_tde, 'controls', subj_idx)
                
                if subj_idx < len(data['meditators_sequences']):
                    visualize_embedded_data(data, meditators_tde, 'meditators', subj_idx)
                    visualize_state_time_course(data, meditators_tde, 'meditators', subj_idx)
        else:
            logger.info("Skipping visualization generation")
        
        # Report overall completion
        elapsed_time = time.time() - start_time
        logger.info(f"\n=== TDE preprocessing completed in {elapsed_time:.2f} seconds ===")
        logger.info(f"Processed data saved to: {output_path}")
        if not args.skip_visualization:
            logger.info(f"Visualizations saved to: {RESULTS_DIR}")
            
    except Exception as e:
        logger.error(f"Error in TDE preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    main()