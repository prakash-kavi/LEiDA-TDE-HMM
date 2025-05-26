"""
Time-Delay Embedding (TDE) Preprocessing for GLHMM Analysis

This script processes leading eigenvector data for HMM analysis using the GLHMM library's
time-delay embedding approach as described in Vidaurre et al. (2018).

The script:
1. Loads eigenvector data from MAT files
2. Standardizes the data
3. Applies time-delay embedding using GLHMM's preproc.build_data_tde function
4. Visualizes the embedded data
5. Saves the processed data for subsequent HMM analysis

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
RAW_DATA_PATH = os.path.join(DATA_DIR, 'preprocessed','eigenvectors_data_yeo.mat')
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
        
def filter_networks(data, networks_to_keep=None, networks_to_exclude=None):
    """Filter networks from Yeo7 + subcortical = 8 subnetworks data to Yeo7 subnetworks only. """
    logger.info("Filtering networks...")
    
    if networks_to_keep is None and networks_to_exclude is None:
        logger.warning("No filtering criteria provided, returning original data")
        return data
    
    # Get indices of networks to keep
    keep_indices = []
    filtered_network_fields = []
    
    if networks_to_keep is not None:
        # Keep only specified networks
        for i, network in enumerate(data['network_fields']):
            if network in networks_to_keep:
                keep_indices.append(i)
                filtered_network_fields.append(network)
    else:
        # Keep all networks except excluded ones
        for i, network in enumerate(data['network_fields']):
            if network not in networks_to_exclude:
                keep_indices.append(i)
                filtered_network_fields.append(network)
    
    logger.info(f"Keeping networks: {filtered_network_fields}")
    logger.info(f"Removed networks: {[n for n in data['network_fields'] if n not in filtered_network_fields]}")
    
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

def prepare_data_for_tde(sequences):
    """Prepare data for TDE by concatenating and creating indices"""
    # Concatenate all sequences
    X_concat = np.vstack(sequences)
    
    # Create indices for each sequence
    indices = statistics.get_indices_from_list(sequences)
    
    return X_concat, indices

def apply_tde_with_glhmm(data, group='controls', lags=None, pca_components=None):
    """Apply Time-Delay Embedding using GLHMM's preproc.build_data_tde function"""
    logger.info(f"Applying TDE to {group} data with GLHMM...")
    
    # Get sequences for this group
    sequences = data[f'{group}_sequences']
    
    # Prepare data (concatenate and create indices)
    X, indices = prepare_data_for_tde(sequences)
    
    # Standardize the data per sequence
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
    # Update: Handle only 2 return values (no pca_model)
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

def main():
    """Main function to run the TDE preprocessing pipeline"""
    logger.info("=== Starting Time-Delay Embedding Preprocessing with GLHMM ===")
    start_time = time.time()
    
    try:
        # Load raw eigenvector data
        raw_data = load_eigenvector_data(RAW_DATA_PATH)
        
        data = raw_data  # Use all networks
        
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
            data, 'controls', lags=lags, pca_components=pca_components
        )
        
        # Process meditators
        meditators_tde = apply_tde_with_glhmm(
            data, 'meditators', lags=lags, pca_components=pca_components
        )
        
        # Create combined processed data structure
        processed_data = {
            'controls_sequences': controls_tde['embedded_sequences'],
            'meditators_sequences': meditators_tde['embedded_sequences'],
            'network_fields': data['network_fields'],
            'lags': lags,
            'pca_components': pca_components,
            'tde_parameters': {
                'controls_indices_original': controls_tde['indices'],
                'controls_indices_tde': controls_tde['indices_tde'],
                'meditators_indices_original': meditators_tde['indices'],
                'meditators_indices_tde': meditators_tde['indices_tde']
            }
        }
        
        # Save the processed data
        output_path = os.path.join(TDE_DIR, 'tde_8networks_persequence.pkl')
        save_processed_data(processed_data, output_path)
        
        # Create visualizations for a few subjects
        for subj_idx in [0, 5, 10, 15]:
            if subj_idx < len(data['controls_sequences']):
                visualize_embedded_data(data, controls_tde, 'controls', subj_idx)
                visualize_state_time_course(data, controls_tde, 'controls', subj_idx)
            
            if subj_idx < len(data['meditators_sequences']):
                visualize_embedded_data(data, meditators_tde, 'meditators', subj_idx)
                visualize_state_time_course(data, meditators_tde, 'meditators', subj_idx)
        
        # Report overall completion
        elapsed_time = time.time() - start_time
        logger.info(f"\n=== TDE preprocessing completed in {elapsed_time:.2f} seconds ===")
        logger.info(f"Processed data saved to: {output_path}")
        logger.info(f"Visualizations saved to: {RESULTS_DIR}")
        
    except Exception as e:
        logger.error(f"Error in TDE preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    main()