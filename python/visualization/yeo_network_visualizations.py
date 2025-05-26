"""
Yeo Network Visualizations for TDE-HMM Models

This script creates specialized visualizations for TDE-HMM models
focusing on the 8 Yeo networks (VIS, SMN, DAN, VAN, LIM, FPN, DMN, SUB):

1. Network Activation with Viterbi State Plot:
   - Individual subject level
   - Shows network activation patterns over time
   - Overlays state assignments from Viterbi path

2. Network-Focused Correlation Matrices:
   - Group level
   - 8×8 correlation matrices between Yeo networks
   - One matrix per state per model
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import seaborn as sns
import pickle
from matplotlib.colors import LinearSegmentedColormap
import time
from datetime import datetime

# Import GLHMM modules
try:
    from glhmm import auxiliary
except ImportError:
    print("Warning: Could not import glhmm modules. Some functionality may be limited.")

# Full network names for better labeling
NETWORK_NAMES = {
    'VIS': 'Visual',
    'SMN': 'Somatomotor',
    'DAN': 'Dorsal Attention',
    'VAN': 'Ventral Attention',
    'LIM': 'Limbic',
    'FPN': 'Frontoparietal',
    'DMN': 'Default Mode',
    'SUB': 'Subcortical'
}

# Setup paths
ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
TRAINED_DIR = os.path.join(DATA_DIR, 'trained', 'glhmm_tde')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
TDE_DATA_DIR = os.path.join(PROCESSED_DIR, 'tde')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
VIS_DIR = os.path.join(RESULTS_DIR, 'visualizations', 'yeo_networks')

# Create visualization directories
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(os.path.join(VIS_DIR, 'controls'), exist_ok=True)
os.makedirs(os.path.join(VIS_DIR, 'meditators'), exist_ok=True)

def load_trained_model(group, k):
    """Load a trained TDE-HMM model."""
    print(f"Loading {group} model with k={k}...")
    
    # Define model path
    model_path = os.path.join(TRAINED_DIR, group, f'k{k}', 'model.pkl')
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Add k to model_data if not present
        if 'k' not in model_data:
            model_data['k'] = k
            
        print(f"Loaded model with {model_data['active_states']}/{k} active states")
        return model_data
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def save_figure(fig, filepath, dpi=300):
    """Helper function to safely save a figure and verify it was saved."""
    try:
        # Make sure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save figure
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        
        # Verify file was saved
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            print(f"Saved: {filepath}")
            return True
        else:
            print(f"Warning: Failed to save {filepath} or file is empty")
            return False
    except Exception as e:
        print(f"Error saving figure to {filepath}: {str(e)}")
        return False

def cov_to_corr(cov_matrix):
    """Convert covariance matrix to correlation matrix."""
    d = np.sqrt(np.diag(cov_matrix))
    corr_matrix = cov_matrix / np.outer(d, d)
    # Ensure values are in valid range [-1, 1]
    corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
    return corr_matrix

def get_network_correlation_matrix(cov_matrix, n_networks=8):
    """
    Extract and convert network-level correlation matrix from covariance.
    For TDE data, extract only the lag=0 correlations between networks.
    """
    # Get the full correlation matrix
    corr_matrix = cov_to_corr(cov_matrix)
    
    # Extract the 8x8 submatrix representing the correlations
    # between the 8 networks at lag=0
    network_corr = corr_matrix[:n_networks, :n_networks]
    
    return network_corr

def visualize_network_correlations(model_data, group, k, state_idx):
    """
    Create visualization of network correlations for a specific state.
    
    Args:
        model_data: Dictionary containing model parameters
        group: 'controls' or 'meditators'
        k: Number of states in the model
        state_idx: Index of the state to visualize
    """
    hmm = model_data['hmm']
    network_fields = model_data['network_fields']
    
    # Create output directory
    output_dir = os.path.join(VIS_DIR, group, f'k{k}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Get covariance matrix for this state
    cov = hmm.get_covariance_matrix(state_idx)
    
    # Extract network-level correlation matrix
    network_corr = get_network_correlation_matrix(cov, len(network_fields))
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot correlation matrix with network labels
    ax = sns.heatmap(
        network_corr, 
        vmin=-1, vmax=1, 
        cmap='coolwarm', 
        center=0,
        square=True,
        annot=True,
        fmt='.2f',
        xticklabels=network_fields,
        yticklabels=network_fields
    )
    
    # Improve axis labels
    plt.title(f'{group.capitalize()} - State {state_idx+1} Network Correlations (k={k})')
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f'{group}_k{k}_state{state_idx+1}_network_correlations.png')
    save_figure(plt.gcf(), output_path)
    plt.close()

def extract_subject_data(model_data, subject_idx=0):
    """
    Extract time series data for a single subject from the model data.
    
    Args:
        model_data: Dictionary containing model parameters
        subject_idx: Index of the subject to extract
        
    Returns:
        tuple: (network_data, vpath_subject, start_idx, end_idx)
    """
    X_preproc = model_data['X_preproc']
    indices = model_data['indices']
    vpath = model_data['vpath']
    network_fields = model_data['network_fields']
    
    # Make sure subject_idx is valid
    if subject_idx >= len(indices):
        print(f"Warning: Subject index {subject_idx} out of range. Using subject 0.")
        subject_idx = 0
    
    # Get subject indices
    start_idx, end_idx = indices[subject_idx]
    subj_length = end_idx - start_idx
    
    # Extract network data (first 8 dimensions/channels)
    network_data = X_preproc[start_idx:end_idx, :len(network_fields)]
    
    # Extract Viterbi path for this subject
    T = auxiliary.get_T(indices)
    
    try:
        # If padGamma is available, use it to get aligned Viterbi path
        if 'lags' in model_data:
            options_tde = {'embeddedlags': list(model_data['lags'])}
            paddedVP = auxiliary.padGamma(vpath, T, options_tde)
            vpath_subject = paddedVP[start_idx:end_idx]
        else:
            # Otherwise extract directly
            vpath_subject = vpath[start_idx:end_idx]
    except Exception as e:
        print(f"Warning: Could not extract Viterbi path - {str(e)}")
        vpath_subject = None
    
    return network_data, vpath_subject, start_idx, end_idx

def visualize_network_heatmap_with_states(model_data, group, k, subject_idx=0, max_timepoints=360):
    """
    Create a heatmap visualization of network activity with state boundaries.
    """
    # Extract subject data
    network_data, vpath_subject, start_idx, end_idx = extract_subject_data(model_data, subject_idx)
    
    # Limit timepoints
    if network_data.shape[0] > max_timepoints:
        network_data = network_data[:max_timepoints]
        if vpath_subject is not None:
            vpath_subject = vpath_subject[:max_timepoints]
    
    # Z-score the time series data
    network_z = (network_data - np.mean(network_data, axis=0)) / np.std(network_data, axis=0)
    
    # Create output directory
    output_dir = os.path.join(VIS_DIR, group, f'k{k}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create color mapping for states
    state_colors = plt.cm.tab10(np.linspace(0, 1, k))
    
    # Create figure
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[8, 1], hspace=0.05)
    
    # Convert vpath to scalar values if they're arrays
    if vpath_subject is not None:
        vpath_scalar = np.zeros(len(vpath_subject), dtype=int)
        for i, state_val in enumerate(vpath_subject):
            if isinstance(state_val, np.ndarray) and state_val.size > 1:
                vpath_scalar[i] = np.argmax(state_val)
            else:
                vpath_scalar[i] = state_val
    else:
        vpath_scalar = None
    
    # Network heatmap
    ax1 = plt.subplot(gs[0])
    
    # Create the heatmap
    im = ax1.imshow(
        network_z.T,  # Transpose to have networks on y-axis
        aspect='auto',
        cmap='coolwarm',
        interpolation='none',
        vmin=-2, vmax=2  # Limit z-scores for better contrast
    )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Z-score')
    
    # Add state boundaries as vertical lines
    if vpath_scalar is not None:
        prev_state = vpath_scalar[0]
        for t in range(1, len(vpath_scalar)):
            if vpath_scalar[t] != prev_state:
                ax1.axvline(x=t, color='black', linestyle='-', linewidth=1, alpha=0.5)
                prev_state = vpath_scalar[t]
    
    # Label axes
    ax1.set_ylabel('Network')
    ax1.set_yticks(range(len(model_data['network_fields'])))
    ax1.set_yticklabels([f"{name}" for name in model_data['network_fields']])
    ax1.set_title(f'{group.capitalize()} - Subject {subject_idx+1} Network Activity (k={k})')
    
    # Remove x-axis ticks for upper plot
    ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    
    # State assignment plot
    ax2 = plt.subplot(gs[1], sharex=ax1)
    
    if vpath_scalar is not None:
        # Create colored regions for state assignments
        prev_state = -1
        start_pos = 0
        
        for t, state in enumerate(vpath_scalar):
            if state != prev_state or t == len(vpath_scalar) - 1:
                if prev_state >= 0:
                    # Add colored rectangle
                    rect = patches.Rectangle(
                        (start_pos, 0), t - start_pos, 1, 
                        color=state_colors[prev_state % len(state_colors)],
                        alpha=0.7
                    )
                    ax2.add_patch(rect)
                    
                    # Add state label if segment is long enough
                    if t - start_pos > max_timepoints / 20:
                        ax2.text(
                            start_pos + (t - start_pos) / 2, 0.5, 
                            f"State {prev_state + 1}", 
                            ha='center', va='center',
                            color='white', fontweight='bold'
                        )
                
                start_pos = t
                prev_state = state
    
    # Configure state assignment plot
    ax2.set_xlim(0, len(network_data))
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('State')
    ax2.set_xlabel('Time (TR)')
    ax2.set_yticks([])  # Remove y ticks for state plot
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f'{group}_k{k}_subject{subject_idx+1}_network_heatmap.png')
    save_figure(fig, output_path)
    plt.close(fig)
    
def main():
    """Main function to run the Yeo network visualization pipeline."""
    print("=== YEO NETWORK VISUALIZATION ===")
    start_time = time.time()
    
    # Define range of k values to visualize (k=3 to k=5)
    k_values = range(3, 6)
    
    try:
        # Process both groups
        for group in ['controls', 'meditators']:
            print(f"\nProcessing {group} group models...")
            
            # Create group directory
            group_dir = os.path.join(VIS_DIR, group)
            os.makedirs(group_dir, exist_ok=True)
            
            for k in k_values:
                print(f"\nGenerating visualizations for k={k}...")
                
                # Load model for this k value
                model_data = load_trained_model(group, k)
                
                if model_data is None:
                    print(f"Warning: Could not load {group} model with k={k}")
                    continue
                
                # Generate network correlation matrices for each state
                print("Generating network correlation matrices...")
                for state_idx in range(k):
                    visualize_network_correlations(model_data, group, k, state_idx)
                
                # Generate network time series visualizations for first 2 subjects
                print("Generating network time series visualizations...")
                for subject_idx in range(min(2, len(model_data['indices']))):                    
                    # Create heatmap with state overlay
                    visualize_network_heatmap_with_states(model_data, group, k, subject_idx)
        
        # Report completion
        elapsed_time = time.time() - start_time
        print(f"\n=== Visualization completed in {elapsed_time/60:.2f} minutes ===")
        print(f"Visualizations saved to: {VIS_DIR}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()