"""
Improved Time-Delay Embedding (TDE) Visualization

This script creates improved visualizations of the TDE processing steps
with accurate color scaling for each level of the hierarchical plot.

Usage:
  python tde_visualization_improved.py --networks [7|8] --standardize [global|bygroup|persequence]
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup paths
ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
PROCESSED_DIR = os.path.join(ROOT_DIR, 'data', 'processed')
TDE_DIR = os.path.join(PROCESSED_DIR, 'tde')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results', 'visualizations', 'tde_improved')

# Create necessary directories
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_processed_tde_data(networks=7, standardize='bygroup'):
    """Load processed TDE data from pickle file"""
    filename = f'tde_{networks}networks_{standardize}.pkl'
    filepath = os.path.join(TDE_DIR, filename)
    
    if not os.path.exists(filepath):
        logger.error(f"TDE data file not found: {filepath}")
        raise FileNotFoundError(f"File not found: {filepath}")
    
    logger.info(f"Loading TDE data from: {filepath}")
    with open(filepath, 'rb') as f:
        processed_data = pickle.load(f)
    
    return processed_data

def load_original_data(networks=7, standardize='bygroup'):
    """Load original eigenvector and standardized data for visualization"""
    # We need to reconstruct paths to original eigenvector data
    # This would likely involve loading the original mat file
    # For simplicity, we'll use a visualization-specific data file
    
    try:
        # Try loading pre-saved visualization data if available
        viz_data_path = os.path.join(TDE_DIR, f'viz_data_{networks}networks_{standardize}.pkl')
        if os.path.exists(viz_data_path):
            with open(viz_data_path, 'rb') as f:
                return pickle.load(f)
        
        # If not available, load from the TDE processing outputs
        controls_tde_path = os.path.join(TDE_DIR, f'controls_{networks}networks_{standardize}.pkl')
        meditators_tde_path = os.path.join(TDE_DIR, f'meditators_{networks}networks_{standardize}.pkl')
        
        if os.path.exists(controls_tde_path) and os.path.exists(meditators_tde_path):
            with open(controls_tde_path, 'rb') as f:
                controls_data = pickle.load(f)
            with open(meditators_tde_path, 'rb') as f:
                meditators_data = pickle.load(f)
            
            return {
                'controls': controls_data,
                'meditators': meditators_data
            }
    except Exception as e:
        logger.warning(f"Could not load original data for improved visualization: {str(e)}")
        logger.warning("Will visualize only embedded data without original and standardized stages.")
    
    return None

def visualize_embedded_data_improved(processed_data, original_data=None, group='controls', subject_idx=0):
    """Create improved visualization of TDE data with proper color scaling"""
    logger.info(f"Creating improved visualization for {group}, subject {subject_idx+1}")
    
    # Get the embedded sequence data
    embedded_seq = processed_data[f'{group}_sequences'][subject_idx]
    
    # Create figure
    plt.figure(figsize=(15, 12))
    
    # Determine how many plots to show based on available data
    n_plots = 3 if original_data else 1
    plot_idx = 1
    
    if original_data and group in original_data:
        # Get original and standardized data if available
        orig_seq = original_data[group]['X'][subject_idx]
        preproc_seq = original_data[group]['X_preproc'][subject_idx]
        
        # Plot original data with proper symmetric scaling
        plt.subplot(n_plots, 1, plot_idx)
        plot_idx += 1
        orig_max = np.max(np.abs(orig_seq))
        plt.imshow(orig_seq.T, aspect='auto', cmap='RdBu_r', 
                  vmin=-orig_max, vmax=orig_max)  # Symmetric scaling
        plt.title(f'Original Leading Eigenvectors ({group.capitalize()}, Subject {subject_idx+1})')
        plt.xlabel('Time (TRs)')
        plt.ylabel('Network')
        if 'network_fields' in processed_data:
            plt.yticks(range(len(processed_data['network_fields'])), processed_data['network_fields'])
        plt.colorbar(label='Eigenvector Value (-1 to +1)')
        
        # Plot standardized data with standard normal scale
        plt.subplot(n_plots, 1, plot_idx)
        plot_idx += 1
        plt.imshow(preproc_seq.T, aspect='auto', cmap='RdBu_r',
                  vmin=-3, vmax=3)  # Standard range for z-scores (-3σ to +3σ)
        plt.title(f'Standardized Data ({processed_data["standardization_method"]} normalization)')
        plt.xlabel('Time (TRs)')
        plt.ylabel('Network')
        if 'network_fields' in processed_data:
            plt.yticks(range(len(processed_data['network_fields'])), processed_data['network_fields'])
        plt.colorbar(label='Z-score (μ=0, σ=1)')
    
    # Plot embedded data with appropriate scaling
    plt.subplot(n_plots, 1, plot_idx)
    emb_max = max(abs(np.max(embedded_seq)), abs(np.min(embedded_seq)))
    plt.imshow(embedded_seq.T, aspect='auto', cmap='RdBu_r',
              vmin=-emb_max, vmax=emb_max)  # Symmetric but data-driven scale
    
    plt.title(f'TDE-Embedded Data with PCA (21 TRs → {embedded_seq.shape[1]} components)')
    plt.xlabel('Time (TRs)')
    plt.ylabel('PCA Component')
    cbar = plt.colorbar(label='PCA Component Value')
    
    # Add annotation about timepoint representation
    plt.figtext(0.5, 0.01, 
               "Note: Each timepoint represents the middle TR of a 21-TR window (±10 TRs)",
               ha='center', fontsize=10, bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save figure
    output_path = os.path.join(RESULTS_DIR, f'{group}_subject_{subject_idx+1}_tde_improved.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    logger.info(f"Improved visualization saved to: {output_path}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Improved TDE Visualization')
    
    parser.add_argument('--standardize', type=str, choices=['global', 'bygroup', 'persequence'],
                        default='bygroup', help='Standardization method (default: bygroup)')
    
    parser.add_argument('--networks', type=int, choices=[7, 8], default=7,
                        help='Number of networks used (default: 7)')
    
    return parser.parse_args()

def main():
    """Main function to create improved TDE visualizations"""
    
    # Parse arguments
    args = parse_arguments()
    
    logger.info("=== Creating Improved TDE Visualizations ===")
    logger.info(f"Configuration: {args.networks} networks, {args.standardize} standardization")
    
    try:
        # Load processed TDE data
        processed_data = load_processed_tde_data(args.networks, args.standardize)
        
        # Try to load original data if available
        original_data = load_original_data(args.networks, args.standardize)
        
        # Create visualizations for a subset of subjects
        for group in ['controls', 'meditators']:
            n_subjects = len(processed_data[f'{group}_sequences'])
            
            # Select a subset of subjects to visualize (adjust as needed)
            subject_indices = [0, min(5, n_subjects-1), min(10, n_subjects-1)]
            
            for subj_idx in subject_indices:
                if subj_idx < n_subjects:
                    visualize_embedded_data_improved(processed_data, original_data, group, subj_idx)
                else:
                    logger.warning(f"Subject index {subj_idx} exceeds available {group} subjects")
        
        logger.info(f"=== Improved TDE visualizations completed ===")
        logger.info(f"Results saved to: {RESULTS_DIR}")
        
    except Exception as e:
        logger.error(f"Error creating TDE visualizations: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()