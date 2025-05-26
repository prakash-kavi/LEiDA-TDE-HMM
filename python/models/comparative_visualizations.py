"""
TDE-HMM Model Visualization Script

This script creates standardized visualizations for TDE-HMM models
trained with different numbers of states (k=3 to k=5) for meditation
and control groups.

It generates visualizations for:
1. Fractional occupancy
2. Switching rates (with adjusted y-axis)
3. State lifetimes (with adjusted y-axis)
4. State time courses
5. Network profiles
6. Transition matrices
7. Functional connectivity (correlation matrices)

Visualizations are saved in a structured directory format for easy comparison.
"""

import os
import numpy as np
import matplotlib
# Force Agg backend for reliable file saving
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import time
from datetime import datetime
import seaborn as sns

# Import GLHMM modules
from glhmm import graphics, auxiliary

# Setup paths
ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
TRAINED_DIR = os.path.join(DATA_DIR, 'trained', 'glhmm_tde')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
VIS_DIR = os.path.join(RESULTS_DIR, 'visualizations')

# Create visualization directories
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(os.path.join(VIS_DIR, 'controls'), exist_ok=True)
os.makedirs(os.path.join(VIS_DIR, 'meditators'), exist_ok=True)
os.makedirs(os.path.join(VIS_DIR, 'comparison'), exist_ok=True)

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

def cov_to_corr(cov_matrix):
    """Convert covariance matrix to correlation matrix."""
    d = np.sqrt(np.diag(cov_matrix))
    corr_matrix = cov_matrix / np.outer(d, d)
    # Ensure values are in valid range [-1, 1]
    corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
    return corr_matrix

def save_figure(fig, filepath, dpi=200):
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

def create_model_visualizations(model_data, group, k):
    """Create all standard visualizations for a model."""
    print(f"Creating visualizations for {group} model with k={k}...")
    
    # Create group-specific output directory
    output_dir = os.path.join(VIS_DIR, group, f'k{k}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract key variables from model data
    hmm = model_data['hmm']
    FO = model_data['FO']
    SR = model_data['SR']
    LTmean = model_data['LTmean']
    LTmed = model_data['LTmed']
    vpath = model_data['vpath']
    indices = model_data['indices']
    state_means = model_data['state_means']
    network_fields = model_data['network_fields']
    X_preproc = model_data['X_preproc']
    lags = model_data['lags']
    P = model_data['P']
    
    # 1. Plot fractional occupancy
    plt.figure(figsize=(10, 6))
    graphics.plot_FO(FO)
    plt.title(f'{group.capitalize()} - Fractional Occupancy (k={k})')
    plt.tight_layout()
    save_figure(plt.gcf(), os.path.join(output_dir, f'{group}_k{k}_FO.png'))
    plt.close()
    
    # 2. Plot switching rates with adjusted y-axis
    plt.figure(figsize=(10, 6))
    graphics.plot_switching_rates(SR)
    plt.title(f'{group.capitalize()} - Switching Rates (k={k})')
    plt.ylim(0, 0.10)  # Adjusted for fMRI data
    plt.tight_layout()
    save_figure(plt.gcf(), os.path.join(output_dir, f'{group}_k{k}_switching_rates.png'))
    plt.close()
    
    # 3. Plot state lifetimes with adjusted y-axis
    plt.figure(figsize=(10, 6))
    graphics.plot_state_lifetimes(LTmed)
    plt.title(f'{group.capitalize()} - State Lifetimes (Median, k={k})')
    plt.ylim(0, 50)  # Adjusted for typical fMRI state lifetimes
    plt.tight_layout()
    save_figure(plt.gcf(), os.path.join(output_dir, f'{group}_k{k}_lifetimes.png'))
    plt.close()
    
    # 4. Plot state time courses with signal (for first subject)
    # Get first subject data
    if len(indices) > 0:
        start_idx, end_idx = indices[0]
        subj_length = end_idx - start_idx
        plot_length = min(1000, subj_length)  # Plot up to 1000 time points
        
        # Apply padGamma to match original time dimensions
        T = auxiliary.get_T(indices)
        options_tde = {'embeddedlags': list(lags)}
        try:
            paddedVP = auxiliary.padGamma(vpath, T, options_tde)
            
            # Plot state time courses with signal
            plt.figure(figsize=(15, 8))
            graphics.plot_vpath(
                paddedVP[start_idx:start_idx+plot_length], 
                signal=X_preproc[start_idx:start_idx+plot_length, 0].copy(),
                title=f"{group.capitalize()} - States and signal example (k={k})"
            )
            plt.tight_layout()
            save_figure(plt.gcf(), os.path.join(output_dir, f'{group}_k{k}_state_timecourse.png'))
            plt.close()
        except Exception as e:
            print(f"Warning: Could not create state time course plot - {str(e)}")
    
    # 5. Plot state network profiles
    fig = plt.figure(figsize=(12, 10))
    
    for state_idx in range(k):
        # Ensure we have data for this state (check if state_idx is within range)
        if state_idx < len(state_means):
            # Calculate z-scores for better visualization
            network_mean = state_means[state_idx]
            
            # Check if dimensions match - if not, use only the first n values
            if len(network_mean) != len(network_fields):
                print(f"Warning: Dimension mismatch - Network mean has {len(network_mean)} elements while network_fields has {len(network_fields)}.")
                # Use only the corresponding network dimensions
                network_mean = network_mean[:len(network_fields)]
            
            z_score = (network_mean - np.mean(network_mean)) / (np.std(network_mean) + 1e-10)
            
            ax = fig.add_subplot(k, 1, state_idx + 1)
            ax.bar(range(len(network_fields)), z_score, color='steelblue')
            ax.axhline(0, color='k', linestyle='-', alpha=0.3)
            ax.axhline(1, color='r', linestyle='--', alpha=0.3)
            ax.axhline(-1, color='r', linestyle='--', alpha=0.3)
            ax.set_xticks(range(len(network_fields)))
            ax.set_xticklabels(network_fields, rotation=45)
            ax.set_title(f'State {state_idx + 1}')
            ax.set_ylabel('Z-score')
            ax.grid(alpha=0.3)
        else:
            # Handle the case where state data is missing
            ax = fig.add_subplot(k, 1, state_idx + 1)
            ax.text(0.5, 0.5, f"State {state_idx + 1}: No data available", 
                    ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
    
    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, f'{group}_k{k}_state_profiles.png'))
    plt.close(fig)
    
    # 6. Plot transition matrix
    fig = plt.figure(figsize=(8, 6))
    plt.imshow(P, cmap='Blues', interpolation='none')
    plt.colorbar(label='Probability')
    plt.title(f'{group.capitalize()} - State Transition Matrix (k={k})')
    
    # Add text annotations
    for i in range(k):
        for j in range(k):
            plt.text(j, i, f'{P[i,j]:.2f}', ha='center', va='center', 
                    color='white' if P[i,j] > 0.5 else 'black')
    
    plt.xlabel('To State')
    plt.ylabel('From State')
    plt.xticks(range(k), [f'State {i+1}' for i in range(k)])
    plt.yticks(range(k), [f'State {i+1}' for i in range(k)])
    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, f'{group}_k{k}_transition_matrix.png'))
    plt.close(fig)
    
    # 7. Plot functional connectivity (correlation matrices) for each state
    try:
        for state in range(k):
            # Get covariance matrix and convert to correlation
            cov = hmm.get_covariance_matrix(state)
            corr = cov_to_corr(cov)
            
            # Plot correlation matrix
            fig = plt.figure(figsize=(10, 8))
            sns.heatmap(corr, cmap='coolwarm', vmin=-1, vmax=1, 
                      square=True, annot=False, center=0)
            plt.title(f'{group.capitalize()} - State {state+1} Correlation Matrix (k={k})')
            plt.tight_layout()
            save_figure(fig, os.path.join(output_dir, f'{group}_k{k}_state{state+1}_correlation.png'))
            plt.close(fig)
    except Exception as e:
        print(f"Warning: Could not create correlation matrix plots - {str(e)}")

def create_comparative_visualizations(k_values, groups=['controls', 'meditators']):
    """Create visualizations that compare metrics across k values and groups."""
    print("Creating comparative visualizations...")
    
    # Create output directory
    output_dir = os.path.join(VIS_DIR, 'comparison')
    os.makedirs(output_dir, exist_ok=True)
    
    # Data structures to store metrics across models
    metrics = {group: {k: {} for k in k_values} for group in groups}
    
    # Load metrics from all models
    for group in groups:
        for k in k_values:
            model_data = load_trained_model(group, k)
            if model_data is not None:
                metrics[group][k]['FO_mean'] = np.mean(model_data['FO'], axis=0)
                metrics[group][k]['SR_mean'] = np.mean(model_data['SR'])
                metrics[group][k]['LT_mean'] = np.mean(model_data['LTmean'], axis=0)
                metrics[group][k]['free_energy'] = model_data['free_energy']
                metrics[group][k]['active_states'] = model_data['active_states']
    
    # 1. Plot mean switching rates across k values
    fig = plt.figure(figsize=(10, 6))
    for group in groups:
        k_list = []
        sr_list = []
        for k in k_values:
            if k in metrics[group] and 'SR_mean' in metrics[group][k]:
                k_list.append(k)
                sr_list.append(metrics[group][k]['SR_mean'])
        
        if k_list:
            plt.plot(k_list, sr_list, 'o-', label=group.capitalize())
    
    plt.title('Mean Switching Rate by Number of States')
    plt.xlabel('Number of States (k)')
    plt.ylabel('Mean Switching Rate')
    plt.ylim(0, 0.10)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, 'switching_rate_by_k.png'))
    plt.close(fig)
    
    # 2. Plot average state lifetimes across k values
    fig = plt.figure(figsize=(10, 6))
    
    for group in groups:
        k_list = []
        lt_list = []
        for k in k_values:
            if k in metrics[group] and 'LT_mean' in metrics[group][k]:
                k_list.append(k)
                lt_list.append(np.mean(metrics[group][k]['LT_mean']))
        
        if k_list:
            plt.plot(k_list, lt_list, 'o-', label=group.capitalize())
    
    plt.title('Mean State Lifetime by Number of States')
    plt.xlabel('Number of States (k)')
    plt.ylabel('Mean Lifetime (TRs)')
    plt.ylim(0, 50)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, 'lifetime_by_k.png'))
    plt.close(fig)
    
    # 3. Plot free energy by k
    fig = plt.figure(figsize=(10, 6))
    
    for group in groups:
        k_list = []
        fe_list = []
        for k in k_values:
            if k in metrics[group] and 'free_energy' in metrics[group][k]:
                k_list.append(k)
                fe_list.append(metrics[group][k]['free_energy'])
        
        if k_list:
            plt.plot(k_list, fe_list, 'o-', label=group.capitalize())
    
    plt.title('Free Energy by Number of States')
    plt.xlabel('Number of States (k)')
    plt.ylabel('Free Energy')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, 'free_energy_by_k.png'))
    plt.close(fig)
    
    # 4. Plot active states ratio by k
    fig = plt.figure(figsize=(10, 6))
    
    for group in groups:
        k_list = []
        active_ratio_list = []
        for k in k_values:
            if k in metrics[group] and 'active_states' in metrics[group][k]:
                k_list.append(k)
                active_ratio = metrics[group][k]['active_states'] / k
                active_ratio_list.append(active_ratio)
        
        if k_list:
            plt.plot(k_list, active_ratio_list, 'o-', label=group.capitalize())
    
    plt.title('Active States Ratio by Number of States')
    plt.xlabel('Number of States (k)')
    plt.ylabel('Active States Ratio')
    plt.ylim(0, 1.1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, 'active_states_ratio_by_k.png'))
    plt.close(fig)

def main():
    """Main function to run the TDE-HMM visualization pipeline."""
    print("=== TDE-HMM MODEL VISUALIZATION ===")
    start_time = time.time()
    
    # Define range of k values to visualize (LIMITED TO k=3 to k=5)
    k_values = range(3, 6)  # k=3 to k=5
    
    try:
        # Create individual model visualizations
        for group in ['controls', 'meditators']:
            print(f"\nProcessing {group} group models...")
            
            # Create group directory
            group_dir = os.path.join(VIS_DIR, group)
            os.makedirs(group_dir, exist_ok=True)
            
            for k in k_values:
                # Load model for this k value
                model_data = load_trained_model(group, k)
                
                if model_data is not None:
                    # Create visualizations for this model
                    create_model_visualizations(model_data, group, k)
                else:
                    print(f"Warning: Could not load {group} model with k={k}")
        
        # Create comparative visualizations
        create_comparative_visualizations(k_values)
        
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