"""
Script to verify that state means are correctly calculated and consistent
between the training approach and the metrics calculation approach.
"""

import os
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
import logging
# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the path to ensure imports work
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Local imports
from analysis.tde_hmm_metrics import calculate_state_means

# Setup paths
DATA_DIR = os.path.join(project_root, 'data')
TRAINED_DIR = os.path.join(DATA_DIR, 'trained')



def load_model(group='meditators', k=5, networks=8):
    """Load a trained model."""
    model_path = os.path.join(TRAINED_DIR, f'selected_model_{networks}networks', 
                              group, f'k{k}', 'model.pkl')
    
    print(f"Loading model from: {model_path}")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # Add k to model_data if it doesn't exist
    if 'k' not in model_data:
        print(f"Adding missing 'k' value ({k}) to model_data")
        model_data['k'] = k
    
    # Verify model structure
    print(f"Model keys: {list(model_data.keys())}")
    if 'state_means' in model_data:
        print(f"State means available with shape: {len(model_data['state_means'])} states")
    
    return model_data

def compare_means(model_data):
    """Compare different approaches to calculate state means."""
    # Get k from model_data or try to infer it
    if 'k' in model_data:
        k = model_data['k']
    elif 'active_states' in model_data:
        k = model_data['active_states']
        print(f"Using active_states ({k}) as substitute for k")
    elif 'state_means' in model_data:
        k = len(model_data['state_means'])
        print(f"Inferring k ({k}) from length of state_means")
    else:
        k = 5  # Default fallback
        print(f"Using default k value: {k}")
    
    network_fields = model_data['network_fields']
    
    # Get the precomputed means from training
    if 'state_means' not in model_data or model_data['state_means'] is None:
        print("No precomputed state means found in model_data!")
        return
    
    precomputed_means = model_data['state_means']
    print(f"Precomputed means shape: {len(precomputed_means)} states, {len(precomputed_means[0])} networks")
    
    # Use our metrics function to calculate means
    calculated_means = calculate_state_means(model_data, network_fields)
    if calculated_means is None:
        print("Error: calculate_state_means returned None!")
        return
        
    print(f"Calculated means shape: {len(calculated_means)} states, {len(calculated_means[0])} networks")
    
    # Compare the two approaches
    print("\nComparing means:")
    for state in range(k):
        original = precomputed_means[state]
        calculated = calculated_means[state]
        
        # Calculate correlation to see if patterns match
        correlation = np.corrcoef(original, calculated)[0, 1]
        
        # Calculate absolute difference
        mean_diff = np.mean(np.abs(original - calculated))
        
        print(f"State {state+1}:")
        print(f"  Correlation: {correlation:.4f}")
        print(f"  Mean absolute difference: {mean_diff:.4f}")
        
    # Visualize the comparison
    visualize_comparison(precomputed_means, calculated_means, network_fields)
    
def visualize_comparison(precomputed_means, calculated_means, network_fields):
    """Visualize the comparison between the two approaches."""
    k = len(precomputed_means)
    
    fig, axes = plt.subplots(k, 1, figsize=(10, 3*k))
    if k == 1:
        axes = [axes]
    
    x = np.arange(len(network_fields))
    width = 0.35
    
    for state in range(k):
        ax = axes[state]
        
        ax.bar(x - width/2, precomputed_means[state], width, label='Training method')
        ax.bar(x + width/2, calculated_means[state], width, label='Metrics method')
        
        ax.set_title(f'State {state+1} Mean Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(network_fields)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('state_means_comparison.png')
    plt.show()

def main():
    """Main function."""
    print("=== Verifying State Means Calculation ===")
    
    # Load a model
    model_data = load_model()
    
    # Compare the means
    compare_means(model_data)

if __name__ == "__main__":
    main()