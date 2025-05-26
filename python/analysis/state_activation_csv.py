import os
import numpy as np
import pandas as pd
import pickle
import logging
import matplotlib.pyplot as plt

# Setup paths
ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')
OUTPUT_DIR = os.path.join(RESULTS_DIR, 'state_patterns')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Constants
K_VALUES = [4, 5]
STANDARDIZATION_METHODS = ['global', 'bygroup']
GROUPS = ['meditators', 'controls']
NETWORK_CONFIGS = [7, 8]

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_normalized_activations(state_means):
    """Normalize state activations using z-score within each state."""
    global_mean = np.mean(state_means)
    global_std = np.std(state_means)
    return (state_means - global_mean) / global_std

def load_metrics(group, networks, k, std_method):
    """Load metrics for a specific configuration."""
    metrics_path = os.path.join(METRICS_DIR, std_method, f'{networks}networks', 
                               group, f'k{k}_metrics.pkl')
    
    if not os.path.exists(metrics_path):
        logger.warning(f"Metrics file not found: {metrics_path}")
        return None
    
    try:
        with open(metrics_path, 'rb') as f:
            metrics_data = pickle.load(f)
        logger.info(f"Loaded metrics for {group}, {networks}-network, k={k}, {std_method} standardization")
        return metrics_data
    except Exception as e:
        logger.error(f"Error loading metrics: {str(e)}")
        return None

def load_correspondence(k, group, networks):
    """Load standardization correspondence data."""
    ref_dir = os.path.join(RESULTS_DIR, 'standardization_reference', f'k{k}')
    corr_path = os.path.join(ref_dir, f'{group}_{networks}networks_state_correspondence.pkl')
    
    if not os.path.exists(corr_path):
        logger.warning(f"Correspondence file not found: {corr_path}")
        return None
    
    try:
        with open(corr_path, 'rb') as f:
            corr_data = pickle.load(f)
        logger.info(f"Loaded correspondence for k={k}, {group}, {networks}-networks")
        return corr_data
    except Exception as e:
        logger.error(f"Error loading correspondence: {str(e)}")
        return None

def load_med_con_correspondence(k, networks):
    """Load meditator-control correspondence data."""
    ref_dir = os.path.join(RESULTS_DIR, 'standardization_reference', f'k{k}')
    
    # Try bygroup correspondence first (preferred)
    corr_path = os.path.join(ref_dir, f'bygroup_med_control_{networks}networks_correspondence.pkl')
    
    if not os.path.exists(corr_path):
        # Try global correspondence as fallback
        corr_path = os.path.join(ref_dir, f'meditator_control_{networks}networks_correspondence.pkl')
    
    if not os.path.exists(corr_path):
        logger.warning(f"Med-con correspondence file not found for k={k}, {networks}-networks")
        return None
    
    try:
        with open(corr_path, 'rb') as f:
            corr_data = pickle.load(f)
        logger.info(f"Loaded med-con correspondence for k={k}, {networks}-networks")
        return corr_data
    except Exception as e:
        logger.error(f"Error loading med-con correspondence: {str(e)}")
        return None

def compute_k4_k5_correspondence(group, networks, std_method):
    """Compute correspondence between k=4 and k=5 states."""
    # Load metrics for k=4 and k=5
    metrics_k4 = load_metrics(group, networks, 4, std_method)
    metrics_k5 = load_metrics(group, networks, 5, std_method)
    
    if metrics_k4 is None or metrics_k5 is None:
        return None
    
    # Extract state means
    try:
        means_k4 = np.array(metrics_k4['temporal_metrics']['state_means'])
        means_k5 = np.array(metrics_k5['temporal_metrics']['state_means'])
    except KeyError:
        logger.error(f"Could not extract state means for {group}, {networks}, {std_method}")
        return None
    
    # Calculate similarity matrix between k=4 and k=5 states
    similarity_matrix = np.zeros((4, 5))
    
    for i in range(4):
        for j in range(5):
            # Handle edge cases
            if len(means_k4[i]) != len(means_k5[j]):
                similarity_matrix[i, j] = 0
                continue
                
            # Normalize state patterns
            norm_state1 = get_normalized_activations(means_k4[i])
            norm_state2 = get_normalized_activations(means_k5[j])
            
            # Use correlation as similarity measure
            correlation = np.corrcoef(norm_state1, norm_state2)[0, 1]
            
            # Handle NaN values
            if np.isnan(correlation):
                similarity_matrix[i, j] = 0
            else:
                # Convert to [0,1] range for similarity (1 = identical)
                similarity_matrix[i, j] = (correlation + 1) / 2
    
    # Find best match for each k4 state
    k4_to_k5 = {}
    threshold = 0.6  # Minimum overlap required for correspondence
    
    for i in range(4):
        best_match = np.argmax(similarity_matrix[i])
        best_score = similarity_matrix[i, best_match]
        
        if best_score >= threshold:
            k4_to_k5[i] = {
                'k5_state': best_match,
                'overlap': best_score
            }
    
    return k4_to_k5

def generate_consolidated_csv():
    """Generate a consolidated CSV with all state activations and correspondences."""
    all_rows = []
    
    # Process each configuration
    for networks in NETWORK_CONFIGS:
        for k in K_VALUES:
            for group in GROUPS:
                for std_method in STANDARDIZATION_METHODS:
                    # Load metrics
                    metrics = load_metrics(group, networks, k, std_method)
                    if metrics is None:
                        continue
                    
                    # Load standardization correspondence
                    std_corr = load_correspondence(k, group, networks)
                    
                    # Load meditator-control correspondence
                    med_con_corr = load_med_con_correspondence(k, networks)
                    
                    # Load or compute k4-k5 correspondence
                    if k == 4:
                        k4_k5_corr = compute_k4_k5_correspondence(group, networks, std_method)
                    
                    # Extract state patterns
                    try:
                        state_means = metrics['temporal_metrics']['state_means']
                        network_fields = metrics['temporal_metrics']['network_fields']
                        
                        # Extract FO if available
                        fo_mean = None
                        if 'FO_mean' in metrics['temporal_metrics']:
                            fo_mean = metrics['temporal_metrics']['FO_mean']
                    except KeyError:
                        logger.error(f"Could not extract data for {group}, {networks}, k={k}, {std_method}")
                        continue
                    
                    # Process each state
                    for state_idx in range(len(state_means)):
                        # Create base row with metadata
                        row = {
                            'group': group,
                            'standardization': std_method,
                            'networks': networks,
                            'k': k,
                            'state_idx': state_idx
                        }
                        
                        # Add network activations
                        for i, network in enumerate(network_fields):
                            if i < len(state_means[state_idx]):
                                row[network] = state_means[state_idx][i]
                        
                        # Add fractional occupancy
                        if fo_mean is not None and state_idx < len(fo_mean):
                            row['fractional_occupancy'] = fo_mean[state_idx]
                        
                        # Add correspondence with other standardization
                        if std_corr is not None and 'global_to_bygroup' in std_corr:
                            if std_method == 'global' and state_idx in std_corr['global_to_bygroup']:
                                bygroup_state = std_corr['global_to_bygroup'][state_idx]
                                row['corresponding_state_bygroup'] = bygroup_state
                                
                                # Get overlap score if available
                                if 'global_bygroup_overlap' in std_corr:
                                    row['bygroup_overlap'] = std_corr['global_bygroup_overlap'][state_idx, bygroup_state]
                            
                            elif std_method == 'bygroup':
                                # Find the reverse mapping
                                for global_state, bygroup_state in std_corr['global_to_bygroup'].items():
                                    if bygroup_state == state_idx:
                                        row['corresponding_state_global'] = global_state
                                        
                                        # Get overlap score if available
                                        if 'global_bygroup_overlap' in std_corr:
                                            row['global_overlap'] = std_corr['global_bygroup_overlap'][global_state, state_idx]
                                        break
                        
                        # For meditators, add correspondence with controls
                        if group == 'meditators' and med_con_corr is not None and 'med_to_con' in med_con_corr:
                            if state_idx in med_con_corr['med_to_con']:
                                control_state = med_con_corr['med_to_con'][state_idx]
                                row['corresponding_state_control'] = control_state
                                
                                # Get overlap score if available
                                if 'med_con_overlap' in med_con_corr:
                                    row['control_overlap'] = med_con_corr['med_con_overlap'][state_idx, control_state]
                        
                        # For k=4, add correspondence with k=5 model
                        if k == 4 and k4_k5_corr is not None and state_idx in k4_k5_corr:
                            row['corresponding_state_k5'] = k4_k5_corr[state_idx]['k5_state']
                            row['k5_overlap'] = k4_k5_corr[state_idx]['overlap']
                        
                        # Add row to our collection
                        all_rows.append(row)
    
    # Convert to DataFrame
    if all_rows:
        df = pd.DataFrame(all_rows)
        
        # Save to CSV
        output_path = os.path.join(OUTPUT_DIR, 'all_states_activation_patterns.csv')
        df.to_csv(output_path, index=False)
        logger.info(f"Saved consolidated state patterns to {output_path}")
        
        # Display summary
        print(f"\nGenerated consolidated CSV with {len(df)} state patterns")
        print(f"File saved to: {output_path}")
        
        # Return the dataframe for potential further processing
        return df
    else:
        logger.warning("No data was found to create the consolidated CSV")
        return None

if __name__ == "__main__":
    logger.info("Starting generation of consolidated state activation patterns CSV")
    generate_consolidated_csv()