"""
Standardization Reference Framework

This script establishes objective reference benchmarks for comparing two 
standardization approaches (global, bygroup) using k=5 models.
It focuses on creating statistical alignments between states from different 
standardization methods and measuring standardization deviations.

The script:
1. Loads metrics from both standardization methods
2. Aligns corresponding states between standardization approaches
3. Quantifies how metrics change across standardization approaches
4. Generates focused reference visualizations and benchmark metrics
"""

import os
import numpy as np
import pandas as pd
import pickle
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy.stats import zscore
from scipy.optimize import linear_sum_assignment

# Import GLHMM modules
from glhmm import utils

# Setup paths
ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')
REFERENCE_DIR = os.path.join(RESULTS_DIR, 'standardization_reference', 'k5')
VIS_DIR = os.path.join(REFERENCE_DIR, 'visualizations')

# Create directories if they don't exist
os.makedirs(REFERENCE_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

K = 5  # Fixed k=5
STANDARDIZATION_METHODS = ['global', 'bygroup']  # Focusing only on these two methods
GROUPS = ['meditators', 'controls']
NETWORK_CONFIGS = [7, 8]

# Key meditation-relevant metrics to focus on
KEY_METRICS = [
    'DMN_activation',           # Default mode network activation
    'SMN_activation',           # Sensorimotor network activation
    'DMN_anticorrelation',      # DMN anticorrelation with task networks
    'fractional_occupancy'      # State prevalence
]

def load_metrics_all_standardizations(group, networks, k=K):
    """ Load metrics from both standardization methods for comparison. """
    metrics_dict = {}
    
    for std_method in STANDARDIZATION_METHODS:
        metrics_path = os.path.join(METRICS_DIR, std_method, f'{networks}networks', 
                                   group, f'k{k}_metrics.pkl')
        
        if not os.path.exists(metrics_path):
            logger.warning(f"Metrics file not found: {metrics_path}")
            continue
        
        try:
            with open(metrics_path, 'rb') as f:
                metrics_data = pickle.load(f)
            logger.info(f"Loaded metrics for {group}, {networks}-network, k={k}, {std_method} standardization")
            metrics_dict[std_method] = metrics_data
        except Exception as e:
            logger.error(f"Error loading metrics for {std_method}: {str(e)}")
    
    return metrics_dict

def extract_state_features_for_alignment(metrics_data):
    """ Extract key features needed for state alignment. """
    features = {}
    
    # Extract state means - primary basis for alignment
    if 'temporal_metrics' in metrics_data and 'state_means' in metrics_data['temporal_metrics']:
        features['state_means'] = metrics_data['temporal_metrics']['state_means']
        features['network_fields'] = metrics_data['temporal_metrics']['network_fields']
        features['fo_mean'] = metrics_data['temporal_metrics']['FO_mean']
    
    return features

def get_normalized_activations(state_means):
    """
    Normalize state activations globally to enable cross-group comparisons.
    Returns both raw and normalized activations.
    """
    # Calculate global mean activation across all states and networks
    global_mean = np.mean(state_means)
    global_std = np.std(state_means)
    
    # Normalize state activations
    normalized_means = (state_means - global_mean) / global_std
    
    return normalized_means

def align_hmm_states(means1, means2, similarity_threshold=0.1):
    """Custom implementation to align HMM states between two models using normalized patterns."""
    if len(means1) == 0 or len(means2) == 0:
        return []
    
    # Calculate similarity matrix (correlation between normalized state patterns)
    similarity_matrix = np.zeros((len(means1), len(means2)))
    
    for i in range(len(means1)):
        for j in range(len(means2)):
            # Handle edge cases
            if len(means1[i]) != len(means2[j]):
                similarity_matrix[i, j] = 0
                continue
                
            # Normalize state patterns before correlation
            norm_state1 = get_normalized_activations(means1[i])
            norm_state2 = get_normalized_activations(means2[j])
            
            # Use correlation as similarity measure
            correlation = np.corrcoef(norm_state1, norm_state2)[0, 1]
            
            # Handle NaN values
            if np.isnan(correlation):
                similarity_matrix[i, j] = 0
            else:
                # Convert to a dissimilarity for the Hungarian algorithm
                similarity_matrix[i, j] = 1 - ((correlation + 1) / 2)
    
    # Use Hungarian algorithm to find optimal assignment
    # (minimizing dissimilarity)
    row_ind, col_ind = linear_sum_assignment(similarity_matrix)
    
    # Actual similarity (not dissimilarity)
    actual_similarity = 1 - similarity_matrix
    
    # Create alignment mapping with threshold
    alignment = [-1] * len(means1)
    for i, j in zip(row_ind, col_ind):
        if actual_similarity[i, j] >= similarity_threshold:
            alignment[i] = j
    
    return alignment

def state_overlap(state1, state2):
    """Calculate overlap (similarity) between two normalized state patterns."""
    if len(state1) != len(state2):
        return 0
    
    # Normalize state patterns before correlation
    norm_state1 = get_normalized_activations(state1)
    norm_state2 = get_normalized_activations(state2)
    
    # Use correlation as similarity measure
    correlation = np.corrcoef(norm_state1, norm_state2)[0, 1]
    
    # Handle NaN
    if np.isnan(correlation):
        return 0
        
    # Convert to [0, 1] range for "overlap" interpretation
    return (correlation + 1) / 2

def align_states_across_standardizations(metrics_by_std):
    """ Align states across standardization methods. """
    correspondences = {}
    
    # Extract features needed for alignment
    features_by_std = {}
    for std, metrics in metrics_by_std.items():
        features_by_std[std] = extract_state_features_for_alignment(metrics)
    
    # Align Global → ByGroup states using custom state alignment function
    if 'global' in features_by_std and 'bygroup' in features_by_std:
        # Use state means for alignment
        global_means = np.array(features_by_std['global']['state_means'])
        bygroup_means = np.array(features_by_std['bygroup']['state_means'])
        
        # Get correspondence between states using custom alignment
        global_to_bygroup = {}
        if global_means.shape[0] > 0 and bygroup_means.shape[0] > 0:
            # Custom align_hmm_states returns the permutation to match model1 to model2
            alignment = align_hmm_states(global_means, bygroup_means)
            for global_state in range(min(K, len(alignment))):
                bygroup_state = alignment[global_state]
                if bygroup_state != -1:  # Only add valid alignments
                    global_to_bygroup[global_state] = bygroup_state
            
            correspondences['global_to_bygroup'] = global_to_bygroup
            logger.info(f"Aligned Global → ByGroup states: {global_to_bygroup}")
    
    # Calculate state overlaps to measure alignment quality
    if 'global_to_bygroup' in correspondences:
        global_means = np.array(features_by_std['global']['state_means'])
        bygroup_means = np.array(features_by_std['bygroup']['state_means'])
        
        overlap_matrix = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                # Use custom state_overlap to quantify alignment quality
                if i < global_means.shape[0] and j < bygroup_means.shape[0]:
                    overlap_matrix[i, j] = state_overlap(global_means[i], bygroup_means[j])
        
        correspondences['global_bygroup_overlap'] = overlap_matrix
        
        # Log quality metrics on the alignment
        if global_to_bygroup:
            mean_overlap = np.mean([overlap_matrix[i, j] for i, j in global_to_bygroup.items()])
            logger.info(f"Mean state overlap: {mean_overlap:.3f}")
    
    return correspondences

def calculate_standardization_deviations(metrics_by_std, correspondences, group, networks):
    """ Calculate deviations in metrics between standardization methods."""
    deviations = {
        'group': group,
        'networks': networks,
        'k': K,
        'metrics_by_std': {},
        'global_to_bygroup': {}
    }
    
    # Ensure all required data exists
    if not all(std in metrics_by_std for std in STANDARDIZATION_METHODS):
        logger.warning(f"Missing metrics for some standardization methods")
        return deviations
    
    # Extract key metrics from each standardization method
    for std, metrics in metrics_by_std.items():
        deviations['metrics_by_std'][std] = {}
        
        # 1. Network activations (from state means)
        if 'temporal_metrics' in metrics and 'state_means' in metrics['temporal_metrics']:
            state_means = metrics['temporal_metrics']['state_means']
            network_fields = metrics['temporal_metrics']['network_fields']
            
            for state in range(min(K, len(state_means))):
                deviations['metrics_by_std'][std][f'state{state}_activations'] = state_means[state]
        
        # 2. FO
        if 'temporal_metrics' in metrics and 'FO_mean' in metrics['temporal_metrics']:
            fo_mean = metrics['temporal_metrics']['FO_mean']
            
            for state in range(min(K, len(fo_mean))):
                deviations['metrics_by_std'][std][f'state{state}_FO'] = fo_mean[state]
        
        # 3. Key network interaction metrics
        if 'network_interactions' in metrics:
            for state in range(K):
                if state in metrics['network_interactions']:
                    interactions = metrics['network_interactions'][state]
                    
                    for metric in KEY_METRICS:
                        if metric in interactions:
                            deviations['metrics_by_std'][std][f'state{state}_{metric}'] = interactions[metric]
    
    # Calculate deviations between global and bygroup
    if 'global_to_bygroup' in correspondences:
        for global_state, bygroup_state in correspondences['global_to_bygroup'].items():
            # Calculate deviations for each metric
            for metric in KEY_METRICS:
                global_key = f'state{global_state}_{metric}'
                bygroup_key = f'state{bygroup_state}_{metric}'
                
                if (global_key in deviations['metrics_by_std']['global'] and 
                    bygroup_key in deviations['metrics_by_std']['bygroup']):
                    
                    global_val = deviations['metrics_by_std']['global'][global_key]
                    bygroup_val = deviations['metrics_by_std']['bygroup'][bygroup_key]
                    
                    deviation = bygroup_val - global_val
                    
                    if metric not in deviations['global_to_bygroup']:
                        deviations['global_to_bygroup'][metric] = []
                    
                    deviations['global_to_bygroup'][metric].append({
                        'global_state': global_state,
                        'bygroup_state': bygroup_state,
                        'global_value': global_val,
                        'bygroup_value': bygroup_val,
                        'deviation': deviation,
                        'abs_deviation': abs(deviation)
                    })
    
    return deviations

def align_bygroup_states_between_groups(all_metrics, networks_list=NETWORK_CONFIGS):
    """Align states between meditator and control groups using bygroup standardization."""
    bygroup_correspondences = {7: {}, 8: {}}
    
    for networks in networks_list:
        # Check if we have data for both groups
        if 'meditators' not in all_metrics[networks] or 'controls' not in all_metrics[networks]:
            logger.warning(f"Missing data for at least one group in {networks}-network configuration")
            continue
            
        # Check if we have bygroup data for both groups
        if ('bygroup' not in all_metrics[networks]['meditators'] or 
            'bygroup' not in all_metrics[networks]['controls']):
            logger.warning(f"Missing bygroup metrics for at least one group in {networks}-network configuration")
            continue
        
        # Extract features for alignment
        med_features = extract_state_features_for_alignment(all_metrics[networks]['meditators']['bygroup'])
        con_features = extract_state_features_for_alignment(all_metrics[networks]['controls']['bygroup'])
        
        # Align meditator → control states
        med_means = np.array(med_features['state_means'])
        con_means = np.array(con_features['state_means'])
        
        # Get correspondence between states using a slightly stricter threshold
        med_to_con = {}
        correspondences = {}
        
        if med_means.shape[0] > 0 and con_means.shape[0] > 0:
            # Use a slightly higher threshold for between-group alignment
            alignment = align_hmm_states(med_means, con_means, similarity_threshold=0.2)
            for med_state in range(min(K, len(alignment))):
                con_state = alignment[med_state]
                if con_state != -1:  # Only add valid alignments
                    med_to_con[med_state] = con_state
            
            correspondences['med_to_con'] = med_to_con
            logger.info(f"Aligned ByGroup Meditator → Control states ({networks}-networks): {med_to_con}")
        
        # Calculate overlap matrix
        overlap_matrix = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                if i < med_means.shape[0] and j < con_means.shape[0]:
                    overlap_matrix[i, j] = state_overlap(med_means[i], con_means[j])
        
        correspondences['med_con_overlap'] = overlap_matrix
        
        # Log quality metrics on the alignment
        if med_to_con:
            mean_overlap = np.mean([overlap_matrix[i, j] for i, j in med_to_con.items()])
            logger.info(f"Mean meditator-control state overlap ({networks}-networks): {mean_overlap:.3f}")
        
        bygroup_correspondences[networks] = correspondences
        
        # Save the correspondences
        correspondence_path = os.path.join(REFERENCE_DIR, f'bygroup_med_control_{networks}networks_correspondence.pkl')
        with open(correspondence_path, 'wb') as f:
            pickle.dump(correspondences, f)
        logger.info(f"Saved bygroup group correspondences to {correspondence_path}")
    
    return bygroup_correspondences
    
def align_states_between_groups(global_metrics_by_group, networks=8):
    """Align states between meditator and control groups using global standardization."""
    correspondences = {}
    
    # Make sure we have both groups
    if 'meditators' not in global_metrics_by_group or 'controls' not in global_metrics_by_group:
        logger.warning("Missing data for at least one group")
        return correspondences
    
    # Extract features for alignment
    med_features = extract_state_features_for_alignment(global_metrics_by_group['meditators'])
    con_features = extract_state_features_for_alignment(global_metrics_by_group['controls'])
    
    # Align meditator → control states
    med_means = np.array(med_features['state_means'])
    con_means = np.array(con_features['state_means'])
    
    # Get correspondence between states
    med_to_con = {}
    if med_means.shape[0] > 0 and con_means.shape[0] > 0:
        alignment = align_hmm_states(med_means, con_means, similarity_threshold=0.2)
        for med_state in range(min(K, len(alignment))):
            con_state = alignment[med_state]
            if con_state != -1:  # Only add valid alignments
                med_to_con[med_state] = con_state
        
        correspondences['med_to_con'] = med_to_con
        logger.info(f"Aligned meditator → control states: {med_to_con}")
    
    # Calculate overlap matrix
    overlap_matrix = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if i < med_means.shape[0] and j < con_means.shape[0]:
                overlap_matrix[i, j] = state_overlap(med_means[i], con_means[j])
    
    correspondences['med_con_overlap'] = overlap_matrix
    
    # Log alignment quality
    if med_to_con:
        mean_overlap = np.mean([overlap_matrix[i, j] for i, j in med_to_con.items()])
        logger.info(f"Mean meditator-control state overlap: {mean_overlap:.3f}")
    
    return correspondences

def save_reference_benchmarks(correspondences, deviations, group, networks):
    """Save reference benchmarks to disk for use by other analyses."""
    
    # Save state correspondences
    correspondence_path = os.path.join(REFERENCE_DIR, f'{group}_{networks}networks_state_correspondence.pkl')
    with open(correspondence_path, 'wb') as f:
        pickle.dump(correspondences, f)
    logger.info(f"Saved state correspondences to {correspondence_path}")
    
    # Save deviations
    deviations_path = os.path.join(REFERENCE_DIR, f'{group}_{networks}networks_standardization_deviations.pkl')
    with open(deviations_path, 'wb') as f:
        pickle.dump(deviations, f)
    logger.info(f"Saved standardization deviations to {deviations_path}")
    
    # Create summary CSV
    summary_rows = []
    
    # Compile summary statistics for key metrics
    if 'global_to_bygroup' in deviations:
        for metric, values in deviations['global_to_bygroup'].items():
            if isinstance(values, list) and values:
                # Calculate statistics
                abs_devs = [v['abs_deviation'] for v in values]
                mean_abs_dev = np.mean(abs_devs)
                max_abs_dev = np.max(abs_devs)
                
                summary_rows.append({
                    'group': group,
                    'networks': networks,
                    'comparison': 'global_to_bygroup',
                    'metric': metric,
                    'mean_abs_deviation': mean_abs_dev,
                    'max_abs_deviation': max_abs_dev
                })
    
    # Create DataFrame and save to CSV
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(REFERENCE_DIR, f'{group}_{networks}networks_deviation_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Saved deviation summary to {summary_path}")
        
def visualize_bygroup_correspondences(bygroup_correspondences, all_metrics, networks_list=NETWORK_CONFIGS):
    """
    Create visualization of state correspondences between meditators and controls using bygroup standardization.
    Shows a row with two panels:
    - Left: 7-network correspondences
    - Right: 8-network correspondences
    """
    # Create figure with 1x2 grid (two panels side by side)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plt.suptitle(f'Meditator → Control State Alignment (ByGroup, k={K})', fontsize=18)
    
    # Plot each network configuration
    for idx, networks in enumerate(networks_list):
        ax = axes[idx]
        
        # Check if we have correspondences for this network configuration
        if networks not in bygroup_correspondences:
            ax.text(0.5, 0.5, "No correspondence data available", ha='center', va='center')
            ax.set_title(f"{networks}-networks")
            continue
        
        correspondences = bygroup_correspondences[networks]
        
        # Extract overlap matrix
        overlap_matrix = np.zeros((K, K))
        if 'med_con_overlap' in correspondences:
            overlap_matrix = correspondences['med_con_overlap']
        
        # Create correspondence indicators
        corr_markers = np.zeros((K, K))
        if 'med_to_con' in correspondences:
            for med_state, con_state in correspondences['med_to_con'].items():
                corr_markers[med_state, con_state] = 1
        
        # Calculate mean overlap
        mean_overlap = 0
        if 'med_to_con' in correspondences:
            med_to_con = correspondences['med_to_con']
            mean_overlap = np.mean([overlap_matrix[i, j] for i, j in med_to_con.items()])
        
        # Plot the overlap matrix with the same styling as standardization correspondences
        sns.heatmap(overlap_matrix, cmap='viridis', vmin=0, vmax=1, 
                   annot=True, fmt='.2f', cbar=True,
                   ax=ax, cbar_kws={'label': 'State Overlap'})
        
        # Highlight the corresponding states with bold red borders
        for i in range(K):
            for j in range(K):
                if corr_markers[i, j] > 0:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, 
                                             edgecolor='red', lw=2))
        
        # Add labels
        ax.set_xlabel('Control State')
        ax.set_ylabel('Meditator State')
        ax.set_title(f"{networks}-networks\nMean Overlap: {mean_overlap:.3f}")
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure with high resolution
    plt.savefig(os.path.join(VIS_DIR, 'bygroup_med_control_correspondence.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()

def visualize_state_activation_patterns(bygroup_correspondences, all_metrics, networks_list=NETWORK_CONFIGS):
    """
    Create visualization of network activation patterns for corresponding states between 
    meditators and controls.
    """
    for networks in networks_list:
        # Check if we have correspondences for this network configuration
        if networks not in bygroup_correspondences:
            logger.warning(f"No correspondence data available for {networks}-networks")
            continue
        
        correspondences = bygroup_correspondences[networks]
        
        # Check if we have the med_to_con mapping
        if 'med_to_con' not in correspondences or not correspondences['med_to_con']:
            logger.warning(f"No state mapping available for {networks}-networks")
            continue
        
        # Check if we have metrics data
        if ('meditators' not in all_metrics[networks] or 
            'controls' not in all_metrics[networks] or
            'bygroup' not in all_metrics[networks]['meditators'] or
            'bygroup' not in all_metrics[networks]['controls']):
            logger.warning(f"Missing metrics data for {networks}-networks")
            continue
        
        # Extract network fields
        med_metrics = all_metrics[networks]['meditators']['bygroup']
        con_metrics = all_metrics[networks]['controls']['bygroup']
        
        if ('temporal_metrics' not in med_metrics or 
            'network_fields' not in med_metrics['temporal_metrics']):
            logger.warning(f"Missing network fields for {networks}-networks")
            continue
        
        network_fields = med_metrics['temporal_metrics']['network_fields']
        
        # Create a figure for this network configuration
        # One row per state correspondence
        med_to_con = correspondences['med_to_con']
        n_correspondences = len(med_to_con)
        
        if n_correspondences == 0:
            continue
        
        fig, axes = plt.subplots(n_correspondences, 1, figsize=(10, 5*n_correspondences)) #changed from k=4 to 5
        if n_correspondences == 1:
            axes = [axes]  # Make it a list for consistent indexing
        
        plt.suptitle(f'Network Activation Patterns for Corresponding States ({networks}-networks)', fontsize=16)
        
        # Get state means
        med_means = med_metrics['temporal_metrics']['state_means']
        con_means = con_metrics['temporal_metrics']['state_means']
        
        # Plot each correspondence
        for i, (med_state, con_state) in enumerate(med_to_con.items()):
            ax = axes[i]
            
            # Plot meditator state
            width = 0.35
            x = np.arange(len(network_fields))

            # Normalize patterns for visualization
            med_norm = get_normalized_activations(med_means[med_state])
            con_norm = get_normalized_activations(con_means[con_state])

            # Meditator bars (normalized)
            med_bars = ax.bar(x - width/2, med_norm, width, label=f'Meditator State {med_state}')

            # Control bars (normalized)
            con_bars = ax.bar(x + width/2, con_norm, width, label=f'Control State {con_state}')
            
            # Add labels and styling
            ax.set_xlabel('Network')
            ax.set_ylabel('Activation (z-score)')
            ax.set_title(f'Meditator State {med_state} ↔ Control State {con_state}')
            ax.set_xticks(x)
            ax.set_xticklabels(network_fields, rotation=45, ha='right')
            ax.legend()
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # Add overlap value as annotation
            overlap = correspondences['med_con_overlap'][med_state, con_state]
            ax.annotate(f'Overlap: {overlap:.2f}', xy=(0.95, 0.95), xycoords='axes fraction',
                       horizontalalignment='right', verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(VIS_DIR, f'bygroup_state_activation_patterns_{networks}networks.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
def visualize_kmeans_clustering(bygroup_correspondences, all_metrics, networks_list=NETWORK_CONFIGS):
    """
    Create visualization of state clustering using K-means and PCA for dimensionality reduction.
    This helps in interpreting the correspondence between meditator and control states.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    
    for networks in networks_list:
        # Check if we have correspondences for this network configuration
        if networks not in bygroup_correspondences:
            logger.warning(f"No correspondence data available for {networks}-networks")
            continue
        
        correspondences = bygroup_correspondences[networks]
        
        # Check if we have the med_to_con mapping
        if 'med_to_con' not in correspondences:
            logger.warning(f"No state mapping available for {networks}-networks")
            continue
        
        # Check if we have metrics data
        if ('meditators' not in all_metrics[networks] or 
            'controls' not in all_metrics[networks] or
            'bygroup' not in all_metrics[networks]['meditators'] or
            'bygroup' not in all_metrics[networks]['controls']):
            logger.warning(f"Missing metrics data for {networks}-networks")
            continue
        
        # Extract network fields
        med_metrics = all_metrics[networks]['meditators']['bygroup']
        con_metrics = all_metrics[networks]['controls']['bygroup']
        
        if ('temporal_metrics' not in med_metrics or 
            'state_means' not in med_metrics['temporal_metrics']):
            logger.warning(f"Missing state means for {networks}-networks")
            continue
        
        # Get state means
        med_means = np.array(med_metrics['temporal_metrics']['state_means'])
        con_means = np.array(con_metrics['temporal_metrics']['state_means'])
        
        # Prepare data for clustering
        combined_means = np.vstack((med_means, con_means))
        
        # Standardize features
        scaler = StandardScaler()
        combined_means_scaled = scaler.fit_transform(combined_means)
        
        # Apply K-means - let's use K=K as a starting point
        kmeans = KMeans(n_clusters=K, random_state=0, n_init=10)
        clusters = kmeans.fit_predict(combined_means_scaled)
        
        # Apply PCA just for visualization purposes
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(combined_means_scaled)
        
        # Split results back to meditator and control
        med_pca = pca_result[:len(med_means)]
        con_pca = pca_result[len(med_means):]
        
        # Split clusters
        med_clusters = clusters[:len(med_means)]
        con_clusters = clusters[len(med_means):]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot clusters
        cluster_colors = plt.cm.viridis(np.linspace(0, 1, K))
        
        # Plot meditator states
        for i in range(len(med_means)):
            ax.scatter(med_pca[i, 0], med_pca[i, 1], c=[cluster_colors[med_clusters[i]]], 
                      marker='o', s=100, edgecolors='black')
            ax.text(med_pca[i, 0] + 0.1, med_pca[i, 1] + 0.1, f'M{i}', fontsize=12)
        
        # Plot control states
        for i in range(len(con_means)):
            ax.scatter(con_pca[i, 0], con_pca[i, 1], c=[cluster_colors[con_clusters[i]]], 
                      marker='s', s=100, edgecolors='black')
            ax.text(con_pca[i, 0] + 0.1, con_pca[i, 1] + 0.1, f'C{i}', fontsize=12)
        
        # Draw lines connecting corresponding states
        for med_state, con_state in correspondences['med_to_con'].items():
            ax.plot([med_pca[med_state, 0], con_pca[con_state, 0]], 
                   [med_pca[med_state, 1], con_pca[con_state, 1]], 
                   'k--', alpha=0.7, linewidth=2)
            
            # Add overlap value as text near midpoint
            midx = (med_pca[med_state, 0] + con_pca[con_state, 0]) / 2
            midy = (med_pca[med_state, 1] + con_pca[con_state, 1]) / 2
            overlap = correspondences['med_con_overlap'][med_state, con_state]
            ax.text(midx, midy, f'{overlap:.2f}', fontsize=10, 
                   bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
        
        # Add legend
        ax.scatter([], [], marker='o', s=100, c='gray', edgecolors='black', label='Meditator')
        ax.scatter([], [], marker='s', s=100, c='gray', edgecolors='black', label='Control')
        
        # Add cluster legend
        for i in range(K):
            ax.scatter([], [], c=[cluster_colors[i]], marker='o', s=50, label=f'Cluster {i}')
        
        ax.set_title(f'K-means Clustering of Brain States ({networks}-networks)')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f}%)')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(VIS_DIR, f'bygroup_kmeans_clustering_{networks}networks.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
def visualize_standardization_correspondences(all_correspondences, all_metrics, networks_list=NETWORK_CONFIGS):
    """ Create a consolidated visualization of state correspondences across groups.
    Uses a 2x2 grid:
    - Top row: 7-network alignments (controls | meditators)
    - Bottom row: 8-network alignments (controls | meditators)
    """
    # Create figure with 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    plt.suptitle(f'Global → ByGroup State Alignment (k={K})', fontsize=18)
    
    # Plot each group and network configuration
    for row_idx, networks in enumerate(networks_list):
        for col_idx, group in enumerate(GROUPS):
            # Get the axis for this subplot
            ax = axes[row_idx, col_idx]
            
            # Check if data exists
            if group not in all_correspondences[networks] or group not in all_metrics[networks]:
                ax.text(0.5, 0.5, "No data available", ha='center', va='center')
                ax.set_title(f"{group} - {networks}-networks")
                continue
                
            correspondences = all_correspondences[networks][group]
            metrics_by_std = all_metrics[networks][group]
            
            # Extract overlap matrix
            overlap_matrix = np.zeros((K, K))
            if 'global_bygroup_overlap' in correspondences:
                overlap_matrix = correspondences['global_bygroup_overlap']
            
            # Create correspondence indicators
            corr_markers = np.zeros((K, K))
            if 'global_to_bygroup' in correspondences:
                for global_state, bygroup_state in correspondences['global_to_bygroup'].items():
                    corr_markers[global_state, bygroup_state] = 1
            
            # Calculate mean overlap
            mean_overlap = 0
            if 'global_to_bygroup' in correspondences:
                global_to_bygroup = correspondences['global_to_bygroup']
                mean_overlap = np.mean([overlap_matrix[i, j] for i, j in global_to_bygroup.items()])
            
            # Plot the overlap matrix with improved visualization
            sns.heatmap(overlap_matrix, cmap='viridis', vmin=0, vmax=1, 
                       annot=True, fmt='.2f', cbar=True if col_idx == 1 else False,
                       ax=ax, cbar_kws={'label': 'State Overlap'} if col_idx == 1 else {})
            
            # Highlight the corresponding states with bold red borders
            for i in range(K):
                for j in range(K):
                    if corr_markers[i, j] > 0:
                        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, 
                                                 edgecolor='red', lw=2))
            
            # Add labels
            ax.set_xlabel('ByGroup State')
            ax.set_ylabel('Global State')
            ax.set_title(f"{group.capitalize()} - {networks}-networks\nMean Overlap: {mean_overlap:.3f}")
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure with high resolution
    plt.savefig(os.path.join(VIS_DIR, 'standardization_correspondence_summary.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to run the standardization reference pipeline."""
    logger.info("=== Starting Standardization Reference Analysis ===")
    
    # Store data for consolidated visualization
    all_correspondences = {7: {}, 8: {}}
    all_metrics = {7: {}, 8: {}}
    
    # Store metrics for group comparison
    global_metrics_by_group = {7: {}, 8: {}}
    
    # Process each group and network configuration
    for group in GROUPS:
        for networks in NETWORK_CONFIGS:
            logger.info(f"\n=== Processing {group}, {networks}-network configuration ===")
            
            try:
                # Load metrics from standardization methods
                metrics_by_std = load_metrics_all_standardizations(group, networks)
                
                if len(metrics_by_std) < 2:
                    logger.warning(f"Not enough standardization methods available for {group}, {networks}-networks")
                    continue
                
                # Store metrics for consolidated visualization
                all_metrics[networks][group] = metrics_by_std
                
                # Store global metrics for later group comparison
                if 'global' in metrics_by_std:
                    global_metrics_by_group[networks][group] = metrics_by_std['global']
                
                # Align states across standardization methods
                correspondences = align_states_across_standardizations(metrics_by_std)
                
                # Store correspondences for consolidated visualization
                all_correspondences[networks][group] = correspondences

                # Calculate deviations in metrics (still useful for reference)
                deviations = calculate_standardization_deviations(metrics_by_std, correspondences, group, networks)

                # Save reference benchmarks
                save_reference_benchmarks(correspondences, deviations, group, networks)

                logger.info(f"Completed standardization analysis for {group}, {networks}-networks")
                
            except Exception as e:
                logger.error(f"Error in standardization analysis: {str(e)}")
                import traceback
                traceback.print_exc()
    
    # Create consolidated visualization
    visualize_standardization_correspondences(all_correspondences, all_metrics)
    
    # Now perform global standardization group alignments
    logger.info("\n=== Aligning States Between Groups (Global Standardization) ===")
    for networks in NETWORK_CONFIGS:
        if len(global_metrics_by_group[networks]) == 2:  # Both groups present
            group_correspondences = align_states_between_groups(global_metrics_by_group[networks], networks)
            
            # Save group correspondences
            correspondence_path = os.path.join(REFERENCE_DIR, f'meditator_control_{networks}networks_correspondence.pkl')
            with open(correspondence_path, 'wb') as f:
                pickle.dump(group_correspondences, f)
            logger.info(f"Saved group correspondences to {correspondence_path}")
    
    # Perform bygroup standardization group alignments with enhanced visualizations
    logger.info("\n=== Aligning States Between Groups (ByGroup Standardization) ===")
    bygroup_correspondences = align_bygroup_states_between_groups(all_metrics)
    
    # Create visualizations for bygroup correspondences
    visualize_bygroup_correspondences(bygroup_correspondences, all_metrics)
    
    # Create activation pattern visualizations
    visualize_state_activation_patterns(bygroup_correspondences, all_metrics)
    
    # Create kmeans clustering visualization
    visualize_kmeans_clustering(bygroup_correspondences, all_metrics)
    
    # Report completion
    logger.info(f"\n=== Standardization Reference Analysis completed ===")
    
if __name__ == "__main__":
    main()