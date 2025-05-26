"""
Subject-Level Permutation Testing for Fractional Occupancy in Meditation Research

This script performs permutation testing on subject-level fractional occupancy data
to identify statistically significant differences between meditators and controls.

Methodological Separation:

FO testing examines "where" brain activity occurs (state occupancy)
Transition testing examines "how" brain activity changes (state dynamics)
Author: [Prakash Kavi]
"""

import os
import numpy as np
import pandas as pd
import pickle
import logging
import argparse
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup paths
ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')
OUTPUT_DIR = os.path.join(RESULTS_DIR, 'permutation_tests')

def load_subject_fo_data(networks=7, k=4, standardize_method='bygroup'):
    """Load subject-level FO data from metrics files."""
    logger.info(f"Loading subject-level FO data for {networks}-network, k={k}...")
    
    metrics_base = os.path.join(METRICS_DIR, standardize_method, f'{networks}networks')
    
    # Load meditator data
    med_path = os.path.join(metrics_base, 'meditators', f'k{k}_metrics.pkl')
    if not os.path.exists(med_path):
        raise FileNotFoundError(f"Meditator metrics file not found: {med_path}")
    
    with open(med_path, 'rb') as f:
        med_metrics = pickle.load(f)
    
    # Load control data
    con_path = os.path.join(metrics_base, 'controls', f'k{k}_metrics.pkl')
    if not os.path.exists(con_path):
        raise FileNotFoundError(f"Control metrics file not found: {con_path}")
    
    with open(con_path, 'rb') as f:
        con_metrics = pickle.load(f)
    
    # Extract subject-level FO data
    try:
        med_fo = med_metrics['temporal_metrics']['FO']  # Shape: (n_med_subjects, k)
        con_fo = con_metrics['temporal_metrics']['FO']  # Shape: (n_con_subjects, k)
        
        logger.info(f"Loaded FO data for {med_fo.shape[0]} meditators and {con_fo.shape[0]} controls")
        logger.info(f"Data shapes - Meditators: {med_fo.shape}, Controls: {con_fo.shape}")
        
        # Check if the data is as expected
        if med_fo.shape[1] != k or con_fo.shape[1] != k:
            logger.warning(f"Expected k={k} states, but found {med_fo.shape[1]} for meditators and {con_fo.shape[1]} for controls")
        
        return {
            'meditators': med_fo, 
            'controls': con_fo,
            'med_mean': med_metrics['temporal_metrics']['FO_mean'],
            'con_mean': con_metrics['temporal_metrics']['FO_mean']
        }
    
    except KeyError as e:
        logger.error(f"Could not find FO data in metrics files: {e}")
        logger.info(f"Available keys in meditator metrics: {list(med_metrics.keys())}")
        if 'temporal_metrics' in med_metrics:
            logger.info(f"Available temporal metrics: {list(med_metrics['temporal_metrics'].keys())}")
        raise

def permutation_test_fo_subjects(fo_data, n_permutations=1000):
    """Perform permutation testing on subject-level FO data."""
    logger.info(f"Starting subject-level FO permutation testing with {n_permutations} permutations...")
    start_time = time.time()
    
    med_fo = fo_data['meditators']  # Shape: (n_med_subjects, k)
    con_fo = fo_data['controls']    # Shape: (n_con_subjects, k)
    
    n_med = med_fo.shape[0]
    n_con = con_fo.shape[0]
    k = med_fo.shape[1]
    
    logger.info(f"Testing {n_med} meditators vs {n_con} controls across {k} states")
    
    # Calculate observed difference between group means
    med_means = np.mean(med_fo, axis=0)
    con_means = np.mean(con_fo, axis=0)
    observed_diff = med_means - con_means
    
    logger.info(f"Observed FO differences (meditators - controls): {observed_diff}")
    
    # Combine all subjects' data
    all_fo = np.vstack((med_fo, con_fo))
    all_subjects = n_med + n_con
    
    # Initialize null distribution
    null_distribution = np.zeros((n_permutations, k))
    
    # Run permutations
    for perm in range(n_permutations):
        if perm % 200 == 0:
            elapsed = (time.time() - start_time) / 60
            logger.info(f"Completed {perm}/{n_permutations} permutations ({elapsed:.2f} min elapsed)")
        
        # Randomly shuffle subject indices
        perm_indices = np.random.permutation(all_subjects)
        perm_med_indices = perm_indices[:n_med]
        perm_con_indices = perm_indices[n_med:]
        
        # Calculate permuted group means
        perm_med_mean = np.mean(all_fo[perm_med_indices], axis=0)
        perm_con_mean = np.mean(all_fo[perm_con_indices], axis=0)
        
        # Calculate difference
        null_distribution[perm] = perm_med_mean - perm_con_mean
    
    # Calculate p-values (two-tailed test)
    p_values = np.zeros(k)
    for i in range(k):
        # Two-tailed test using absolute values
        obs_abs_diff = np.abs(observed_diff[i])
        null_abs_diffs = np.abs(null_distribution[:, i])
        
        # p-value = (count of null values more extreme than observed + 1) / (n_permutations + 1)
        p_values[i] = (np.sum(null_abs_diffs >= obs_abs_diff) + 1) / (n_permutations + 1)
    
    # Apply FDR correction
    reject, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
    significant = p_corrected < 0.05
    
    elapsed_time = (time.time() - start_time) / 60
    logger.info(f"FO permutation testing completed in {elapsed_time:.2f} minutes")
    
    return {
        'observed_diff': observed_diff,
        'null_distribution': null_distribution,
        'p_values': p_values,
        'p_corrected': p_corrected,
        'significant': significant,
        'n_permutations': n_permutations,
        'n_meditators': n_med,
        'n_controls': n_con
    }
    
def transition_permutation_test(med_trans_matrices, con_trans_matrices, n_permutations=5000):
    """Perform permutation test on transition matrices between groups.
    
    Parameters:
    -----------
    med_trans_matrices : list of np.ndarray
        List of subject-level transition matrices for meditators
    con_trans_matrices : list of np.ndarray
        List of subject-level transition matrices for controls
    n_permutations : int
        Number of permutations to run
        
    Returns:
    --------
    Dictionary with test results
    """
    k = med_trans_matrices[0].shape[0]  # Number of states
    
    # Observed difference in average transition matrices
    med_avg = np.mean(med_trans_matrices, axis=0)
    con_avg = np.mean(con_trans_matrices, axis=0)
    observed_diff = med_avg - con_avg
    
    # Combined data for permutation
    all_matrices = med_trans_matrices + con_trans_matrices
    n_med = len(med_trans_matrices)
    n_con = len(con_trans_matrices)
    n_total = n_med + n_con
    
    # Initialize results storage
    null_distribution = np.zeros((n_permutations, k, k))
    
    # Run permutations
    for perm in range(n_permutations):
        # Randomly shuffle subject indices
        perm_indices = np.random.permutation(n_total)
        perm_med_indices = perm_indices[:n_med]
        perm_con_indices = perm_indices[n_med:]
        
        # Calculate permuted group means
        perm_med_mean = np.mean([all_matrices[i] for i in perm_med_indices], axis=0)
        perm_con_mean = np.mean([all_matrices[i] for i in perm_con_indices], axis=0)
        
        # Store difference
        null_distribution[perm] = perm_med_mean - perm_con_mean
    
    # Calculate p-values for each transition
    p_values = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            # Two-tailed test using absolute values
            obs_abs_diff = np.abs(observed_diff[i, j])
            null_abs_diffs = np.abs(null_distribution[:, i, j])
            
            # p-value calculation
            p_values[i, j] = (np.sum(null_abs_diffs >= obs_abs_diff) + 1) / (n_permutations + 1)
    
    # FDR correction for multiple comparisons
    from statsmodels.stats.multitest import multipletests
    
    # Flatten for correction
    p_flat = p_values.flatten()
    reject, p_corrected, _, _ = multipletests(p_flat, alpha=0.05, method='fdr_bh')
    p_corrected = p_corrected.reshape(k, k)
    significant = p_corrected < 0.05
    
    return {
        'observed_diff': observed_diff,
        'null_distribution': null_distribution,
        'p_values': p_values,
        'p_corrected': p_corrected,
        'significant': significant
    }

def visualize_results(perm_results, k, networks, output_dir):
    """Create visualizations of the FO permutation results."""
    logger.info("Generating FO permutation visualizations...")
    
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot FO differences with significance markers
    plt.figure(figsize=(10, 6))
    
    # Set up bar positions and colors
    x = np.arange(k)
    colors = ['red' if d > 0 else 'blue' for d in perm_results['observed_diff']]
    
    # Create bars
    bars = plt.bar(x, perm_results['observed_diff'], color=colors, alpha=0.7)
    
    # Add significance markers
    for i, bar in enumerate(bars):
        height = bar.get_height()
        sign = 1 if height >= 0 else -1
        
        if perm_results['significant'][i]:
            marker = "**"
        elif perm_results['p_values'][i] < 0.05:
            marker = "*"
        else:
            marker = ""
            
        if marker:
            plt.text(i, height + sign*0.01, marker, 
                    ha='center', va='bottom' if sign > 0 else 'top', 
                    fontweight='bold')
    
    # Add value labels
    for i, v in enumerate(perm_results['observed_diff']):
        plt.text(i, v + (0.01 if v >= 0 else -0.03), 
                f"{v:.3f}", ha='center', va='bottom' if v >= 0 else 'top')
    
    # Styling
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.xlabel('State', fontsize=12)
    plt.ylabel('Difference in Fractional Occupancy\n(Meditators - Controls)', fontsize=12)
    plt.title(f'State Occupancy Differences: {networks}-Network, k={k}\n* p<0.05 (uncorrected), ** p<0.05 (FDR-corrected)', 
             fontsize=14)
    plt.xticks(x, [f'State {i}' for i in range(k)])
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    fo_plot_path = os.path.join(plots_dir, f'{networks}net_k{k}_subject_fo_differences.png')
    plt.savefig(fo_plot_path, dpi=300)
    plt.savefig(os.path.join(plots_dir, f'{networks}net_k{k}_subject_fo_differences.pdf'))
    plt.close()
    
    # Plot p-values
    plt.figure(figsize=(10, 6))
    
    # Create grouped bars for p-values
    width = 0.35
    plt.bar(x - width/2, perm_results['p_values'], width, label='Uncorrected', color='steelblue')
    plt.bar(x + width/2, perm_results['p_corrected'], width, label='FDR-corrected', color='darkblue')
    
    # Add significance threshold line
    plt.axhline(y=0.05, color='red', linestyle='--', label='p=0.05 threshold')
    
    # Styling
    plt.xlabel('State', fontsize=12)
    plt.ylabel('p-value', fontsize=12)
    plt.title(f'Statistical Significance of FO Differences: {networks}-Network, k={k}', fontsize=14)
    plt.xticks(x, [f'State {i}' for i in range(k)])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    pval_plot_path = os.path.join(plots_dir, f'{networks}net_k{k}_subject_fo_pvalues.png')
    plt.savefig(pval_plot_path, dpi=300)
    plt.close()
    
    # Plot null distributions for each state
    for i in range(k):
        plt.figure(figsize=(8, 5))
        null_vals = perm_results['null_distribution'][:, i]
        observed = perm_results['observed_diff'][i]
        
        plt.hist(null_vals, bins=30, alpha=0.7, color='gray')
        plt.axvline(x=observed, color='red', linestyle='--', linewidth=2)
        
        plt.title(f'Permutation Distribution: State {i}, {networks}-Network, k={k}')
        plt.xlabel('Difference in Fractional Occupancy')
        plt.ylabel('Frequency')
        plt.text(0.95, 0.95, 
                f'observed = {observed:.3f}\np = {perm_results["p_values"][i]:.3f}\np(FDR) = {perm_results["p_corrected"][i]:.3f}',
                transform=plt.gca().transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.tight_layout()
        
        null_plot_path = os.path.join(plots_dir, f'{networks}net_k{k}_subject_fo_null_dist_state{i}.png')
        plt.savefig(null_plot_path, dpi=300)
        plt.close()
    
    logger.info(f"FO permutation visualizations saved to {plots_dir}")
    return plots_dir

def save_results(perm_results, k, networks, output_dir):
    """Save permutation results to files."""
    # Save raw results as pickle
    results_path = os.path.join(output_dir, f'{networks}net_k{k}_subject_fo_permutation_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(perm_results, f)
    
    # Save text summary
    summary_path = os.path.join(output_dir, f'{networks}net_k{k}_subject_fo_permutation_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write(f"SUBJECT-LEVEL FRACTIONAL OCCUPANCY PERMUTATION TEST RESULTS\n")
        f.write(f"{networks}-NETWORK CONFIGURATION, k={k}\n")
        f.write("==========================================================\n\n")
        
        f.write(f"Number of subjects: {perm_results['n_meditators']} meditators, {perm_results['n_controls']} controls\n")
        f.write(f"Number of permutations: {perm_results['n_permutations']}\n")
        f.write(f"Test date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("OBSERVED DIFFERENCES (Meditators - Controls):\n\n")
        f.write("State   Difference    P-value    FDR-corrected P\n")
        f.write("------------------------------------------------\n")
        
        for i in range(k):
            value = perm_results['observed_diff'][i]
            p = perm_results['p_values'][i]
            p_corr = perm_results['p_corrected'][i]
            
            if perm_results['significant'][i]:
                sig_marker = "**"
            elif p < 0.05:
                sig_marker = "*"
            else:
                sig_marker = ""
                
            f.write(f"S{i}      {value:+.4f}{sig_marker}    {p:.4f}     {p_corr:.4f}\n")
        
        f.write("\n* p<0.05 (uncorrected), ** p<0.05 (FDR-corrected)\n\n")
        
        # Count significant results
        n_uncorrected = np.sum(perm_results['p_values'] < 0.05)
        n_corrected = np.sum(perm_results['significant'])
        
        f.write(f"\nSummary: {n_uncorrected} states show significant differences at p<0.05 (uncorrected)\n")
        f.write(f"         {n_corrected} states show significant differences after FDR correction\n")
    
    logger.info(f"FO permutation results saved to {summary_path}")
    return summary_path

def main():
    """Main function to run subject-level FO permutation testing."""
    parser = argparse.ArgumentParser(description='Subject-Level Permutation Testing for Fractional Occupancy')
    parser.add_argument('--permutations', type=int, default=5000,
                        help='Number of permutations to run (default: 1000)')
    parser.add_argument('--networks', type=int, default=7,
                        help='Network configuration to analyze (default: 7)')
    parser.add_argument('--k', type=int, default=4,
                        help='Number of states (default: 4)')
    parser.add_argument('--standardize', type=str, default='bygroup',
                        help='Standardization method (default: bygroup)')
    args = parser.parse_args()
    
    logger.info("=== Starting Subject-Level Fractional Occupancy Permutation Testing ===")
    logger.info(f"Running {args.permutations} permutations for {args.networks}-network, k={args.k}")
    
    start_time = time.time()
    
    try:
        # Create output directory
        network_dir = os.path.join(OUTPUT_DIR, f'{args.networks}networks')
        os.makedirs(network_dir, exist_ok=True)
        
        # Load subject-level FO data
        fo_data = load_subject_fo_data(args.networks, args.k, args.standardize)
        
        # Run permutation test
        perm_results = permutation_test_fo_subjects(fo_data, args.permutations)
        
        # Visualize results
        plots_dir = visualize_results(perm_results, args.k, args.networks, network_dir)
        
        # Save results
        summary_path = save_results(perm_results, args.k, args.networks, network_dir)
        
        # Report completion
        elapsed_time = (time.time() - start_time) / 60
        logger.info(f"=== Subject-level FO permutation testing completed in {elapsed_time:.2f} minutes ===")
        logger.info(f"Results saved to {network_dir}")
        
    except Exception as e:
        logger.error(f"Error in subject-level FO permutation testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()