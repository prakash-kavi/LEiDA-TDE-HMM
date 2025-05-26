"""
GLHMM-Integrated Permutation Testing for State Transitions in Meditation Research

This script performs permutation testing on state succession probabilities and
fractional occupancy to identify statistically significant differences between 
meditators and controls. Implementation aligns with the GLHMM statistical framework.

Key features:
1. Loads succession data saved by state_transition_analysis.py
2. Uses GLHMM's test_across_subjects with optimized parallel processing
3. Performs both transition probability and fractional occupancy analysis
4. Saves comprehensive test statistics for further analysis
5. Separates computation from visualization (use visualize_glhmm_results.py for plots)

References:
- Vidaurre et al. (2025). The Gaussian-linear hidden Markov model: A Python package.
  https://direct.mit.edu/imag/article/doi/10.1162/imag_a_00460/127499
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import logging
import argparse
import time
import multiprocessing
from datetime import datetime
from pathlib import Path
from statsmodels.stats.multitest import multipletests
from glhmm import statistics

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup paths - using pathlib for more robust path handling
ROOT_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = ROOT_DIR / 'data'
RESULTS_DIR = ROOT_DIR / 'results'
METRICS_DIR = RESULTS_DIR / 'metrics'
TRANSITIONS_DIR = RESULTS_DIR / 'transitions'

def get_paths(networks, k):
    """Create standardized paths for input/output data."""
    # Input data paths
    data_path = TRANSITIONS_DIR / 'glhmm_analysis' / f'succession_data_{networks}networks_k{k}.pkl'
    
    # Output data paths
    output_base = TRANSITIONS_DIR / 'glhmm_analysis' / f'{networks}networks'
    
    return {
        'input_data': data_path,
        'output_dir': output_base
    }

def load_succession_data(networks, k):
    """Load succession data for permutation testing."""
    paths = get_paths(networks, k)
    data_path = paths['input_data']
    
    if not data_path.exists():
        raise FileNotFoundError(f"Succession data not found: {data_path}. Run state_transition_analysis.py first.")
    
    try:
        with open(data_path, 'rb') as f:
            succession_data = pickle.load(f)
        
        logger.info(f"Loaded succession data for {networks}-network, k={k} configuration")
        return succession_data
    except Exception as e:
        logger.error(f"Error loading succession data: {str(e)}")
        raise

def compute_succession_matrices(succession_data):
    """Compute succession matrices from subject-level data."""
    controls_data = succession_data['controls']
    meditators_data = succession_data['meditators']
    k = succession_data['k']
    
    # Extract successions into arrays
    controls_successions = np.zeros((len(controls_data), k, k))
    for i, subj_data in enumerate(controls_data):
        controls_successions[i] = subj_data['succession_probs']
    
    meditators_successions = np.zeros((len(meditators_data), k, k))
    for i, subj_data in enumerate(meditators_data):
        meditators_successions[i] = subj_data['succession_probs']
    
    # Compute group means
    controls_mean = np.nanmean(controls_successions, axis=0)
    meditators_mean = np.nanmean(meditators_successions, axis=0)
    
    # Handle NaNs (transitions that don't occur for any subject)
    controls_mean = np.nan_to_num(controls_mean)
    meditators_mean = np.nan_to_num(meditators_mean)
    
    return {
        'controls': controls_successions,
        'meditators': meditators_successions,
        'controls_mean': controls_mean,
        'meditators_mean': meditators_mean,
        'observed_diff': meditators_mean - controls_mean
    }

def extract_fractional_occupancy(succession_data):
    """Extract fractional occupancy data from succession data."""
    controls_data = succession_data['controls']
    meditators_data = succession_data['meditators']
    k = succession_data['k']
    
    # Extract fractional occupancy
    controls_fo = np.zeros((len(controls_data), k))
    for i, subj_data in enumerate(controls_data):
        if 'fractional_occupancy' in subj_data:
            controls_fo[i] = subj_data['fractional_occupancy']
    
    meditators_fo = np.zeros((len(meditators_data), k))
    for i, subj_data in enumerate(meditators_data):
        if 'fractional_occupancy' in subj_data:
            meditators_fo[i] = subj_data['fractional_occupancy']
    
    return controls_fo, meditators_fo

def prepare_data_for_glhmm(succession_matrices):
    """Convert 3D succession matrices to GLHMM-compatible 2D format."""
    controls_data = succession_matrices['controls']  # (n_controls, k, k)
    meditators_data = succession_matrices['meditators']  # (n_meditators, k, k)
    
    n_controls = controls_data.shape[0]
    n_meditators = meditators_data.shape[0]
    k = controls_data.shape[1]  # Number of states
    
    # Flatten transition matrices for each subject
    controls_flat = controls_data.reshape(n_controls, k*k)
    meditators_flat = meditators_data.reshape(n_meditators, k*k)
    
    # Stack data and create group labels (0=control, 1=meditator)
    D_data = np.vstack([controls_flat, meditators_flat])
    group_labels = np.zeros(n_controls + n_meditators)
    group_labels[n_controls:] = 1
    
    return D_data, group_labels, n_controls, n_meditators, k

def permutation_test_glhmm(succession_data, n_permutations=5000):
    """Perform permutation testing using GLHMM statistical framework."""
    logger.info(f"Starting GLHMM permutation testing with {n_permutations} permutations...")
    start_time = time.time()
    
    # Prepare data
    k = succession_data['k']
    succession_matrices = compute_succession_matrices(succession_data)
    D_data, group_labels, n_controls, n_meditators, k = prepare_data_for_glhmm(succession_matrices)
    
    # Determine optimal core count (~80% of available logical processors)
    n_cores = int(0.8 * multiprocessing.cpu_count())
    logger.info(f"Using {n_cores} cores for parallel processing")
    
    # Run GLHMM permutation test with univariate method (test each transition individually)
    logger.info("Running univariate tests for each state transition...")
    glhmm_results = statistics.test_across_subjects(
        D_data=D_data,
        R_data=group_labels,
        method="univariate",
        Nperm=n_permutations  
    )
    
    # Debug: Print the keys available in the results dictionary
    logger.info(f"Available keys in GLHMM results: {list(glhmm_results.keys())}")
    
    # Reshape p-values to transition matrix format
    p_values = glhmm_results["pval"].reshape(-1, k, k)[0]
    
    # Apply FDR correction
    logger.info("Applying FDR correction for multiple comparisons...")
    pval_corrected, _ = statistics.pval_correction(glhmm_results, method="fdr_bh")
    p_corrected = pval_corrected.reshape(-1, k, k)[0]
    
    # Create significance mask
    significant = p_corrected < 0.05
    
    # Convert t-stats (stored in base_statistics) to transition matrix format
    t_stats = glhmm_results["base_statistics"].reshape(-1, k, k)[0]
    
    # Convert null distributions to our matrix format
    null_distribution = np.zeros((n_permutations, k, k))
    for i in range(k):
        for j in range(k):
            idx = i * k + j
            null_distribution[:, i, j] = glhmm_results["base_stat_distribution"][:, idx]
    
    # Create result dictionary
    perm_results = {
        'observed_diff': succession_matrices['observed_diff'],
        'null_distribution': null_distribution,
        'p_values': p_values,
        'p_corrected': p_corrected,
        'significant': significant,
        'n_permutations': n_permutations,
        'glhmm_results': glhmm_results,  # Keep raw GLHMM results
        't_statistics': t_stats,
    }
    
    elapsed_time = (time.time() - start_time) / 60
    logger.info(f"GLHMM permutation testing completed in {elapsed_time:.2f} minutes")
    
    return perm_results

def analyze_fractional_occupancy(succession_data, n_permutations=5000):
    """Analyze fractional occupancy differences between groups using GLHMM."""
    logger.info("Starting fractional occupancy analysis...")
    start_time = time.time()
    
    # Extract FO data
    k = succession_data['k']
    controls_fo, meditators_fo = extract_fractional_occupancy(succession_data)
    
    # Prepare data for GLHMM
    n_controls = controls_fo.shape[0]
    n_meditators = meditators_fo.shape[0] 
    D_data = np.vstack([controls_fo, meditators_fo])
    group_labels = np.zeros(n_controls + n_meditators)
    group_labels[n_controls:] = 1
    
    # Calculate observed difference
    observed_diff = np.mean(meditators_fo, axis=0) - np.mean(controls_fo, axis=0)
    
    # Optimize core count
    n_cores = int(0.8 * multiprocessing.cpu_count())
    
    # Run GLHMM permutation test
    logger.info(f"Running FO analysis with {n_permutations} permutations...")
    fo_results = statistics.test_across_subjects(
        D_data=D_data,
        R_data=group_labels,
        method="univariate",
        Nperm=n_permutations  
    )
    
    # Apply FDR correction
    pval_corrected, _ = statistics.pval_correction(fo_results, method="fdr_bh")
    
    # Create result dictionary
    fo_analysis = {
        'observed_diff': observed_diff,
        'glhmm_results': fo_results,
        'p_values': fo_results["pval"].flatten(),
        'p_corrected': pval_corrected.flatten(),
        'significant': pval_corrected.flatten() < 0.05,
        'controls_fo': controls_fo,
        'meditators_fo': meditators_fo,
    }
    
    elapsed_time = (time.time() - start_time) / 60
    logger.info(f"Fractional occupancy analysis completed in {elapsed_time:.2f} minutes")
    
    return fo_analysis

def save_results_summary(perm_results, fo_analysis, k, output_dir, networks):
    """Save comprehensive results summary for both permutation tests."""
    # Save permutation results
    perm_results_path = output_dir / f'{networks}net_k{k}_glhmm_results.pkl'
    with open(perm_results_path, 'wb') as f:
        pickle.dump(perm_results, f)
    
    # Save FO analysis results if available
    if fo_analysis is not None:
        fo_results_path = output_dir / f'{networks}net_k{k}_fo_glhmm_results.pkl'
        with open(fo_results_path, 'wb') as f:
            pickle.dump(fo_analysis, f)
    
    # Create text summary
    summary_path = output_dir / f'{networks}net_k{k}_glhmm_summary.txt'
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"GLHMM PERMUTATION TEST RESULTS: {networks}-NETWORK CONFIGURATION, k={k}\n")
        f.write("==============================================================\n\n")
        
        f.write(f"Number of permutations: {perm_results['n_permutations']}\n")
        f.write(f"Test date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # State transition results
        f.write("TRANSITION PROBABILITY RESULTS\n")
        f.write("---------------------------------\n\n")
        
        f.write("OBSERVED DIFFERENCES (Meditators - Controls):\n\n")
        
        # Write header
        f.write("From\\To ")
        for j in range(k):
            f.write(f"S{j+1:d}      ")
        f.write("\n")
        
        # Write difference matrix with significance markers
        for i in range(k):
            f.write(f"S{i+1:d}     ")
            for j in range(k):
                value = perm_results['observed_diff'][i, j]
                if perm_results['significant'][i, j]:
                    f.write(f"{value:+.3f}** ")
                elif perm_results['p_values'][i, j] < 0.05:
                    f.write(f"{value:+.3f}*  ")
                else:
                    f.write(f"{value:+.3f}   ")
            f.write("\n")
        
        f.write("\n* p<0.05 (uncorrected), ** p<0.05 (FDR-corrected)\n\n")
        
        # T-statistic matrix
        if 't_statistics' in perm_results:
            f.write("T-STATISTICS:\n\n")
            
            # Write header
            f.write("From\\To ")
            for j in range(k):
                f.write(f"S{j+1:d}      ")
            f.write("\n")
            
            # Write t-stat matrix
            for i in range(k):
                f.write(f"S{i+1:d}     ")
                for j in range(k):
                    value = perm_results['t_statistics'][i, j]
                    f.write(f"{value:+.3f}   ")
                f.write("\n")
            f.write("\n")
        
        # Write p-values
        f.write("UNCORRECTED P-VALUES:\n\n")
        
        # Write header
        f.write("From\\To ")
        for j in range(k):
            f.write(f"S{j+1:d}      ")
        f.write("\n")
        
        # Write p-value matrix
        for i in range(k):
            f.write(f"S{i+1:d}     ")
            for j in range(k):
                p = perm_results['p_values'][i, j]
                f.write(f"{p:.3f}   ")
            f.write("\n")
        
        f.write("\nFDR-CORRECTED P-VALUES:\n\n")
        
        # Write header
        f.write("From\\To ")
        for j in range(k):
            f.write(f"S{j+1:d}      ")
        f.write("\n")
        
        # Write corrected p-value matrix
        for i in range(k):
            f.write(f"S{i+1:d}     ")
            for j in range(k):
                p = perm_results['p_corrected'][i, j]
                f.write(f"{p:.3f}   ")
            f.write("\n")
        
        # Count significant transitions
        n_uncorrected = np.sum(perm_results['p_values'] < 0.05)
        n_corrected = np.sum(perm_results['significant'])
        
        f.write(f"\nSummary: {n_uncorrected} transitions significant at p<0.05 (uncorrected)\n")
        f.write(f"         {n_corrected} transitions significant after FDR correction\n\n")
        
        # Fractional occupancy results if available
        if fo_analysis is not None:
            f.write("\nFRACTIONAL OCCUPANCY RESULTS\n")
            f.write("---------------------------------\n\n")
            
            f.write("OBSERVED DIFFERENCES (Meditators - Controls):\n\n")
            
            # Write header
            f.write("State   Difference   p-value   p-FDR   Significant\n")
            f.write("--------------------------------------------------\n")
            
            # Write FO results
            for i in range(k):
                diff = fo_analysis['observed_diff'][i]
                p_val = fo_analysis['p_values'][i]
                p_fdr = fo_analysis['p_corrected'][i]
                sig = "Yes" if fo_analysis['significant'][i] else "No"
                
                if fo_analysis['significant'][i]:
                    f.write(f"S{i+1:d}      {diff:+.3f}      {p_val:.3f}    {p_fdr:.3f}    {sig}**\n")
                elif p_val < 0.05:
                    f.write(f"S{i+1:d}      {diff:+.3f}      {p_val:.3f}    {p_fdr:.3f}    {sig}*\n")
                else:
                    f.write(f"S{i+1:d}      {diff:+.3f}      {p_val:.3f}    {p_fdr:.3f}    {sig}\n")
            
            # Count significant FO differences
            n_fo_uncorrected = np.sum(fo_analysis['p_values'] < 0.05)
            n_fo_corrected = np.sum(fo_analysis['significant'])
            
            f.write(f"\nSummary: {n_fo_uncorrected} states show significant FO differences at p<0.05 (uncorrected)\n")
            f.write(f"         {n_fo_corrected} states show significant FO differences after FDR correction\n")
        
        # Write interpretation notes
        f.write("\nINTERPRETATION NOTES:\n")
        f.write("- For transitions: Positive values indicate higher succession probability in meditators\n")
        f.write("- For FO: Positive values indicate higher occupancy in meditators\n")
        f.write("- All p-values are two-tailed (testing for difference in either direction)\n")
        f.write("- FDR correction controls the false discovery rate across all comparisons\n")
    
    logger.info(f"Results summary saved to {summary_path}")
    return summary_path

def run_analysis(networks_list=[7, 8], k_values=[4, 5], n_permutations=5000):
    """Run complete GLHMM analysis pipeline."""
    logger.info(f"Starting GLHMM analysis pipeline with {n_permutations} permutations...")
    
    # Create output directory
    permutation_dir = TRANSITIONS_DIR / 'glhmm_analysis'
    permutation_dir.mkdir(exist_ok=True, parents=True)
    
    results_dirs = {}
    total_configs = len(networks_list) * len(k_values)
    config_count = 0
    
    # Process each network configuration
    for networks in networks_list:
        # Create network-specific output directory
        network_dir = permutation_dir / f'{networks}networks'
        network_dir.mkdir(exist_ok=True, parents=True)
        results_dirs[networks] = network_dir
        
        # Process each k value
        for k in k_values:
            config_count += 1
            logger.info(f"\n[{config_count}/{total_configs}] Processing {networks}-network, k={k} models...")
            
            try:
                # Load succession data
                succession_data = load_succession_data(networks, k)
                
                # Step 1: Transition probability analysis
                logger.info(f"Running transition probability analysis for {networks}-network, k={k}...")
                perm_results = permutation_test_glhmm(succession_data, n_permutations)
                
                # Step 2: Fractional occupancy analysis
                logger.info(f"Running fractional occupancy analysis for {networks}-network, k={k}...")
                fo_analysis = analyze_fractional_occupancy(succession_data, n_permutations)
                
                # Step 3: Save comprehensive results
                logger.info("Saving comprehensive results...")
                save_results_summary(perm_results, fo_analysis, k, network_dir, networks)
                
                logger.info(f"✓ Analysis for {networks}-network, k={k} completed successfully")
                
            except Exception as e:
                logger.error(f"✗ Error in analysis for {networks}-network, k={k}: {str(e)}")
                import traceback
                traceback.print_exc()
                
            # Progress update
            percent_complete = (config_count / total_configs) * 100
            logger.info(f"Progress: {percent_complete:.1f}% complete ({config_count}/{total_configs} configurations)")
    
    return results_dirs

def main():
    """Main function."""
    # Set up argument parser
    np.random.seed(42)  # This will make the permutation testing reproducible

    parser = argparse.ArgumentParser(description='GLHMM Permutation Testing for State Transition Analysis')
    parser.add_argument('--permutations', type=int, default=5000,
                        help='Number of permutations for statistical testing')
    parser.add_argument('--networks', type=int, nargs='+', default=[7, 8],
                        help='Network configurations to analyze (default: 7 8)')
    parser.add_argument('--k-values', type=int, nargs='+', default=[4, 5],
                        help='K values to analyze (default: 4 5)')
    args = parser.parse_args()
    
    logger.info("=== Starting GLHMM-Integrated State Transition Analysis ===")
    logger.info(f"Running with {args.permutations} permutations for networks {args.networks} with k values {args.k_values}")
    
    start_time = time.time()
    
    try:
        # Run analysis pipeline
        results_dirs = run_analysis(
            networks_list=args.networks,
            k_values=args.k_values,
            n_permutations=args.permutations
        )
        
        # Report completion
        elapsed_time = (time.time() - start_time) / 60
        logger.info(f"\n=== Complete analysis pipeline completed in {elapsed_time:.2f} minutes ===")
        
        # Print paths to result directories
        for networks, result_dir in results_dirs.items():
            logger.info(f"{networks}-network results: {result_dir}")
        
    except Exception as e:
        logger.error(f"Error in analysis pipeline: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()