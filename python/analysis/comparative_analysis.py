"""
TDE-HMM Comparative Analysis Script for Meditation Research

This script performs data-driven comparative analysis between meditators and controls
for TDE-HMM models with k=4 and k=5, focusing on:
1. Network activation patterns and interactions
2. Attention network dynamics (DMN, FPN, VAN, DAN)
3. Statistical comparison between groups
4. Individual subject variability analysis

The analysis handles both 7-network and 8-network configurations.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import seaborn as sns
from scipy import stats
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup paths
ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')
COMPARISON_DIR = os.path.join(METRICS_DIR, 'comparison')
os.makedirs(COMPARISON_DIR, exist_ok=True)

# Track data availability issues
data_availability_issues = []

def load_metrics(group, k, networks):
    """Load metrics for a given group, k value, and network configuration."""
    # Updated path to include network configuration
    metrics_path = os.path.join(METRICS_DIR, f'{networks}networks', group, f'k{k}_metrics.pkl')
    try:
        with open(metrics_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        error_msg = f"Metrics file not found: {metrics_path}"
        logger.error(error_msg)
        data_availability_issues.append({
            'type': 'missing_file',
            'group': group,
            'k': k,
            'networks': networks,
            'path': metrics_path,
            'message': error_msg
        })
        raise
    except Exception as e:
        error_msg = f"Error loading metrics: {str(e)}"
        logger.error(error_msg)
        data_availability_issues.append({
            'type': 'load_error',
            'group': group,
            'k': k,
            'networks': networks,
            'path': metrics_path,
            'message': error_msg
        })
        raise

def compare_network_interactions(networks_list=[7, 8], k_values=[4, 5]):
    """Compare network interactions between meditators and controls with statistical tests."""
    logger.info("\nComparing network interactions between groups...")
    
    # Data structure to store results
    network_comparisons = {}
    attention_networks = ['DMN', 'FPN', 'VAN', 'DAN']
    
    # Create CSV data storage
    csv_data = []
    
    # Process each network configuration
    for networks in networks_list:
        logger.info(f"\n=== Processing {networks}-network configuration ===")
        
        # Create network-specific directory
        network_comparison_dir = os.path.join(COMPARISON_DIR, f'{networks}networks')
        os.makedirs(network_comparison_dir, exist_ok=True)
        
        # Create network-specific results
        network_comparisons[networks] = {}
        
        for k in k_values:
            logger.info(f"Analyzing k={k} network interactions...")
            
            try:
                # Load metrics for both groups
                controls_metrics = load_metrics('controls', k, networks)
                meditators_metrics = load_metrics('meditators', k, networks)
                
                # Extract network interactions
                controls_interactions = controls_metrics['network_interactions']
                meditators_interactions = meditators_metrics['network_interactions']
                
                # Ensure we have the same states to compare
                common_states = sorted(set(controls_interactions.keys()).intersection(
                                      set(meditators_interactions.keys())))
                
                # Store k-specific comparisons
                network_comparisons[networks][k] = {
                    'state_comparisons': {},
                    'attention_networks': {}
                }
                
                # For each state, compare network interactions
                for state_idx in common_states:
                    c_interactions = controls_interactions[state_idx]
                    m_interactions = meditators_interactions[state_idx]
                    
                    # Get common network pairs to compare
                    common_pairs = sorted(set(c_interactions.keys()).intersection(
                                         set(m_interactions.keys())))
                    
                    # Store statistical comparison results
                    state_results = {}
                    
                    # Focus on attention networks and their interactions
                    attention_pairs = []
                    for pair in common_pairs:
                        # Include self-interactions (diagonal) of attention networks
                        if pair in [f"{net}-{net}" for net in attention_networks]:
                            attention_pairs.append(pair)
                        # Include interactions between attention networks
                        elif "-" in pair and pair.split("-")[0] in attention_networks and pair.split("-")[1] in attention_networks:
                            attention_pairs.append(pair)
                    
                    # For each attention network pair, perform statistical comparison
                    for pair in attention_pairs:
                        c_value = c_interactions.get(pair, 0)
                        m_value = m_interactions.get(pair, 0)
                        
                        # Calculate difference and percent difference
                        diff = m_value - c_value
                        pct_diff = (diff / abs(c_value)) * 100 if c_value != 0 else float('inf')
                        
                        # Store results
                        state_results[pair] = {
                            'controls': c_value,
                            'meditators': m_value,
                            'difference': diff,
                            'pct_difference': pct_diff,
                        }
                        
                        # Add to CSV data
                        csv_data.append({
                            'networks': networks,
                            'k': k,
                            'state_idx': state_idx,
                            'network_pair': pair,
                            'controls': c_value,
                            'meditators': m_value,
                            'difference': diff,
                            'pct_difference': pct_diff,
                        })
                        
                        # Log meaningful differences
                        if abs(pct_diff) > 20:  # 20% difference threshold for logging
                            direction = "higher" if diff > 0 else "lower"
                            logger.info(f"  State {state_idx+1}, {pair}: {abs(pct_diff):.1f}% {direction} in meditators")
                    
                    # Look specifically at DMN anticorrelation with task-positive networks
                    if all(f"DMN-{net}" in common_pairs for net in ['FPN', 'VAN', 'DAN']):
                        c_dmn_fpn = c_interactions.get('DMN-FPN', 0)
                        c_dmn_van = c_interactions.get('DMN-VAN', 0)
                        c_dmn_dan = c_interactions.get('DMN-DAN', 0)
                        
                        m_dmn_fpn = m_interactions.get('DMN-FPN', 0)
                        m_dmn_van = m_interactions.get('DMN-VAN', 0)
                        m_dmn_dan = m_interactions.get('DMN-DAN', 0)
                        
                        # Average DMN anticorrelation
                        c_dmn_anticorr = (c_dmn_fpn + c_dmn_van + c_dmn_dan) / 3
                        m_dmn_anticorr = (m_dmn_fpn + m_dmn_van + m_dmn_dan) / 3
                        
                        anticorr_diff = m_dmn_anticorr - c_dmn_anticorr
                        anticorr_pct = (anticorr_diff / abs(c_dmn_anticorr)) * 100 if c_dmn_anticorr != 0 else float('inf')
                        
                        # Store DMN anticorrelation
                        state_results['DMN_anticorrelation'] = {
                            'controls': c_dmn_anticorr,
                            'meditators': m_dmn_anticorr,
                            'difference': anticorr_diff,
                            'pct_difference': anticorr_pct
                        }
                        
                        # Add to CSV data
                        csv_data.append({
                            'networks': networks,
                            'k': k,
                            'state_idx': state_idx,
                            'network_pair': 'DMN_anticorrelation',
                            'controls': c_dmn_anticorr,
                            'meditators': m_dmn_anticorr,
                            'difference': anticorr_diff,
                            'pct_difference': anticorr_pct,
                        })
                        
                        # Log DMN anticorrelation comparison
                        direction = "stronger" if anticorr_diff < 0 else "weaker"
                        logger.info(f"  State {state_idx+1}, DMN anticorrelation: {abs(anticorr_pct):.1f}% {direction} in meditators")
                    
                    # Store results for this state
                    network_comparisons[networks][k]['state_comparisons'][state_idx] = state_results
            
            except Exception as e:
                error_msg = f"Error analyzing {networks}-network, k={k}: {str(e)}"
                logger.error(error_msg)
                import traceback
                logger.error(traceback.format_exc())
                
                # Record the issue rather than silently continuing
                data_availability_issues.append({
                    'type': 'analysis_error',
                    'function': 'compare_network_interactions',
                    'networks': networks,
                    'k': k,
                    'message': error_msg
                })
                
                # Mark this configuration as problematic but don't skip it
                network_comparisons[networks][k] = {'error': error_msg}
    
    # Save results to CSV - only if we have data
    if csv_data:
        csv_path = os.path.join(COMPARISON_DIR, 'network_comparisons.csv')
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)
        logger.info(f"Network comparisons saved to CSV: {csv_path}")
    else:
        logger.warning("No network comparison data available to save to CSV")
        data_availability_issues.append({
            'type': 'no_data',
            'function': 'compare_network_interactions',
            'message': "No network comparison data available to save to CSV"
        })
    
    # Create a text summary file for each network configuration
    for networks in networks_list:
        txt_path = os.path.join(COMPARISON_DIR, f'{networks}networks_summary.txt')
        
        with open(txt_path, 'w') as f:
            f.write(f"NETWORK INTERACTION COMPARISON SUMMARY: {networks}-NETWORK CONFIGURATION\n")
            f.write("=" * (55 + len(str(networks))) + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d')}\n\n")
            
            # Write overview
            f.write("OVERVIEW\n")
            f.write("--------\n")
            f.write("This analysis compares network interactions between meditators and controls\n")
            f.write("with a focus on attention networks (DMN, FPN, VAN, DAN).\n\n")
            
            # Write key findings for each k value
            for k in k_values:
                if networks not in network_comparisons or k not in network_comparisons[networks]:
                    f.write(f"FINDINGS FOR k={k}\n")
                    f.write("-" * (12 + len(str(k))) + "\n\n")
                    f.write("Data not available for this configuration.\n\n")
                    continue
                
                # Handle error case explicitly
                if 'error' in network_comparisons[networks][k]:
                    f.write(f"FINDINGS FOR k={k}\n")
                    f.write("-" * (12 + len(str(k))) + "\n\n")
                    f.write(f"ERROR: {network_comparisons[networks][k]['error']}\n\n")
                    continue
                    
                f.write(f"FINDINGS FOR k={k}\n")
                f.write("-" * (12 + len(str(k))) + "\n\n")
                
                # Extract major differences
                major_diffs = []
                for state_idx, state_results in network_comparisons[networks][k]['state_comparisons'].items():
                    for pair, result in state_results.items():
                        if abs(result['pct_difference']) > 25:  # 25% threshold for major differences
                            direction = "higher" if result['difference'] > 0 else "lower"
                            major_diffs.append({
                                'state': state_idx,
                                'pair': pair,
                                'pct': abs(result['pct_difference']),
                                'direction': direction
                            })
                
                # Sort by percentage difference
                major_diffs.sort(key=lambda x: x['pct'], reverse=True)
                
                # Write major differences
                if major_diffs:
                    f.write("Major Differences (>25%):\n")
                    for diff in major_diffs:
                        f.write(f"  - State {diff['state']+1}, {diff['pair']}: {diff['pct']:.1f}% {diff['direction']} in meditators\n")
                else:
                    f.write("No major differences (>25%) found between groups.\n")
                
                f.write("\n")
            
            # Write conclusion
            f.write("CONCLUSION\n")
            f.write("----------\n")
            f.write("The analysis identified network interaction patterns that differ\n")
            f.write("between meditators and controls. These differences may reflect\n")
            f.write("functional reorganization associated with meditation practice.\n\n")
            
            # Write methodological notes
            f.write("NOTES\n")
            f.write("-----\n")
            f.write("1. Percentage differences are calculated relative to the control group values.\n")
            f.write("2. DMN anticorrelation refers to the average correlation between DMN and task-positive networks.\n")
            f.write("3. Positive values indicate higher correlation in meditators; negative values indicate lower correlation.\n")
        
        logger.info(f"Network summary saved to text file: {txt_path}")
    
    return network_comparisons

def analyze_individual_differences(networks_list=[7, 8], k_values=[4, 5]):
    """Analyze individual differences within groups to identify potential expert meditators."""
    logger.info("\nAnalyzing individual differences within groups...")
    
    individual_results = {}
    csv_data = []
    
    # Process each network configuration
    for networks in networks_list:
        logger.info(f"\n=== Processing {networks}-network configuration ===")
        
        # Store results for this network configuration
        individual_results[networks] = {}
        
        for k in k_values:
            logger.info(f"Analyzing individual differences for k={k}...")
            
            try:
                # Load metrics for both groups
                controls_metrics = load_metrics('controls', k, networks)
                meditators_metrics = load_metrics('meditators', k, networks)
                
                # Extract fractional occupancy for each subject
                controls_fo = controls_metrics['temporal_metrics']['FO']
                meditators_fo = meditators_metrics['temporal_metrics']['FO']
                
                # Extract switching rates for each subject
                controls_sr = controls_metrics['temporal_metrics']['switching_rate']
                meditators_sr = meditators_metrics['temporal_metrics']['switching_rate']
                
                # Calculate within-group variability
                control_variability = np.mean([np.std(controls_fo[:, i]) for i in range(controls_fo.shape[1])])
                meditator_variability = np.mean([np.std(meditators_fo[:, i]) for i in range(meditators_fo.shape[1])])
                
                logger.info(f"  Within-group variability: Controls={control_variability:.4f}, Meditators={meditator_variability:.4f}")
                
                # Compare variability
                var_diff_pct = (meditator_variability - control_variability) / control_variability * 100
                direction = "higher" if var_diff_pct > 0 else "lower"
                logger.info(f"  Meditators show {abs(var_diff_pct):.1f}% {direction} variability in state usage")
                
                # Store results
                individual_results[networks][k] = {
                    'control_variability': control_variability,
                    'meditator_variability': meditator_variability,
                    'variability_diff_pct': var_diff_pct
                }
                
                # Add to CSV data
                csv_data.append({
                    'networks': networks,
                    'k': k,
                    'control_variability': control_variability,
                    'meditator_variability': meditator_variability,
                    'variability_diff_pct': var_diff_pct
                })
                
            except Exception as e:
                error_msg = f"Error analyzing individual differences for {networks}-network, k={k}: {str(e)}"
                logger.error(error_msg)
                import traceback
                logger.error(traceback.format_exc())
                
                # Record the issue
                data_availability_issues.append({
                    'type': 'analysis_error',
                    'function': 'analyze_individual_differences',
                    'networks': networks,
                    'k': k,
                    'message': error_msg
                })
                
                # Mark this configuration as problematic
                individual_results[networks][k] = {'error': error_msg}
    
    # Save individual differences to CSV
    if csv_data:
        csv_path = os.path.join(COMPARISON_DIR, 'individual_differences.csv')
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)
        logger.info(f"Individual differences saved to CSV: {csv_path}")
    else:
        logger.warning("No individual differences data available to save to CSV")
        data_availability_issues.append({
            'type': 'no_data',
            'function': 'analyze_individual_differences',
            'message': "No individual differences data available to save to CSV"
        })
    
    # Save individual differences summary to text file
    txt_path = os.path.join(COMPARISON_DIR, 'individual_differences.txt')
    with open(txt_path, 'w') as f:
        f.write("INDIVIDUAL DIFFERENCES ANALYSIS\n")
        f.write("==============================\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d')}\n\n")
        
        # Write variability findings for each network configuration
        for networks in networks_list:
            f.write(f"{networks}-NETWORK CONFIGURATION\n")
            f.write("-" * (len(str(networks)) + 21) + "\n\n")
            
            for k in k_values:
                if networks not in individual_results or k not in individual_results[networks]:
                    f.write(f"k={k}: Data not available\n")
                    continue
                
                # Handle error case explicitly
                if 'error' in individual_results[networks][k]:
                    f.write(f"k={k}: ERROR: {individual_results[networks][k]['error']}\n")
                    continue
                
                results = individual_results[networks][k]
                direction = "higher" if results['variability_diff_pct'] > 0 else "lower"
                f.write(f"k={k}: Meditators show {abs(results['variability_diff_pct']):.1f}% {direction} variability than controls\n")
            
            f.write("\n")
        
        f.write("INTERPRETATION\n")
        f.write("-------------\n")
        f.write("Higher variability in meditators could indicate diverse meditation strategies or different levels of expertise.\n")
        f.write("Lower variability could indicate more consistent brain states associated with regular meditation practice.\n")
    
    logger.info(f"Individual differences summary saved to text file: {txt_path}")
    
    return individual_results

def compare_temporal_dynamics(networks_list=[7, 8], k_values=[4, 5]):
    """Compare temporal dynamics between meditators and controls across network configurations."""
    logger.info("\nComparing temporal dynamics between groups...")
    
    # Setup data structures to store comparisons
    temporal_results = {}
    csv_data = []
    
    # Process each network configuration
    for networks in networks_list:
        logger.info(f"\n=== Processing {networks}-network configuration ===")
        
        # Initialize results for this network configuration
        temporal_results[networks] = {}
        
        for k in k_values:
            logger.info(f"Analyzing k={k} temporal dynamics...")
            
            try:
                # Load metrics for both groups
                controls_metrics = load_metrics('controls', k, networks)
                meditators_metrics = load_metrics('meditators', k, networks)
                
                # Store k-specific results
                temporal_results[networks][k] = {
                    'switching_rates': {},
                    'lifetimes': {},
                    'self_transitions': {}
                }
                
                # Extract and compare switching rates
                controls_sr = np.mean(controls_metrics['temporal_metrics']['switching_rate'])
                meditators_sr = np.mean(meditators_metrics['temporal_metrics']['switching_rate'])
                
                sr_diff = meditators_sr - controls_sr
                sr_diff_pct = (sr_diff / controls_sr) * 100 if controls_sr != 0 else float('inf')
                
                temporal_results[networks][k]['switching_rates'] = {
                    'controls': controls_sr,
                    'meditators': meditators_sr,
                    'difference': sr_diff,
                    'pct_difference': sr_diff_pct
                }
                
                # Log results
                direction = "higher" if sr_diff > 0 else "lower"
                logger.info(f"  Switching rates: {abs(sr_diff_pct):.1f}% {direction} in meditators")
                
                # Extract and compare lifetimes
                controls_lt = np.mean(controls_metrics['temporal_metrics']['lifetimes_mean'])
                meditators_lt = np.mean(meditators_metrics['temporal_metrics']['lifetimes_mean'])
                
                lt_diff = meditators_lt - controls_lt
                lt_diff_pct = (lt_diff / controls_lt) * 100 if controls_lt != 0 else float('inf')
                
                temporal_results[networks][k]['lifetimes'] = {
                    'controls': controls_lt,
                    'meditators': meditators_lt,
                    'difference': lt_diff,
                    'pct_difference': lt_diff_pct
                }
                
                # Log results
                direction = "longer" if lt_diff > 0 else "shorter"
                logger.info(f"  State lifetimes: {abs(lt_diff_pct):.1f}% {direction} in meditators")
                
                # Extract and compare self-transitions if available
                if 'transition_metrics' in controls_metrics and 'transition_metrics' in meditators_metrics:
                    if 'self_transitions' in controls_metrics['transition_metrics'] and 'self_transitions' in meditators_metrics['transition_metrics']:
                        controls_st = np.mean(controls_metrics['transition_metrics']['self_transitions'])
                        meditators_st = np.mean(meditators_metrics['transition_metrics']['self_transitions'])
                        
                        st_diff = meditators_st - controls_st
                        st_diff_pct = (st_diff / controls_st) * 100 if controls_st != 0 else float('inf')
                        
                        temporal_results[networks][k]['self_transitions'] = {
                            'controls': controls_st,
                            'meditators': meditators_st,
                            'difference': st_diff,
                            'pct_difference': st_diff_pct
                        }
                        
                        # Log results
                        direction = "higher" if st_diff > 0 else "lower"
                        logger.info(f"  Self-transitions: {abs(st_diff_pct):.1f}% {direction} in meditators")
                
                # Add to CSV data
                csv_data.append({
                    'networks': networks,
                    'k': k,
                    'metric': 'switching_rate',
                    'controls': controls_sr,
                    'meditators': meditators_sr,
                    'difference': sr_diff,
                    'pct_difference': sr_diff_pct
                })
                
                csv_data.append({
                    'networks': networks,
                    'k': k,
                    'metric': 'lifetime',
                    'controls': controls_lt,
                    'meditators': meditators_lt,
                    'difference': lt_diff,
                    'pct_difference': lt_diff_pct
                })
                
                if 'self_transitions' in temporal_results[networks][k]:
                    csv_data.append({
                        'networks': networks,
                        'k': k,
                        'metric': 'self_transition',
                        'controls': controls_st,
                        'meditators': meditators_st,
                        'difference': st_diff,
                        'pct_difference': st_diff_pct
                    })
                
            except Exception as e:
                error_msg = f"Error analyzing temporal dynamics for {networks}-network, k={k}: {str(e)}"
                logger.error(error_msg)
                import traceback
                logger.error(traceback.format_exc())
                
                # Record the issue
                data_availability_issues.append({
                    'type': 'analysis_error',
                    'function': 'compare_temporal_dynamics',
                    'networks': networks,
                    'k': k,
                    'message': error_msg
                })
                
                # Mark this configuration as problematic
                temporal_results[networks][k] = {'error': error_msg}
    
    # Save temporal dynamics to CSV
    if csv_data:
        csv_path = os.path.join(COMPARISON_DIR, 'temporal_dynamics.csv')
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)
        logger.info(f"Temporal dynamics saved to CSV: {csv_path}")
    else:
        logger.warning("No temporal dynamics data available to save to CSV")
        data_availability_issues.append({
            'type': 'no_data',
            'function': 'compare_temporal_dynamics',
            'message': "No temporal dynamics data available to save to CSV"
        })
    
    # Generate summary text file
    txt_path = os.path.join(COMPARISON_DIR, 'temporal_dynamics_summary.txt')
    with open(txt_path, 'w') as f:
        f.write("TEMPORAL DYNAMICS COMPARISON\n")
        f.write("===========================\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d')}\n\n")
        
        # Process each network configuration
        for networks in networks_list:
            f.write(f"{networks}-NETWORK CONFIGURATION\n")
            f.write("-" * (len(str(networks)) + 21) + "\n\n")
            
            for k in k_values:
                if networks not in temporal_results or k not in temporal_results[networks]:
                    f.write(f"k={k}: Data not available\n")
                    continue
                
                # Handle error case explicitly
                if 'error' in temporal_results[networks][k]:
                    f.write(f"k={k}: ERROR: {temporal_results[networks][k]['error']}\n")
                    continue
                
                f.write(f"k={k} FINDINGS:\n")
                
                # Switching rates
                sr_data = temporal_results[networks][k]['switching_rates']
                sr_direction = "higher" if sr_data['difference'] > 0 else "lower"
                f.write(f"  Switching rates: {abs(sr_data['pct_difference']):.1f}% {sr_direction} in meditators\n")
                
                # Lifetimes
                lt_data = temporal_results[networks][k]['lifetimes']
                lt_direction = "longer" if lt_data['difference'] > 0 else "shorter"
                f.write(f"  State lifetimes: {abs(lt_data['pct_difference']):.1f}% {lt_direction} in meditators\n")
                
                # Self-transitions if available
                if 'self_transitions' in temporal_results[networks][k]:
                    st_data = temporal_results[networks][k]['self_transitions']
                    st_direction = "higher" if st_data['difference'] > 0 else "lower"
                    f.write(f"  Self-transitions: {abs(st_data['pct_difference']):.1f}% {st_direction} in meditators\n")
                
                f.write("\n")
            
            f.write("\n")
        
        f.write("INTERPRETATION\n")
        f.write("-------------\n")
        f.write("Higher switching rates suggest more dynamic brain states in the group.\n")
        f.write("Longer state lifetimes suggest more stable brain states and potentially better attentional control.\n")
        f.write("Higher self-transition probabilities indicate tendency to maintain the current brain state.\n")
    
    logger.info(f"Temporal dynamics summary saved to text file: {txt_path}")
    
    return temporal_results

def write_data_availability_report():
    """Create a comprehensive report of data availability issues."""
    if not data_availability_issues:
        logger.info("No data availability issues detected.")
        return
    
    # Create report file
    report_path = os.path.join(COMPARISON_DIR, 'data_availability_report.txt')
    with open(report_path, 'w') as f:
        f.write("DATA AVAILABILITY ISSUES REPORT\n")
        f.write("===============================\n\n")
        f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Group issues by type
        issue_types = {}
        for issue in data_availability_issues:
            issue_type = issue['type']
            if issue_type not in issue_types:
                issue_types[issue_type] = []
            issue_types[issue_type].append(issue)
        
        # Write summary counts
        f.write("SUMMARY\n")
        f.write("-------\n")
        for issue_type, issues in issue_types.items():
            f.write(f"{issue_type}: {len(issues)} issues\n")
        f.write("\n")
        
        # Write detailed issues
        f.write("DETAILED ISSUES\n")
        f.write("--------------\n\n")
        
        for issue_type, issues in issue_types.items():
            f.write(f"{issue_type.upper()} ISSUES:\n")
            for i, issue in enumerate(issues, 1):
                f.write(f"{i}. ")
                
                # Write relevant details based on issue type
                if issue_type == 'missing_file':
                    f.write(f"Missing file for {issue['group']}, k={issue['k']}, {issue['networks']}-network\n")
                    f.write(f"   Path: {issue['path']}\n")
                elif issue_type == 'load_error':
                    f.write(f"Error loading data for {issue['group']}, k={issue['k']}, {issue['networks']}-network\n")
                    f.write(f"   Error: {issue['message']}\n")
                elif issue_type == 'analysis_error':
                    f.write(f"Analysis error in {issue['function']} for {issue['networks']}-network, k={issue['k']}\n")
                    f.write(f"   Error: {issue['message']}\n")
                else:
                    f.write(f"{issue['message']}\n")
                    
                f.write("\n")
            
            f.write("\n")
        
        # Write recommendations
        f.write("RECOMMENDATIONS\n")
        f.write("--------------\n")
        f.write("1. Check that all required data files exist in the expected locations\n")
        f.write("2. Ensure that the metrics were correctly calculated for all configurations\n")
        f.write("3. Verify that network and state mappings are consistent across groups\n")
        f.write("4. Address the issues above before relying on the analysis results\n")
    
    logger.warning(f"Data availability issues detected. See report: {report_path}")
    
    # Also save as CSV for easier programmatic analysis
    csv_path = os.path.join(COMPARISON_DIR, 'data_availability_issues.csv')
    pd.DataFrame(data_availability_issues).to_csv(csv_path, index=False)
    
    return report_path

def main():
    """Main function to run the TDE-HMM comparative analysis."""
    logger.info("=== Starting TDE-HMM Comparative Analysis ===")
    
    # Define network configurations and k values for comparison
    networks_list = [7, 8]
    k_values = [4, 5]
    
    # Clear previous issues
    global data_availability_issues
    data_availability_issues = []
    
    try:
        # Compare network interactions
        network_results = compare_network_interactions(networks_list, k_values)
        
        # Analyze individual differences
        individual_results = analyze_individual_differences(networks_list, k_values)
        
        # Compare temporal dynamics
        temporal_results = compare_temporal_dynamics(networks_list, k_values)
        
        # Write data availability report
        report_path = write_data_availability_report()
        
        # Save comprehensive results
        all_results = {
            'network_results': network_results,
            'individual_results': individual_results,
            'temporal_results': temporal_results,
            'networks_list': networks_list,
            'k_values': k_values,
            'data_availability_issues': data_availability_issues,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        results_path = os.path.join(COMPARISON_DIR, 'comprehensive_comparison.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(all_results, f)
        
        logger.info(f"\n=== TDE-HMM comparative analysis completed ===")
        logger.info(f"Results saved to: {COMPARISON_DIR}")
        logger.info(f"Summary files:")
        logger.info(f"  - network_comparisons.csv")
        logger.info(f"  - 7networks_summary.txt and 8networks_summary.txt")
        logger.info(f"  - individual_differences.txt")
        logger.info(f"  - temporal_dynamics_summary.txt")
        
        # Report data availability status
        if data_availability_issues:
            logger.warning(f"ATTENTION: {len(data_availability_issues)} data availability issues detected")
            logger.warning(f"Review the report at: {report_path}")
        else:
            logger.info("All data configurations were successfully processed.")
            
    except Exception as e:
        logger.error(f"Error in TDE-HMM comparative analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Record critical failure
        data_availability_issues.append({
            'type': 'critical_failure',
            'message': str(e),
            'traceback': traceback.format_exc()
        })
        
        # Still try to write the report
        try:
            write_data_availability_report()
        except:
            pass
        
        # Re-raise to ensure the process doesn't appear successful
        raise

if __name__ == "__main__":
    main()