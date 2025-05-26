"""
State Block Transition Analysis for Meditation Research

This script analyzes state block transitions from trained HMM models,
extracting continuous state blocks and transitions between states.
It focuses on extracting transition patterns and preparing data for
subsequent permutation testing (performed separately in state_permutation_testing.py).

Key functionality:
1. Extracts continuous state blocks from Viterbi paths
2. Calculates succession probabilities between states
3. Computes subject-level and group-level transition matrices
4. Stores transition data for downstream statistical analysis
5. Visualizes transition probability differences between groups
"""
import os
import argparse
import numpy as np
import pandas as pd
import pickle
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from glhmm import statistics, graphics, auxiliary, utils

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup paths and directories
ROOT_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = ROOT_DIR / 'data' / 'trained'
RESULTS_DIR = ROOT_DIR / 'results'
METRICS_DIR = RESULTS_DIR / 'metrics'
TRANSITIONS_DIR = METRICS_DIR / 'transitions'
os.makedirs(TRANSITIONS_DIR, exist_ok=True)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='State Block Transition Analysis')
    
    parser.add_argument('--networks', type=int, nargs='+', default=[7],
                        help='Network configurations to process (default: 7)')
    
    parser.add_argument('--k', type=int, default=4,
                        help='Number of states (default: 4)')
    
    return parser.parse_args()

def load_model(group, k, networks):
    """Load trained HMM model for a group."""
    model_dir = DATA_DIR / 'bygroup' / f'{networks}networks' / group / f'k{k}'
    model_path = model_dir / 'model.pkl'
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except Exception as e:
        logger.error(f"Error loading model for {group}: {str(e)}")
        return None

def preprocess_vpath(vpath, k):
    """
    Process the Viterbi path to handle 1-indexed states (1 to k) when code expects 0-indexed (0 to k-1).
    
    Args:
        vpath: The original Viterbi path
        k: Number of states (0-indexed expected, so k=4 means states 0,1,2,3)
        
    Returns:
        Processed Viterbi path with states adjusted to 0-indexed
    """
    # Convert vpath to 1D representation for consistent processing
    vpath_1d = statistics.generate_vpath_1D(vpath)
    
    # Find the actual range of states in the data
    unique_states = np.unique(vpath_1d)
    logger.info(f"Original unique states: {unique_states} (expect states to be 1-based: 1 to {k})")
    
    # If states are 1-indexed, shift them to 0-indexed
    if np.min(unique_states) == 1 and np.max(unique_states) == k:
        logger.info("Converting 1-indexed states (1 to k) to 0-indexed (0 to k-1)")
        vpath_1d = vpath_1d - 1
    
    # Ensure all states are within valid range (0 to k-1)
    invalid_mask = ~((0 <= vpath_1d) & (vpath_1d < k))
    invalid_count = np.sum(invalid_mask)
    
    if invalid_count > 0:
        logger.warning(f"Found {invalid_count} invalid states after preprocessing, clipping to valid range")
        vpath_1d = np.clip(vpath_1d, 0, k-1)
    
    return vpath_1d

def viterbi_path_to_stc(vpath, k):
    """
    Convert Viterbi path (1D vector of state assignments) to state time courses (STC).
    
    Args:
        vpath: 1D array with state assignments
        k: Number of states
        
    Returns:
        STC matrix (one-hot encoded states, shape: timepoints x states)
    """
    stc = np.zeros((len(vpath), k), dtype=np.int8)
    for i in range(k):
        stc[vpath == i, i] = 1
    return stc

def extract_state_blocks(vpath, indices, k):
    """
    Extract continuous state blocks from Viterbi paths.
    
    This implementation:
    1. Converts 1-indexed states to 0-indexed if needed
    2. Extracts continuous blocks of the same state
    3. Explicitly removes the first and last block for each subject
    4. Calculates transitions between the remaining blocks
    
    Args:
        vpath: Viterbi path data
        indices: Subject boundary indices
        k: Number of states
        
    Returns:
        subject_blocks: List of blocks per subject
        subject_successions: List of transitions per subject
    """
    # Preprocess vpath to ensure consistent state indexing
    vpath_1d = preprocess_vpath(vpath, k)
    
    # Process each subject separately
    subject_blocks = []
    subject_successions = []
    
    # Loop through each subject
    for i in range(len(indices)):
        start, end = indices[i]
        segment = vpath_1d[start:end].copy()
        
        # Convert to state time courses format expected by GLHMM
        segment_stc = viterbi_path_to_stc(segment, k)
        
        # Extract all blocks for this subject using GLHMM's utility function
        blocks = []
        for state in range(k):
            lengths, onsets = utils.get_visits(segment_stc, state)
            for length, onset in zip(lengths, onsets):
                blocks.append((state, start + onset, start + onset + int(length), int(length)))
        
        # Sort blocks by onset time
        blocks = sorted(blocks, key=lambda x: x[1])
        
        # Skip if not enough blocks for this subject
        if len(blocks) < 3:
            logger.warning(f"Subject {i+1} has only {len(blocks)} blocks (too few for transition analysis)")
            subject_blocks.append([])
            subject_successions.append([])
            continue
        
        # Remove first and last blocks as required by proper HMM block analysis
        filtered_blocks = blocks[1:-1]
        subject_blocks.append(filtered_blocks)
        
        # Extract transitions between blocks (after removing first/last)
        successions = []
        for j in range(len(filtered_blocks) - 1):
            curr_block = filtered_blocks[j]
            next_block = filtered_blocks[j+1]
            
            # Check if this is a valid transition (blocks are contiguous)
            if curr_block[2] == next_block[1]:
                curr_state = curr_block[0]
                next_state = next_block[0]
                successions.append((curr_state, next_state))
        
        subject_successions.append(successions)
    
    return subject_blocks, subject_successions

def analyze_transition_dynamics(results, networks, k):
    """Analyze transition dynamics using metrics from Vidaurre's papers."""
    logger.info("Analyzing state transition dynamics...")
    
    # Extract transition matrices
    med_trans = results['meditators']['transition_matrix']
    con_trans = results['controls']['transition_matrix']
    
    # 1. Calculate entropy of transitions (measure of predictability)
    # Higher entropy = more random transitions
    def calc_entropy(P):
        entropy = np.zeros(k)
        for i in range(k):
            p = P[i]
            # Avoid log(0)
            p_nonzero = p[p > 0]
            if len(p_nonzero) > 0:
                entropy[i] = -np.sum(p_nonzero * np.log2(p_nonzero))
        return entropy
    
    med_entropy = calc_entropy(med_trans)
    con_entropy = calc_entropy(con_trans)
    
    # 2. Calculate "outgoingness" (sum of off-diagonal transitions)
    # Higher = more likely to transition to different states
    def calc_outgoingness(P):
        outgoing = np.zeros(k)
        for i in range(k):
            # Sum all transitions except self-transitions
            outgoing[i] = np.sum(P[i]) - P[i,i]
        return outgoing
    
    med_outgoing = calc_outgoingness(med_trans)
    con_outgoing = calc_outgoingness(con_trans)
    
    # 3. Calculate "metastability" (tendency to switch between states)
    # Higher = more frequently switching states
    med_metastability = 1 - np.mean(np.diag(med_trans))
    con_metastability = 1 - np.mean(np.diag(con_trans))
    
    dynamics = {
        'meditators': {
            'entropy': med_entropy,
            'outgoingness': med_outgoing,
            'metastability': med_metastability
        },
        'controls': {
            'entropy': con_entropy,
            'outgoingness': con_outgoing,
            'metastability': con_metastability
        },
        'diff': {
            'entropy': med_entropy - con_entropy,
            'outgoingness': med_outgoing - con_outgoing,
            'metastability': med_metastability - con_metastability
        }
    }
    
    return dynamics

def calculate_succession_matrix(successions, k):
    """Calculate state succession matrix from succession list."""
    # Initialize count matrix
    counts = np.zeros((k, k))
    
    # All states are guaranteed to be valid from extract_state_blocks
    for from_state, to_state in successions:
        counts[from_state, to_state] += 1
    
    # Convert to probabilities
    row_sums = counts.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    probs = counts / row_sums
    
    return counts, probs

def compare_transition_matrices(med_model, con_model, k):
    """Compare transition matrices between meditator and control groups."""
    # Extract data we need
    med_vpath = med_model['vpath']
    med_indices = med_model['indices']
    con_vpath = con_model['vpath']
    con_indices = con_model['indices']
    
    # Extract blocks and transitions (per subject)
    med_blocks, med_successions = extract_state_blocks(med_vpath, med_indices, k)
    con_blocks, con_successions = extract_state_blocks(con_vpath, con_indices, k)
    
    # Flatten all successions for group-level matrices
    med_all_successions = [s for subj in med_successions for s in subj]
    con_all_successions = [s for subj in con_successions for s in subj]
    
    # Calculate group-level transition matrices
    med_counts, med_probs = calculate_succession_matrix(med_all_successions, k)
    con_counts, con_probs = calculate_succession_matrix(con_all_successions, k)
    
    # Calculate subject-level transition matrices
    med_subj_matrices = []
    for subj_succ in med_successions:
        if subj_succ:  # Only if there are transitions
            _, subj_probs = calculate_succession_matrix(subj_succ, k)
            med_subj_matrices.append(subj_probs)
        else:
            # Empty matrix if no transitions
            med_subj_matrices.append(np.zeros((k, k)))
    
    con_subj_matrices = []
    for subj_succ in con_successions:
        if subj_succ:
            _, subj_probs = calculate_succession_matrix(subj_succ, k)
            con_subj_matrices.append(subj_probs)
        else:
            con_subj_matrices.append(np.zeros((k, k)))
    
    # Compile results
    results = {
        'meditators': {
            'blocks': med_blocks,
            'successions': med_successions,
            'transition_matrix': med_probs,
            'subject_matrices': med_subj_matrices
        },
        'controls': {
            'blocks': con_blocks,
            'successions': con_successions, 
            'transition_matrix': con_probs,
            'subject_matrices': con_subj_matrices
        },
        'diff': med_probs - con_probs,
        'metadata': {
            'k': k,
            'n_meditators': len(med_indices),
            'n_controls': len(con_indices),
            'first_last_excluded': True,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    }
    
    return results

def analyze_transitions(networks, k):
    """Analyze transitions between meditators and controls."""
    logger.info(f"Analyzing transitions for {networks}-network, k={k} configuration")
    
    try:
        # Load models
        med_model = load_model('meditators', k, networks)
        con_model = load_model('controls', k, networks)
        
        if med_model is None or con_model is None:
            logger.error("Failed to load models")
            return None
        
        # Compare transitions
        results = compare_transition_matrices(med_model, con_model, k)
        
        # Create output directory if needed
        network_dir = TRANSITIONS_DIR / f'{networks}networks'
        os.makedirs(network_dir, exist_ok=True)
        
        # Save results
        out_path = network_dir / f'k{k}_succession_data.pkl'
        with open(out_path, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"Saved transition data to {out_path}")
        
        # Visualize transition differences
        visualize_transition_matrices(results, networks, k)
        
        return results
    
    except Exception as e:
        logger.error(f"Error in transition analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def visualize_transition_matrices(results, networks, k):
    """Visualize transition probability matrices using seaborn heatmaps."""
    # Create output directory
    vis_dir = TRANSITIONS_DIR / 'visualizations' / f'{networks}networks'
    os.makedirs(vis_dir, exist_ok=True)
    
    # Plot meditator transition matrix
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        results['meditators']['transition_matrix'],
        annot=True, 
        fmt=".2f", 
        cmap="viridis",
        xticklabels=range(1, k+1),
        yticklabels=range(1, k+1)
    )
    plt.title(f'Meditators: Transition Probabilities ({networks}-networks, k={k})')
    plt.xlabel('To State')
    plt.ylabel('From State')
    plt.tight_layout()
    plt.savefig(vis_dir / f'meditators_transitions_k{k}.png')
    plt.close()
    
    # Same for controls
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        results['controls']['transition_matrix'],
        annot=True, 
        fmt=".2f", 
        cmap="viridis",
        xticklabels=range(1, k+1),
        yticklabels=range(1, k+1)
    )
    plt.title(f'Controls: Transition Probabilities ({networks}-networks, k={k})')
    plt.xlabel('To State')
    plt.ylabel('From State')
    plt.tight_layout()
    plt.savefig(vis_dir / f'controls_transitions_k{k}.png')
    plt.close()
    
    # For difference matrix
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        results['diff'], 
        cmap='coolwarm', 
        center=0, 
        annot=True, 
        fmt=".2f", 
        xticklabels=range(1, k+1), 
        yticklabels=range(1, k+1)
    )
    plt.title(f'Transition Probability Differences (Meditators - Controls)\n{networks}-networks, k={k}')
    plt.xlabel('To State')
    plt.ylabel('From State')
    plt.tight_layout()
    plt.savefig(vis_dir / f'transition_diffs_k{k}.png')
    plt.close()
    
def visualize_transition_dynamics(results, networks, k, output_dir):
    """Create specialized transition dynamics visualizations."""
    # Create output directory
    vis_dir = output_dir / f'{networks}networks'
    os.makedirs(vis_dir, exist_ok=True)
    
    # Get data
    med_trans = results['meditators']['transition_matrix']
    con_trans = results['controls']['transition_matrix']
    
    # Transition dynamics plot
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # Self-transition rates
    self_trans_med = np.diag(med_trans)
    self_trans_con = np.diag(con_trans)
    
    # Plot 1: Self-transition (state stability)
    ax[0].bar(np.arange(k) - 0.2, self_trans_med, width=0.4, label='Meditators')
    ax[0].bar(np.arange(k) + 0.2, self_trans_con, width=0.4, label='Controls')
    ax[0].set_xlabel('State')
    ax[0].set_ylabel('Self-transition probability')
    ax[0].set_title('State Stability')
    ax[0].set_xticks(np.arange(k))
    ax[0].set_xticklabels([f'S{i+1}' for i in range(k)])
    ax[0].legend()
    
    # Plot 2 & 3: Original heatmaps
    sns.heatmap(med_trans, annot=True, fmt=".2f", cmap="viridis", ax=ax[1])
    ax[1].set_title('Meditators: Transitions')
    ax[1].set_xlabel('To State')
    ax[1].set_ylabel('From State')
    
    sns.heatmap(con_trans, annot=True, fmt=".2f", cmap="viridis", ax=ax[2])
    ax[2].set_title('Controls: Transitions')
    ax[2].set_xlabel('To State')
    ax[2].set_ylabel('From State')
    
    plt.tight_layout()
    plt.savefig(vis_dir / f'transition_dynamics_k{k}.png')
    plt.close()

def main():
    """Main function to run the analysis."""
    args = parse_arguments()
    
    logger.info("=== Starting State Block Transition Analysis ===")
    start_time = datetime.now()
    
    # Create required directories
    for networks in args.networks:
        os.makedirs(TRANSITIONS_DIR / f'{networks}networks', exist_ok=True)
        os.makedirs(TRANSITIONS_DIR / 'visualizations' / f'{networks}networks', exist_ok=True)
    
    # Process each network configuration
    for networks in args.networks:
        logger.info(f"\n=== Processing {networks}-network configuration with k={args.k} ===")
        results = analyze_transitions(networks, args.k)
        
        if results:
            # Add the new dynamics analysis
            dynamics = analyze_transition_dynamics(results, networks, args.k)
            
            # Add dynamics to results
            results['dynamics'] = dynamics
            
            # Save updated results
            network_dir = TRANSITIONS_DIR / f'{networks}networks'
            out_path = network_dir / f'k{args.k}_succession_data.pkl'
            with open(out_path, 'wb') as f:
                pickle.dump(results, f)
            
            # Create additional dynamics visualizations
            vis_dir = TRANSITIONS_DIR / 'visualizations' / f'{networks}networks'
            visualize_transition_dynamics(results, networks, args.k, vis_dir)
            
            logger.info(f"Successfully analyzed {networks}-network transitions")
        else:
            logger.error(f"Failed to analyze {networks}-network transitions")
    
    # Log completion
    elapsed = datetime.now() - start_time
    logger.info(f"=== Analysis completed in {elapsed.total_seconds():.1f} seconds ===")

if __name__ == "__main__":
    main()