"""
State Transition Analysis for Meditators in Anapanasati Meditation

Analyzes state successions for meditators (7-network, k=4, by-group standardization),
using transitions from second to second-to-last state to reduce entry state bias.
Loads activations from all_states_activation_patterns.csv for accurate state labels.

Features:
- Succession matrix computation
- Consecutive pattern detection (n_length=3)
- Visualization of succession matrix
- Summary report with transitions and patterns
"""

import os
import numpy as np
import pandas as pd
import pickle
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from glhmm import statistics

# Setup paths
ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')
TRANSITIONS_DIR = os.path.join(RESULTS_DIR, 'transitions')
VIS_DIR = os.path.join(TRANSITIONS_DIR, 'visualizations')
STATE_PATTERNS_DIR = r'G:\leida_hmm\python\results\state_patterns'  # Explicit CSV path

# Create directories
os.makedirs(TRANSITIONS_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
K = 4  # Fixed k=4
NETWORK_CONFIGS = [7]  # Only 7-network
STD_METHOD = 'bygroup'

def load_model(group, k, networks, std_method=STD_METHOD):
    """Load trained HMM model."""
    model_dir = os.path.join(DATA_DIR, 'trained', std_method, f'{networks}networks', group, f'k{k}')
    model_path = os.path.join(model_dir, 'model.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    if 'k' not in model_data:
        model_data['k'] = k
        
    logger.info(f"Loaded {group} k={k} model for {networks}-network ({std_method})")
    return model_data

def extract_state_blocks(vpath, indices, k):
    """Extract continuous state blocks from Viterbi path."""
    if len(vpath.shape) > 1 and vpath.shape[1] > 1:
        vpath_1d = statistics.generate_vpath_1D(vpath)
    else:
        vpath_1d = vpath
    
    all_blocks = []
    all_successions = []
    
    for i, (start, end) in enumerate(indices):
        subject_states = vpath_1d[start:end]
        
        blocks = []
        current_state = subject_states[0]
        block_start = 0
        is_first_block = True
        
        for t in range(1, len(subject_states)):
            if subject_states[t] != current_state or t == len(subject_states) - 1:
                block_end = t-1 if t < len(subject_states) - 1 else t
                
                if current_state < k:
                    blocks.append({
                        'state': int(current_state),
                        'start': block_start,
                        'end': block_end,
                        'length': block_end - block_start + 1,
                        'subject_idx': i,
                        'is_first_block': is_first_block
                    })
                    is_first_block = False
                
                current_state = subject_states[t]
                block_start = t
        
        successions = []
        for b in range(len(blocks)-1):
            from_state = blocks[b]['state']
            to_state = blocks[b+1]['state']
            successions.append({
                'from_state': from_state,
                'to_state': to_state,
                'from_length': blocks[b]['length'],
                'to_length': blocks[b+1]['length'],
                'subject_idx': i,
                'is_from_first_block': blocks[b]['is_first_block']
            })
        
        all_blocks.extend(blocks)
        all_successions.extend(successions)
    
    return all_blocks, all_successions

def calculate_succession_matrix(successions, k=4, exclude_first_blocks=True):
    """Compute succession matrix for k=4 states, normalizing valid transitions."""
    succession_counts = np.zeros((k, k))
    valid_states = set()
    valid_transitions = 0
    
    for succession in successions:
        if exclude_first_blocks and succession.get('is_from_first_block', False):
            continue
            
        from_state = succession['from_state']
        to_state = succession['to_state']
        
        if from_state >= k or to_state >= k:
            logger.warning(f"Invalid state transition: {from_state} → {to_state}")
            continue
        
        valid_states.add(from_state)
        valid_states.add(to_state)
        
        succession_counts[from_state, to_state] += 1
        valid_transitions += 1
    
    if valid_transitions < k:
        logger.warning(f"Insufficient valid transitions: {valid_transitions} found")
    
    succession_probs = np.zeros_like(succession_counts)
    for s in range(k):
        row_sum = np.sum(succession_counts[s, :])
        if row_sum > 0:
            succession_probs[s, :] = succession_counts[s, :] / row_sum
    
    return succession_counts, succession_probs

def detect_state_patterns(successions, k=4, n_length=3):
    """Detect consecutive state patterns for k=4 states, starting from second state."""
    subject_successions = {}
    for succession in successions:
        subject_idx = succession['subject_idx']
        if subject_idx not in subject_successions:
            subject_successions[subject_idx] = []
        subject_successions[subject_idx].append(succession)
    
    subject_sequences = {}
    state_patterns = {}
    
    for subject_idx, transitions in subject_successions.items():
        sequence = []
        valid_transitions = 0
        skip_first = True  # Skip first transition
        for succ in transitions:
            if succ.get('is_from_first_block', False):
                continue
            from_state = succ['from_state']
            to_state = succ['to_state']
            if from_state >= k or to_state >= k:
                logger.warning(f"Invalid state in subject {subject_idx}: {from_state} → {to_state}")
                continue
            if skip_first:
                skip_first = False
                sequence.append(from_state)  # Start with second state
                continue
            sequence.append(to_state)
            valid_transitions += 1
        
        # Exclude last transition
        if len(sequence) > 1:
            sequence = sequence[:-1]
        
        if len(sequence) < n_length:
            logger.warning(f"Subject {subject_idx} has sequence too short: {len(sequence)} states")
            continue
        if valid_transitions < 1:
            logger.warning(f"Subject {subject_idx} has no valid transitions")
            continue
        
        subject_sequences[subject_idx] = sequence
        possible_patterns = len(sequence) - n_length + 1
        
        for i in range(possible_patterns):
            pattern = tuple(sequence[i:i+n_length])
            if pattern not in state_patterns:
                state_patterns[pattern] = {
                    'count': 1,
                    'subjects': {subject_idx},
                    'sequence_length': len(sequence)
                }
            else:
                state_patterns[pattern]['count'] += 1
                state_patterns[pattern]['subjects'].add(subject_idx)
                state_patterns[pattern]['sequence_length'] += len(sequence)
    
    pattern_frequencies = {}
    for pattern, data in state_patterns.items():
        n_subjects = len(data['subjects'])
        frequency = data['count'] / max(1, data['sequence_length'])
        
        pattern_frequencies[pattern] = {
            'count': data['count'],
            'n_subjects': n_subjects,
            'frequency': frequency,
            'subjects': data['subjects']
        }
    
    return pattern_frequencies

def visualize_succession_matrices(results):
    """Visualize meditator succession matrix."""
    networks = results['networks']
    k = results['k']
    med_probs = results['succession']['meditators']['probs']
    med_state_info = results['state_info']['meditators']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(med_probs, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=[f"S{i} ({med_state_info[i]['dominant_network']})" for i in range(k)],
                yticklabels=[f"S{i} ({med_state_info[i]['dominant_network']})" for i in range(k)])
    plt.title(f"Meditator Succession Matrix ({networks}-network, k={k})")
    plt.xlabel("To State")
    plt.ylabel("From State")
    save_path = os.path.join(VIS_DIR, f'meditator_succession_matrix_{networks}networks_k{k}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved succession matrix visualization to {save_path}")

def write_summary_report(results):
    """Create a report summarizing meditator transitions and patterns."""
    networks = results['networks']
    k = results['k']
    med_probs = results['succession']['meditators']['probs']
    med_state_info = results['state_info']['meditators']
    
    report_path = os.path.join(TRANSITIONS_DIR, f'meditator_transition_report_{networks}networks_k{k}.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"MEDITATOR STATE TRANSITION ANALYSIS REPORT\n")
        f.write(f"====================================\n")
        f.write(f"Network configuration: {networks}-network\n")
        f.write(f"Number of states (k): {k}\n\n")
        
        f.write(f"STATE DEFINITIONS\n")
        f.write(f"----------------\n")
        for s in range(k):
            network = med_state_info[s]['dominant_network'].split('_')[0] if s in med_state_info else "Unknown"
            f.write(f"State {s}: Dominant Network = {network}, FO = {med_state_info[s]['fractional_occupancy']:.3f}\n")
        f.write("\n")
        
        f.write(f"KEY TRANSITIONS\n")
        f.write(f"--------------\n")
        flat_probs = med_probs.flatten()
        top_indices = np.argsort(flat_probs)[-5:]
        
        f.write("Top transitions for MEDITATORS:\n")
        for idx in reversed(top_indices):
            i, j = idx // k, idx % k
            network_i = med_state_info[i]['dominant_network'].split('_')[0] if i in med_state_info else "Unknown"
            network_j = med_state_info[j]['dominant_network'].split('_')[0] if j in med_state_info else "Unknown"
            f.write(f"  {i} → {j}: {med_probs[i, j]:.3f} ({network_i} → {network_j})\n")
        
        f.write("\nRECURRING STATE PATTERNS\n")
        f.write("=====================\n\n")

        if 'patterns' in results and results['patterns']['meditators']:
            f.write("Top patterns for MEDITATORS:\n")
            sorted_patterns = sorted(results['patterns']['meditators'].items(), 
                                    key=lambda x: x[1]['frequency'], reverse=True)[:5]
            for pattern, data in sorted_patterns:
                pattern_str = ' → '.join([f"S{s}" for s in pattern])
                f.write(f"  {pattern_str}: {data['frequency']:.3f} (Count: {data['count']}, Subjects: {data['n_subjects']})\n")
        else:
            f.write("No recurring patterns detected meeting criteria.\n")
            
def analyze_transitions(networks=7, k=4):
    """
    Analyze transitions for meditators only for 7-network, k=4 configuration.
    """
    logger.info(f"\n=== Analyzing meditator transitions for {networks}-network configuration ===")
    
    # Load meditator model
    med_model = load_model('meditators', k, networks)
    
    if med_model is None:
        logger.warning("Could not load meditator model data")
        return None
    
    # Extract state blocks and successions
    med_blocks, med_successions = extract_state_blocks(med_model['vpath'], med_model['indices'], k)
    
    # Calculate succession matrix
    med_counts, med_probs = calculate_succession_matrix(med_successions, k, exclude_first_blocks=True)
    
    # Detect patterns
    med_patterns = detect_state_patterns(med_successions, k, n_length=3)
    
    # Load state activations from CSV
    patterns_df = pd.read_csv(os.path.join(STATE_PATTERNS_DIR, 'all_states_activation_patterns.csv'))
    med_df = patterns_df[(patterns_df['group'] == 'meditators') & 
                        (patterns_df['networks'] == networks) & 
                        (patterns_df['k'] == k) & 
                        (patterns_df['standardization'] == 'bygroup')]
    
    med_state_info = {}
    network_fields = ['VIS', 'SMN', 'DAN', 'VAN', 'LIM', 'FPN', 'DMN']
    
    for s in range(k):
        med_state_data = med_df[med_df['state_idx'] == s]
        if not med_state_data.empty:
            med_state_means = med_state_data[network_fields].values[0]
            max_idx = np.argmax(np.abs(med_state_means))  # Use absolute value for dominant network
            med_state_info[s] = {
                'dominant_network': network_fields[max_idx],
                'activation': med_state_means[max_idx],
                'fractional_occupancy': med_state_data['fractional_occupancy'].values[0]
            }
    
    results = {
        'networks': networks,
        'k': k,
        'succession': {
            'meditators': {'counts': med_counts, 'probs': med_probs}
        },
        'state_info': {
            'meditators': med_state_info
        },
        'blocks': {'meditators': med_blocks},
        'patterns': {'meditators': med_patterns}
    }
    
    results_path = os.path.join(TRANSITIONS_DIR, f'meditator_transitions_{networks}networks_k{k}.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    logger.info(f"Saved meditator transition results to {results_path}")
    
    return results

def main():
    """Main function to run the meditator state transition analysis pipeline."""
    logger.info("=== Starting Meditator State Transition Analysis ===")
    
    for networks in NETWORK_CONFIGS:
        logger.info(f"\n=== Analyzing {networks}-network configuration ===")
        
        try:
            results = analyze_transitions(networks)
            
            if results is not None:
                visualize_succession_matrices(results)
                write_summary_report(results)
                logger.info(f"Completed transition analysis for {networks}-networks")
            else:
                logger.warning(f"Skipping visualizations for {networks}-networks due to missing data")
                
        except Exception as e:
            logger.error(f"Error in transition analysis for {networks}-networks: {str(e)}")
            import traceback
            traceback.print_exc()
    
    logger.info("=== Meditator State Transition Analysis Complete ===")

if __name__ == "__main__":
    main()