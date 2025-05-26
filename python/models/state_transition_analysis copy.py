"""
State Transition Analysis for Meditation Research

This script analyzes raw state transitions from TDE-HMM models
without imposing phenomenological labels, focusing on:
1. State transition matrices for individuals and groups
2. Common transition patterns and cycles
3. Differences in transition dynamics between meditators and controls


"""

import os
import numpy as np
import pickle
import logging
from datetime import datetime

# Setup minimal logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup paths
ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
TRAINED_DIR = os.path.join(DATA_DIR, 'trained', 'glhmm_tde')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')
TRANSITIONS_DIR = os.path.join(METRICS_DIR, 'transitions')
os.makedirs(TRANSITIONS_DIR, exist_ok=True)

def load_model(group, k):
    """Load trained HMM model."""
    model_path = os.path.join(TRAINED_DIR, group, f'k{k}', 'model.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # Make sure k is included
    if 'k' not in model_data:
        model_data['k'] = k
        
    logger.info(f"Loaded {group} k={k} model")
    return model_data

def calculate_subject_transitions(vpath, indices, k):
    """Calculate transition matrices for individual subjects."""
    subject_transitions = []
    
    # Ensure vpath is 1D array of state indices
    if len(vpath.shape) > 1 and vpath.shape[1] > 1:
        from glhmm import statistics
        vpath_1d = statistics.generate_vpath_1D(vpath)
    else:
        vpath_1d = vpath
    
    # For each subject
    for i, (start, end) in enumerate(indices):
        # Extract subject's state sequence
        subject_states = vpath_1d[start:end]
        
        # Initialize transition count matrix
        trans_counts = np.zeros((k, k))
        
        # Count transitions
        for t in range(len(subject_states)-1):
            from_state = int(subject_states[t])
            to_state = int(subject_states[t+1])
            
            # Only count valid state transitions
            if from_state < k and to_state < k:
                trans_counts[from_state, to_state] += 1
        
        # Convert to probabilities (row-normalized)
        trans_probs = np.zeros_like(trans_counts)
        for s in range(k):
            row_sum = np.sum(trans_counts[s, :])
            if row_sum > 0:
                trans_probs[s, :] = trans_counts[s, :] / row_sum
        
        subject_transitions.append({
            'subject_idx': i,
            'trans_counts': trans_counts,
            'trans_probs': trans_probs
        })
    
    return subject_transitions

def find_common_sequences(vpath, indices, k, min_length=3, max_length=5):
    """Find common state sequences in the Viterbi path."""
    if len(vpath.shape) > 1 and vpath.shape[1] > 1:
        from glhmm import statistics
        vpath_1d = statistics.generate_vpath_1D(vpath)
    else:
        vpath_1d = vpath
    
    # Store sequences by length
    all_sequences = {length: {} for length in range(min_length, max_length+1)}
    
    # For each subject
    for i, (start, end) in enumerate(indices):
        subject_states = vpath_1d[start:end]
        
        # For each possible sequence length
        for length in range(min_length, min(max_length+1, len(subject_states))):
            # Extract all sequences of this length
            for t in range(len(subject_states) - length + 1):
                seq = tuple(subject_states[t:t+length])
                
                # Skip sequences with invalid states
                if max(seq) >= k:
                    continue
                
                # Update sequence count
                if seq in all_sequences[length]:
                    all_sequences[length][seq]['count'] += 1
                    if i not in all_sequences[length][seq]['subjects']:
                        all_sequences[length][seq]['subjects'].append(i)
                else:
                    all_sequences[length][seq] = {
                        'count': 1,
                        'subjects': [i]
                    }
    
    # Find most common sequences
    common_sequences = {}
    for length, sequences in all_sequences.items():
        # Sort by count
        sorted_seqs = sorted(sequences.items(), key=lambda x: x[1]['count'], reverse=True)
        
        # Take top 5 most common sequences
        common_sequences[length] = sorted_seqs[:5]
    
    return common_sequences

def analyze_transitions(k_values=[4, 5]):
    """Analyze state transitions and successions for meditators and controls."""
    logger.info("Analyzing state transitions and successions...")
    
    # Process each k value
    for k in k_values:
        logger.info(f"Processing k={k} models...")
        
        results = {}
        
        # Process each group
        for group in ['controls', 'meditators']:
            logger.info(f"Analyzing {group} group...")
            
            # Load model data
            model_data = load_model(group, k)
            vpath = model_data['vpath']
            indices = model_data['indices']
            P = model_data['P']  # Group-level transition matrix
            
            # Calculate subject-level transitions
            subject_transitions = calculate_subject_transitions(vpath, indices, k)
            
            # Find common sequences
            common_sequences = find_common_sequences(vpath, indices, k)
            
            # Analyze state successions (NEW)
            succession_results = analyze_state_successions(vpath, indices, k)
            
            # Store results for this group
            results[group] = {
                'P': P,  # Group transition matrix
                'subject_transitions': subject_transitions,
                'common_sequences': common_sequences,
                'succession_results': succession_results  # NEW
            }
        
        # Export transition analysis
        output_path = os.path.join(TRANSITIONS_DIR, f'k{k}_transitions.txt')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # [Existing code for writing transition analysis]
            
            # NEW: Add state succession analysis
            f.write("\nSTATE SUCCESSION ANALYSIS\n")
            f.write("=======================\n\n")
            
            for group in ['controls', 'meditators']:
                f.write(f"{group.upper()} STATE SUCCESSIONS\n")
                f.write("-" * (len(group) + 18) + "\n\n")
                
                # Write succession probability matrix
                f.write("State Succession Matrix (after state X, which state follows?):\n")
                S = results[group]['succession_results']['succession_probs']
                
                # Write header
                f.write("From\\To ")
                for j in range(k):
                    f.write(f"S{j+1:d}    ")
                f.write("\n")
                
                # Write matrix
                for i in range(k):
                    f.write(f"S{i+1:d}     ")
                    for j in range(k):
                        f.write(f"{S[i, j]:.3f}  ")
                    f.write("\n")
                
                f.write("\n")
                
                # Write common state sequences (without repetitions)
                f.write("Common State Sequences (excluding repetitions):\n")
                common_successions = results[group]['succession_results']['common_successions']
                
                for idx, (seq, info) in enumerate(common_successions):
                    if idx >= 5:  # Limit to top 5
                        break
                    seq_str = '->'.join([f"S{int(s)+1}" for s in seq])
                    f.write(f"  {seq_str}: {info['count']} occurrences across {len(info['subjects'])} subjects\n")
                
                f.write("\n")
            
            # Compare succession probabilities between groups
            f.write("SUCCESSION PROBABILITY DIFFERENCES (Meditators - Controls):\n")
            S_control = results['controls']['succession_results']['succession_probs']
            S_meditator = results['meditators']['succession_results']['succession_probs']
            S_diff = S_meditator - S_control
            
            # Write header
            f.write("From\\To ")
            for j in range(k):
                f.write(f"S{j+1:d}    ")
            f.write("\n")
            
            # Write difference matrix
            for i in range(k):
                f.write(f"S{i+1:d}     ")
                for j in range(k):
                    value = S_diff[i, j]
                    if abs(value) > 0.1:  # Highlight differences > 10%
                        f.write(f"{value:+.3f}* ")
                    else:
                        f.write(f"{value:+.3f}  ")
                f.write("\n")
            
            f.write("\n")
            
            # Add notes specific to succession analysis
            f.write("SUCCESSION ANALYSIS NOTES\n")
            f.write("------------------------\n")
            f.write("- State successions show which state follows another after a continuous block\n")
            f.write("- This captures the natural progression of states rather than immediate transitions\n")
            f.write("- Sequences exclude repetitions (S1->S1->S1->S2 is recorded as S1->S2)\n")
            f.write("- * indicates notable differences (>10%) between groups\n")
            
        logger.info(f"Transition and succession analysis for k={k} saved to {output_path}")
    
    return TRANSITIONS_DIR

def analyze_state_successions(vpath, indices, k):
    """Analyze which states follow other states after continuous blocks."""
    # Ensure vpath is 1D array of state indices
    if len(vpath.shape) > 1 and vpath.shape[1] > 1:
        from glhmm import statistics
        vpath_1d = statistics.generate_vpath_1D(vpath)
    else:
        vpath_1d = vpath
    
    # Initialize succession count matrix
    succession_counts = np.zeros((k, k))
    
    # Store common state successions (which state follows which)
    common_successions = []
    
    # Track subject-level successions
    subject_successions = []
    
    # For each subject
    for i, (start, end) in enumerate(indices):
        # Extract subject's state sequence
        subject_states = vpath_1d[start:end]
        
        # Identify state blocks
        blocks = []
        current_state = subject_states[0]
        block_start = 0
        
        # Extract continuous state blocks
        for t in range(1, len(subject_states)):
            if subject_states[t] != current_state or t == len(subject_states) - 1:
                # Store the completed block
                block_end = t-1 if t < len(subject_states) - 1 else t
                
                if current_state < k:  # Ensure valid state
                    blocks.append({
                        'state': int(current_state),
                        'start': block_start,
                        'end': block_end,
                        'length': block_end - block_start + 1
                    })
                
                # Start a new block
                current_state = subject_states[t]
                block_start = t
        
        # Calculate block-to-block successions for this subject
        subject_succession_counts = np.zeros((k, k))
        subject_succession_seqs = []
        
        # Track the sequence of states for this subject (ignoring repetitions)
        state_sequence = []
        
        for b in range(len(blocks)-1):
            from_state = blocks[b]['state']
            to_state = blocks[b+1]['state']
            
            # Record the succession
            subject_succession_counts[from_state, to_state] += 1
            succession_counts[from_state, to_state] += 1
            
            # Add to state sequence
            if not state_sequence or state_sequence[-1] != from_state:
                state_sequence.append(from_state)
            
            # Record this specific succession
            subject_succession_seqs.append((from_state, to_state))
        
        # Add the last state to the sequence
        if blocks and (not state_sequence or state_sequence[-1] != blocks[-1]['state']):
            state_sequence.append(blocks[-1]['state'])
        
        # Convert to probabilities
        subject_succession_probs = np.zeros_like(subject_succession_counts)
        for s in range(k):
            row_sum = np.sum(subject_succession_counts[s, :])
            if row_sum > 0:
                subject_succession_probs[s, :] = subject_succession_counts[s, :] / row_sum
        
        # Store results for this subject
        subject_successions.append({
            'subject_idx': i,
            'blocks': blocks,
            'succession_counts': subject_succession_counts,
            'succession_probs': subject_succession_probs,
            'state_sequence': state_sequence,
            'succession_sequences': subject_succession_seqs
        })
    
    # Calculate group succession probabilities
    succession_probs = np.zeros_like(succession_counts)
    for s in range(k):
        row_sum = np.sum(succession_counts[s, :])
        if row_sum > 0:
            succession_probs[s, :] = succession_counts[s, :] / row_sum
    
    # Find common state sequences (excluding repetitions)
    state_sequences = {}
    for subj in subject_successions:
        seq = tuple(subj['state_sequence'])
        if seq in state_sequences:
            state_sequences[seq]['count'] += 1
            state_sequences[seq]['subjects'].append(subj['subject_idx'])
        else:
            state_sequences[seq] = {
                'count': 1,
                'subjects': [subj['subject_idx']]
            }
    
    # Sort sequences by frequency
    sorted_sequences = sorted(state_sequences.items(), key=lambda x: x[1]['count'], reverse=True)
    
    # Take top sequences
    common_successions = sorted_sequences[:10]
    
    return {
        'succession_counts': succession_counts,
        'succession_probs': succession_probs,
        'subject_successions': subject_successions,
        'common_successions': common_successions
    }
    
    

def main():
    """Main function for state transition analysis."""
    logger.info("=== Starting State Transition Analysis ===")
    
    try:
        # Analyze transitions for k=4 and k=5 only
        transition_dir = analyze_transitions(k_values=[4, 5])
        logger.info(f"Transition analysis complete. Results saved to {transition_dir}")
        
    except Exception as e:
        logger.error(f"Error in state transition analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()