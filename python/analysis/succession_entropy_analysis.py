"""
Entropy-Based Brain State Analysis for Meditation Research

This script implements information-theoretic analysis of brain state dynamics
during meditation using the normalized succession entropy approach from
Deco et al. (2021).

Key features:
1. Calculation of normalized succession entropy for state sequences
2. State-specific entropy and transition directionality assessment
3. Identification of expert meditators based on optimal entropy values
4. Visualization of brain state organizational principles

References:
- Deco, G., Vidaurre, D., & Kringelbach, M.L. (2021). Revisiting the global workspace
  orchestrating the hierarchical organization of the human brain. 
  Nature Human Behaviour, 5(4), 497-511.
- Vidaurre, D., et al. (2017). Brain network dynamics are hierarchically 
  organized in time. PNAS, 114(48), 12827-12832.
"""

import os
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glhmm import statistics, auxiliary

# Setup paths (following your existing structure)
ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
TRAINED_DIR = os.path.join(DATA_DIR, 'trained', 'glhmm_tde')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
ENTROPY_DIR = os.path.join(RESULTS_DIR, 'entropy_analysis')
os.makedirs(ENTROPY_DIR, exist_ok=True)

def load_trained_model(group, k):
    """Load a trained TDE-HMM model (reusing your existing function)."""
    print(f"Loading {group} model with k={k}...")
    
    model_path = os.path.join(TRAINED_DIR, group, f'k{k}', 'model.pkl')
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        if 'k' not in model_data:
            model_data['k'] = k
            
        print(f"Loaded model with {model_data.get('active_states', 'unknown')}/{k} states")
        return model_data
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def extract_state_blocks(vpath, indices, k):
    """
    Extract continuous state blocks from the Viterbi path.
    
    Parameters:
    -----------
    vpath : numpy.ndarray
        Viterbi path from HMM
    indices : list
        List of (start, end) tuples defining subject boundaries
    k : int
        Number of states
        
    Returns:
    --------
    list
        List of dictionaries with subject-level block information
    """
    # Ensure vpath is 1D array of state indices
    if len(vpath.shape) > 1 and vpath.shape[1] > 1:
        vpath_1d = statistics.generate_vpath_1D(vpath)
    else:
        vpath_1d = vpath
    
    # Initialize results list
    subject_blocks = []
    
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
        succession_counts = np.zeros((k, k))
        
        # Track the sequence of states for this subject (ignoring repetitions)
        state_sequence = []
        
        for b in range(len(blocks)-1):
            from_state = blocks[b]['state']
            to_state = blocks[b+1]['state']
            
            # Record the succession
            succession_counts[from_state, to_state] += 1
            
            # Add to state sequence
            if not state_sequence or state_sequence[-1] != from_state:
                state_sequence.append(from_state)
        
        # Add the last state to the sequence
        if blocks and (not state_sequence or state_sequence[-1] != blocks[-1]['state']):
            state_sequence.append(blocks[-1]['state'])
        
        # Convert to probabilities
        succession_probs = np.zeros_like(succession_counts)
        for s in range(k):
            row_sum = np.sum(succession_counts[s, :])
            if row_sum > 0:
                succession_probs[s, :] = succession_counts[s, :] / row_sum
        
        # Store results for this subject
        subject_blocks.append({
            'subject_idx': i,
            'blocks': blocks,
            'succession_counts': succession_counts,
            'succession_probs': succession_probs,
            'state_sequence': state_sequence
        })
    
    return subject_blocks

def calculate_normalized_succession_entropy(succession_probs, fractional_occupancy):
    """
    Calculate normalized succession entropy following Deco et al. (2021).
    
    Parameters:
    -----------
    succession_probs : numpy.ndarray
        Succession probability matrix, where each element (i,j) represents 
        the probability of transitioning from state i to state j
    fractional_occupancy : numpy.ndarray
        Fractional occupancy of each state
        
    Returns:
    --------
    float
        Normalized succession entropy value (0-1)
    dict
        Additional entropy metrics
    """
    k = succession_probs.shape[0]
    state_entropies = np.zeros(k)
    
    # Calculate entropy for each state
    for i in range(k):
        # Only consider valid probability distributions (sum to ~1)
        row_sum = np.sum(succession_probs[i, :])
        if row_sum > 0.5:  # Consider valid if sum is close to 1
            probs = succession_probs[i, :]
            # Avoid log(0) by only considering non-zero probabilities
            valid_indices = probs > 0
            if np.any(valid_indices):
                valid_probs = probs[valid_indices]
                state_entropies[i] = -np.sum(valid_probs * np.log2(valid_probs))
    
    # Calculate weighted average entropy
    avg_entropy = np.sum(fractional_occupancy * state_entropies)
    
    # Normalize by maximum possible entropy
    max_entropy = np.log2(k)
    normalized_entropy = avg_entropy / max_entropy if max_entropy > 0 else 0
    
    # Calculate additional metrics
    transition_directionality = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i != j:
                p_ij = succession_probs[i, j]
                p_ji = succession_probs[j, i]
                
                # Calculate directional asymmetry
                if p_ij > 0 or p_ji > 0:
                    transition_directionality[i, j] = (p_ij - p_ji) / (p_ij + p_ji)
    
    # Return both the normalized entropy and additional metrics
    additional_metrics = {
        'state_entropies': state_entropies,
        'avg_entropy': avg_entropy,
        'max_entropy': max_entropy,
        'transition_directionality': transition_directionality
    }
    
    return normalized_entropy, additional_metrics

def identify_expert_meditators(subject_data, k, top_n=5):
    """
    Identify expert meditators based on optimal entropy values and other metrics.
    
    Parameters:
    -----------
    subject_data : list
        List of subject-level block data including entropy values
    k : int
        Number of states
    top_n : int
        Number of top experts to identify
        
    Returns:
    --------
    list
        Indices of top expert meditators
    """
    # Create scoring metrics
    n_subjects = len(subject_data)
    expertise_scores = np.zeros(n_subjects)
    
    # Ideal entropy range (based on Deco's theory of optimal balance)
    # Not too rigid (>0.3) and not too random (<0.7)
    optimal_entropy_min = 0.3
    optimal_entropy_max = 0.7
    
    for i, subject in enumerate(subject_data):
        # Get normalized entropy
        entropy = subject['normalized_entropy']
        
        # Calculate how close entropy is to optimal range
        if entropy < optimal_entropy_min:
            entropy_score = entropy / optimal_entropy_min
        elif entropy > optimal_entropy_max:
            entropy_score = 1 - ((entropy - optimal_entropy_max) / (1 - optimal_entropy_max))
        else:
            entropy_score = 1.0  # In optimal range
        
        # Factor in presence of cyclic patterns
        cyclic_score = 0
        if 'has_cycles' in subject:
            cyclic_score = 1.0 if subject['has_cycles'] else 0.0
        
        # Combine metrics (weighted sum)
        expertise_scores[i] = (0.6 * entropy_score) + (0.4 * cyclic_score)
    
    # Identify top experts
    top_indices = np.argsort(expertise_scores)[-top_n:][::-1]
    return top_indices.tolist(), expertise_scores

def analyze_succession_entropy(k_values=[4, 5]):
    """
    Main function to analyze succession entropy for meditation research.
    
    Parameters:
    -----------
    k_values : list
        List of k values to analyze
        
    Returns:
    --------
    dict
        Results dictionary with entropy metrics
    """
    print("=== ENTROPY-BASED BRAIN STATE ANALYSIS ===")
    
    results = {}
    
    # Process each k value
    for k in k_values:
        print(f"\nAnalyzing models with k={k}...")
        
        k_results = {}
        
        # Process each group
        for group in ['controls', 'meditators']:
            print(f"Processing {group} group...")
            
            # Load model data
            model_data = load_trained_model(group, k)
            
            if model_data is None:
                print(f"Skipping {group} with k={k} - model not found")
                continue
            
            # Extract key variables
            vpath = model_data['vpath']
            indices = model_data['indices']
            FO = model_data.get('FO', np.ones(k)/k)  # Default if missing
            
            # Calculate group-level fractional occupancy
            fo_group = np.mean(FO, axis=0)
            
            # Extract block-level data
            subject_blocks = extract_state_blocks(vpath, indices, k)
            
            # Calculate group-level succession probabilities
            group_succession_counts = np.zeros((k, k))
            for subj in subject_blocks:
                group_succession_counts += subj['succession_counts']
            
            group_succession_probs = np.zeros_like(group_succession_counts)
            for s in range(k):
                row_sum = np.sum(group_succession_counts[s, :])
                if row_sum > 0:
                    group_succession_probs[s, :] = group_succession_counts[s, :] / row_sum
            
            # Calculate group-level normalized entropy
            norm_entropy, add_metrics = calculate_normalized_succession_entropy(
                group_succession_probs, fo_group
            )
            
            # Calculate subject-level entropy values
            for subj in subject_blocks:
                subj_entropy, subj_metrics = calculate_normalized_succession_entropy(
                    subj['succession_probs'], fo_group
                )
                subj['normalized_entropy'] = subj_entropy
                subj['entropy_metrics'] = subj_metrics
                
                # Identify cyclic patterns
                state_seq = subj['state_sequence']
                has_cycles = False
                if len(state_seq) >= 3:
                    # Check for any state that appears twice with states in between
                    for i in range(len(state_seq)-2):
                        for j in range(i+2, len(state_seq)):
                            if state_seq[i] == state_seq[j]:
                                has_cycles = True
                                break
                        if has_cycles:
                            break
                subj['has_cycles'] = has_cycles
            
            # For meditator group, identify expert meditators
            expert_indices = []
            expertise_scores = []
            if group == 'meditators':
                expert_indices, expertise_scores = identify_expert_meditators(subject_blocks, k)
            
            # Store results for this group
            k_results[group] = {
                'group_succession_probs': group_succession_probs,
                'normalized_entropy': norm_entropy,
                'additional_metrics': add_metrics,
                'subject_blocks': subject_blocks,
                'expert_indices': expert_indices,
                'expertise_scores': expertise_scores
            }
        
        # Store results for this k
        results[k] = k_results
        
        # Generate report for this k
        generate_entropy_report(k, k_results)
        
        # Create visualizations
        create_entropy_visualizations(k, k_results)
    
    print("\n=== ENTROPY ANALYSIS COMPLETE ===")
    return results

def generate_entropy_report(k, results):
    """
    Generate a text report of entropy analysis results.
    
    Parameters:
    -----------
    k : int
        Number of states
    results : dict
        Results dictionary for this k value
    """
    output_path = os.path.join(ENTROPY_DIR, f'entropy_analysis_k{k}.txt')
    
    with open(output_path, 'w') as f:
        f.write(f"ENTROPY-BASED BRAIN STATE ANALYSIS (k={k})\n")
        f.write("=" * 50 + "\n\n")
        
        # Write group-level entropy results
        f.write("GROUP-LEVEL NORMALIZED SUCCESSION ENTROPY\n")
        f.write("-" * 50 + "\n\n")
        
        for group in results:
            f.write(f"{group.upper()}:\n")
            f.write(f"Normalized Entropy: {results[group]['normalized_entropy']:.4f}\n")
            
            # Write state-specific entropy values
            state_entropies = results[group]['additional_metrics']['state_entropies']
            f.write("\nState-specific entropy values:\n")
            for i, entropy in enumerate(state_entropies):
                f.write(f"  State {i+1}: {entropy:.4f}\n")
            
            f.write("\n")
        
        # Write entropy comparison
        if 'meditators' in results and 'controls' in results:
            med_entropy = results['meditators']['normalized_entropy']
            con_entropy = results['controls']['normalized_entropy']
            diff = med_entropy - con_entropy
            
            f.write("\nGROUP COMPARISON\n")
            f.write("-" * 50 + "\n\n")
            f.write(f"Meditators vs Controls Entropy Difference: {diff:+.4f}\n")
            
            interpretation = ""
            if abs(diff) < 0.05:
                interpretation = "Similar overall organization but potentially different patterns"
            elif diff < 0:
                interpretation = "Meditators show more structured/predictable state progressions"
            else:
                interpretation = "Meditators show more variable/flexible state progressions"
            
            f.write(f"Interpretation: {interpretation}\n\n")
        
        # Write expert meditator information if available
        if 'meditators' in results and results['meditators']['expert_indices']:
            f.write("\nEXPERT MEDITATORS\n")
            f.write("-" * 50 + "\n\n")
            
            expert_indices = results['meditators']['expert_indices']
            expertise_scores = results['meditators']['expertise_scores']
            subjects = results['meditators']['subject_blocks']
            
            f.write("Identified expert meditators:\n")
            for i, idx in enumerate(expert_indices):
                subject = next((s for s in subjects if s['subject_idx'] == idx), None)
                if subject:
                    f.write(f"  Subject {idx+1}: Entropy = {subject['normalized_entropy']:.4f}, ")
                    f.write(f"Expertise Score = {expertise_scores[i]:.4f}\n")
            
            f.write("\n")
        
        f.write("\nINTERPRETATION NOTES\n")
        f.write("-" * 50 + "\n\n")
        f.write("- Normalized entropy ranges from 0 (completely predictable) to 1 (random)\n")
        f.write("- Values around 0.4-0.6 often represent optimal balance between structure and flexibility\n")
        f.write("- Expert meditators typically show more organized yet adaptable patterns\n")
        f.write("- State-specific entropy identifies which brain states serve as decision points\n")
    
    print(f"Report generated: {output_path}")

def create_entropy_visualizations(k, results):
    """
    Create visualizations for entropy analysis results.
    
    Parameters:
    -----------
    k : int
        Number of states
    results : dict
        Results dictionary for this k value
    """
    output_dir = os.path.join(ENTROPY_DIR, f'k{k}_visualizations')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Plot group entropy comparison
    if 'meditators' in results and 'controls' in results:
        plt.figure(figsize=(8, 6))
        groups = ['Controls', 'Meditators']
        entropies = [
            results['controls']['normalized_entropy'],
            results['meditators']['normalized_entropy']
        ]
        
        plt.bar(groups, entropies, color=['lightblue', 'darkblue'])
        plt.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Balanced point')
        plt.ylim(0, 1)
        plt.ylabel('Normalized Succession Entropy')
        plt.title(f'Brain State Organization (k={k})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'group_entropy_comparison.png'), dpi=300)
        plt.close()
    
    # 2. Plot state-specific entropy values
    plt.figure(figsize=(10, 6))
    
    if 'meditators' in results and 'controls' in results:
        med_state_entropies = results['meditators']['additional_metrics']['state_entropies']
        con_state_entropies = results['controls']['additional_metrics']['state_entropies']
        
        x = np.arange(k)
        width = 0.35
        
        plt.bar(x - width/2, con_state_entropies, width, label='Controls', color='lightblue')
        plt.bar(x + width/2, med_state_entropies, width, label='Meditators', color='darkblue')
        
        plt.xlabel('State')
        plt.ylabel('Entropy')
        plt.title(f'State-Specific Entropy (k={k})')
        plt.xticks(x, [f'State {i+1}' for i in range(k)])
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'state_specific_entropy.png'), dpi=300)
        plt.close()
    
    # 3. Plot transition directionality heatmap
    for group in results:
        transition_dir = results[group]['additional_metrics']['transition_directionality']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(transition_dir, cmap='coolwarm', center=0, 
                   xticklabels=[f'S{i+1}' for i in range(k)],
                   yticklabels=[f'S{i+1}' for i in range(k)])
        plt.title(f'{group.capitalize()} - Transition Directionality (k={k})')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{group}_transition_directionality.png'), dpi=300)
        plt.close()
    
    # 4. Plot subject-level entropy distributions
    plt.figure(figsize=(10, 6))
    
    if 'meditators' in results and 'controls' in results:
        med_subjects = results['meditators']['subject_blocks']
        con_subjects = results['controls']['subject_blocks']
        
        med_entropy = [s['normalized_entropy'] for s in med_subjects]
        con_entropy = [s['normalized_entropy'] for s in con_subjects]
        
        plt.hist(con_entropy, alpha=0.5, label='Controls', color='lightblue')
        plt.hist(med_entropy, alpha=0.5, label='Meditators', color='darkblue')
        
        plt.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Balanced point')
        plt.xlabel('Normalized Succession Entropy')
        plt.ylabel('Number of Subjects')
        plt.title(f'Distribution of Subject-Level Entropy (k={k})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'subject_entropy_distribution.png'), dpi=300)
        plt.close()
    
    # 5. Highlight expert meditators
    if 'meditators' in results and results['meditators']['expert_indices']:
        plt.figure(figsize=(10, 6))
        
        med_subjects = results['meditators']['subject_blocks']
        all_entropy = np.array([s['normalized_entropy'] for s in med_subjects])
        all_indices = np.array([s['subject_idx'] for s in med_subjects])
        
        expert_indices = results['meditators']['expert_indices']
        expert_mask = np.isin(all_indices, expert_indices)
        
        # Plot all subjects
        plt.scatter(all_indices[~expert_mask], all_entropy[~expert_mask], 
                   label='Regular meditators', color='lightblue', s=50)
        
        # Highlight experts
        plt.scatter(all_indices[expert_mask], all_entropy[expert_mask], 
                   label='Expert meditators', color='darkred', s=100)
        
        plt.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Balanced point')
        plt.xlabel('Subject Index')
        plt.ylabel('Normalized Succession Entropy')
        plt.title(f'Expert Meditators Identification (k={k})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'expert_meditators.png'), dpi=300)
        plt.close()
    
    print(f"Visualizations saved to: {output_dir}")

if __name__ == "__main__":
    # Run analysis for k=4 and k=5 models
    results = analyze_succession_entropy(k_values=[4, 5])