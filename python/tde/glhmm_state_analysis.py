import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from glhmm import utils, graphics

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, 'results', 'hmm_time_delay_embedded')
PLOT_DIR = os.path.join(RESULTS_DIR, 'state_visualization')
os.makedirs(PLOT_DIR, exist_ok=True)

def load_model_results(results_dir):
    """Load model results from the saved file"""
    with open(os.path.join(results_dir, 'glhmm_results_improved.pkl'), 'rb') as f:
        results = pickle.load(f)
    return results

def load_time_delay_models():
    """Load time-delay embedded models"""
    print("Loading time-delay embedded models...")
    
    # Load control model
    control_path = os.path.join(RESULTS_DIR, 'controls', '4states_full_model.pkl')
    with open(control_path, 'rb') as f:
        control_data = pickle.load(f)
    
    # Load meditator model
    meditator_path = os.path.join(RESULTS_DIR, 'meditators', '4states_full_model.pkl')
    with open(meditator_path, 'rb') as f:
        meditator_data = pickle.load(f)
    
    # Extract components
    model_controls = control_data['model']
    model_meditators = meditator_data['model']
    network_fields = control_data['network_fields']
    
    print(f"Loaded models: Controls ({model_controls.get_active_K()} states), Meditators ({model_meditators.get_active_K()} states)")
    
    return model_controls, model_meditators, network_fields

def visualize_transition_matrices(model_controls, model_meditators, plot_dir):
    """Visualize transition matrices for both groups"""
    # Get transition matrices
    P_controls = model_controls.get_P()
    P_meditators = model_meditators.get_P()
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Controls transition matrix
    im0 = axes[0].imshow(P_controls, cmap='hot', interpolation='nearest')
    axes[0].set_title('Controls: Transition Matrix', fontsize=14)
    axes[0].set_xlabel('To State', fontsize=12)
    axes[0].set_ylabel('From State', fontsize=12)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Meditators transition matrix
    im1 = axes[1].imshow(P_meditators, cmap='hot', interpolation='nearest')
    axes[1].set_title('Meditators: Transition Matrix', fontsize=14)
    axes[1].set_xlabel('To State', fontsize=12)
    axes[1].set_ylabel('From State', fontsize=12)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'transition_matrices.png'), dpi=150)
    plt.close()
    
    # Also plot without self-transitions for clearer visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Remove self-transitions
    P_controls_noself = P_controls - np.diag(np.diag(P_controls))
    P_controls_noself = P_controls_noself / np.maximum(P_controls_noself.sum(axis=1, keepdims=True), 1e-10)
    
    P_meditators_noself = P_meditators - np.diag(np.diag(P_meditators))
    P_meditators_noself = P_meditators_noself / np.maximum(P_meditators_noself.sum(axis=1, keepdims=True), 1e-10)
    
    # Controls transition matrix without self-transitions
    im0 = axes[0].imshow(P_controls_noself, cmap='hot', interpolation='nearest')
    axes[0].set_title('Controls: Transitions (excl. self)', fontsize=14)
    axes[0].set_xlabel('To State', fontsize=12)
    axes[0].set_ylabel('From State', fontsize=12)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Meditators transition matrix without self-transitions
    im1 = axes[1].imshow(P_meditators_noself, cmap='hot', interpolation='nearest')
    axes[1].set_title('Meditators: Transitions (excl. self)', fontsize=14)
    axes[1].set_xlabel('To State', fontsize=12)
    axes[1].set_ylabel('From State', fontsize=12)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'transition_matrices_no_self.png'), dpi=150)
    plt.close()

def visualize_viterbi_paths(results, plot_dir):
    """Decode and visualize Viterbi paths for both groups"""
    model_controls = results['model_controls']
    model_meditators = results['model_meditators']
    X_controls = results['X_controls']
    Y_controls = results['Y_controls']
    X_meditators = results['X_meditators']
    Y_meditators = results['Y_meditators']
    indices_controls = results['indices_controls']
    indices_meditators = results['indices_meditators']
    
    # Decode Viterbi paths
    print("Decoding Viterbi path for controls...")
    vpath_controls = model_controls.decode(X=X_controls, Y=Y_controls, indices=indices_controls, viterbi=True)
    print("Decoding Viterbi path for meditators...")
    vpath_meditators = model_meditators.decode(X=X_meditators, Y=Y_meditators, indices=indices_meditators, viterbi=True)
    
    # Handle case where decode returns 2D array instead of 1D array of indices
    if vpath_controls.ndim > 1 and vpath_controls.shape[1] > 1:
        print("Converting 2D Viterbi path to 1D array of state indices...")
        vpath_controls = np.argmax(vpath_controls, axis=1)
    else:
        vpath_controls = np.round(vpath_controls).astype(int)
        
    if vpath_meditators.ndim > 1 and vpath_meditators.shape[1] > 1:
        vpath_meditators = np.argmax(vpath_meditators, axis=1)
    else:
        vpath_meditators = np.round(vpath_meditators).astype(int)
    
    # Convert to state-time matrix for visualization
    from glhmm.statistics import viterbi_path_to_stc
    K_controls = model_controls.hyperparameters["K"]
    K_meditators = model_meditators.hyperparameters["K"]
    
    vpath_stc_controls = viterbi_path_to_stc(vpath_controls, K_controls)
    vpath_stc_meditators = viterbi_path_to_stc(vpath_meditators, K_meditators)
    
    # Plot controls Viterbi path
    graphics.plot_vpath(vpath_stc_controls, idx_data=indices_controls, 
                       title="Controls: State Time Courses", 
                       ylabel="States",
                       figsize=(12, 6),
                       save_path=os.path.join(plot_dir, 'viterbi_path_controls.png'))
    
    # Plot meditators Viterbi path
    graphics.plot_vpath(vpath_stc_meditators, idx_data=indices_meditators, 
                       title="Meditators: State Time Courses", 
                       ylabel="States",
                       figsize=(12, 6),
                       save_path=os.path.join(plot_dir, 'viterbi_path_meditators.png'))
    
    return vpath_controls, vpath_meditators

def plot_network_profiles(model_controls, model_meditators, network_fields, plot_dir):
    """Plot network activation profiles for each state in both groups"""
    # Get number of states
    K_controls = model_controls.get_active_K()
    K_meditators = model_meditators.get_active_K()
    
    # Get state means
    means_controls = [model_controls.get_mean(k) for k in range(K_controls)]
    means_meditators = [model_meditators.get_mean(k) for k in range(K_meditators)]
    
    # Determine number of embeddings (assuming equal feature dimensions)
    n_networks = len(network_fields)
    n_features = len(means_controls[0])
    n_embeddings = n_features // n_networks
    
    print(f"Detected {n_embeddings} time embeddings in the data")
    
    # Only use the first embedding for visualization (current state)
    means_controls_current = [mean[:n_networks] for mean in means_controls]
    means_meditators_current = [mean[:n_networks] for mean in means_meditators]
    
    # Plot controls
    plt.figure(figsize=(12, 8))
    for k in range(K_controls):
        plt.plot(network_fields, means_controls_current[k], marker='o', linewidth=2, 
                 label=f'State {k+1}')
    
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    plt.axhline(y=0.2, color='green', linestyle=':', alpha=0.5)
    plt.axhline(y=-0.2, color='red', linestyle=':', alpha=0.5)
    
    plt.ylabel("Network Activation", fontsize=12)
    plt.title(f"Controls Brain State Network Profiles", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'network_profiles_controls.png'), dpi=150)
    plt.close()
    
    # Plot meditators
    plt.figure(figsize=(12, 8))
    for k in range(K_meditators):
        plt.plot(network_fields, means_meditators_current[k], marker='o', linewidth=2,
                 label=f'State {k+1}')
    
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    plt.axhline(y=0.2, color='green', linestyle=':', alpha=0.5)
    plt.axhline(y=-0.2, color='red', linestyle=':', alpha=0.5)
    
    plt.ylabel("Network Activation", fontsize=12)
    plt.title(f"Meditators Brain State Network Profiles", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'network_profiles_meditators.png'), dpi=150)
    plt.close()
    
    # Create individual state bar plots for clearer visualization
    for k in range(K_controls):
        plt.figure(figsize=(10, 6))
        plt.bar(network_fields, means_controls_current[k], color='blue', alpha=0.7)
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        plt.title(f"Controls State {k+1} Network Profile", fontsize=14)
        plt.ylabel("Network Activation", fontsize=12)
        plt.xticks(rotation=45, fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'network_profile_controls_state_{k+1}.png'), dpi=150)
        plt.close()
        
    for k in range(K_meditators):
        plt.figure(figsize=(10, 6))
        plt.bar(network_fields, means_meditators_current[k], color='red', alpha=0.7)
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        plt.title(f"Meditators State {k+1} Network Profile", fontsize=14)
        plt.ylabel("Network Activation", fontsize=12)
        plt.xticks(rotation=45, fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'network_profile_meditators_state_{k+1}.png'), dpi=150)
        plt.close()

def analyze_dwell_times(results, plot_dir):
    """Calculate and visualize average dwell times for each state"""
    # Extract necessary data
    model_controls = results['model_controls']
    model_meditators = results['model_meditators']
    indices_controls = results['indices_controls']
    indices_meditators = results['indices_meditators']
    
    # Get state lifetimes from utils
    mean_dwell_controls, median_dwell_controls, _ = utils.get_life_times(model_controls.Gamma, indices_controls)
    mean_dwell_meditators, median_dwell_meditators, _ = utils.get_life_times(model_meditators.Gamma, indices_meditators)
    
    # Get number of active states
    K_controls = model_controls.get_active_K()
    K_meditators = model_meditators.get_active_K()
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Controls
    bars1 = ax1.bar(range(1, K_controls+1), mean_dwell_controls, color='blue', alpha=0.7)
    ax1.set_title('Controls: Mean Dwell Time by State', fontsize=14)
    ax1.set_xlabel('State', fontsize=12)
    ax1.set_ylabel('Mean Dwell Time (timepoints)', fontsize=12)
    ax1.set_xticks(range(1, K_controls+1))
    
    # Add values on top of bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}', ha='center', va='bottom')
    
    # Meditators
    bars2 = ax2.bar(range(1, K_meditators+1), mean_dwell_meditators, color='red', alpha=0.7)
    ax2.set_title('Meditators: Mean Dwell Time by State', fontsize=14)
    ax2.set_xlabel('State', fontsize=12)
    ax2.set_ylabel('Mean Dwell Time (timepoints)', fontsize=12)
    ax2.set_xticks(range(1, K_meditators+1))
    
    # Add values on top of bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'dwell_times.png'), dpi=150)
    plt.close()
    
    # Print a table with dwell time information
    print("\nDwell Time Analysis:")
    print("-" * 60)
    print(f"{'State':<10} {'Controls Mean':<15} {'Controls Median':<15} {'Meditators Mean':<15} {'Meditators Median':<15}")
    print("-" * 60)
    
    for i in range(max(K_controls, K_meditators)):
        c_mean = f"{mean_dwell_controls[i]:.1f}" if i < K_controls else "N/A"
        c_median = f"{median_dwell_controls[i]:.1f}" if i < K_controls else "N/A"
        m_mean = f"{mean_dwell_meditators[i]:.1f}" if i < K_meditators else "N/A"
        m_median = f"{median_dwell_meditators[i]:.1f}" if i < K_meditators else "N/A"
        
        print(f"{i+1:<10} {c_mean:<15} {c_median:<15} {m_mean:<15} {m_median:<15}")
    
    print("-" * 60)

def plot_subject_viterbi_paths(results, plot_dir, n_subjects=5):
    """Plot Viterbi paths for the first few individual subjects"""
    # Create directory for subject plots
    subject_dir = os.path.join(plot_dir, 'subjects')
    os.makedirs(subject_dir, exist_ok=True)
    
    # Extract data
    model_controls = results['model_controls']
    model_meditators = results['model_meditators']
    X_controls = results['X_controls']
    Y_controls = results['Y_controls']
    X_meditators = results['X_meditators']
    Y_meditators = results['Y_meditators']
    indices_controls = results['indices_controls']
    indices_meditators = results['indices_meditators']
    
    # Decode Viterbi paths
    vpath_controls = model_controls.decode(X=X_controls, Y=Y_controls, indices=indices_controls, viterbi=True)
    vpath_meditators = model_meditators.decode(X=X_meditators, Y=Y_meditators, indices=indices_meditators, viterbi=True)
    
    # Handle case where decode returns 2D array instead of 1D array of indices
    if vpath_controls.ndim > 1 and vpath_controls.shape[1] > 1:
        vpath_controls = np.argmax(vpath_controls, axis=1)
    else:
        vpath_controls = np.round(vpath_controls).astype(int)
        
    if vpath_meditators.ndim > 1 and vpath_meditators.shape[1] > 1:
        vpath_meditators = np.argmax(vpath_meditators, axis=1)
    else:
        vpath_meditators = np.round(vpath_meditators).astype(int)
    
    # Convert to state-time matrix format for visualization
    from glhmm.statistics import viterbi_path_to_stc
    K_controls = model_controls.hyperparameters["K"]
    K_meditators = model_meditators.hyperparameters["K"]
    
    # Plot for the first n_subjects
    for i in range(min(n_subjects, len(indices_controls))):
        # Get subject timepoints for controls
        start_idx = indices_controls[i, 0]
        end_idx = indices_controls[i, 1]
        
        # Get subject Viterbi path
        subject_vpath = vpath_controls[start_idx:end_idx]
        
        # Convert to state-time matrix
        subject_vpath_stc = viterbi_path_to_stc(subject_vpath, K_controls)
        
        # Plot
        graphics.plot_vpath(subject_vpath_stc, 
                          title=f"Control Subject {i+1} State Time Course", 
                          ylabel="States",
                          figsize=(12, 3),
                          save_path=os.path.join(subject_dir, f'viterbi_control_subject_{i+1}.png'))
    
    # Plot for meditator subjects
    for i in range(min(n_subjects, len(indices_meditators))):
        # Get subject timepoints for meditators
        start_idx = indices_meditators[i, 0]
        end_idx = indices_meditators[i, 1]
        
        # Get subject Viterbi path
        subject_vpath = vpath_meditators[start_idx:end_idx]
        
        # Convert to state-time matrix
        subject_vpath_stc = viterbi_path_to_stc(subject_vpath, K_meditators)
        
        # Plot
        graphics.plot_vpath(subject_vpath_stc, 
                          title=f"Meditator Subject {i+1} State Time Course", 
                          ylabel="States",
                          figsize=(12, 3),
                          save_path=os.path.join(subject_dir, f'viterbi_meditator_subject_{i+1}.png'))
        
def analyze_transition_matrices(model, original_prior, group_name, save_dir):
    """Analyze how transition matrices changed during training"""
    # Get final transition matrix
    final_P = model.get_P()
    
    # Calculate differences
    diff = final_P - original_prior
    
    # Save matrix comparison
    matrix_info = {
        'prior': original_prior,
        'trained': final_P,
        'difference': diff
    }
    
    # Print key transitions
    print(f"\n{group_name} Transition Matrix Analysis:")
    print("  Key transitions (Prior → Trained):")
    
    # Sustained states
    print(f"  BF→BF: {original_prior[0,0]:.2f} → {final_P[0,0]:.2f} (Δ={diff[0,0]:.2f})")
    print(f"  MW→MW: {original_prior[1,1]:.2f} → {final_P[1,1]:.2f} (Δ={diff[1,1]:.2f})")
    
    # Trigger transitions
    print(f"  MW→MA: {original_prior[1,2]:.2f} → {final_P[1,2]:.2f} (Δ={diff[1,2]:.2f})")
    print(f"  MA→RA: {original_prior[2,3]:.2f} → {final_P[2,3]:.2f} (Δ={diff[2,3]:.2f})")
    print(f"  RA→BF: {original_prior[3,0]:.2f} → {final_P[3,0]:.2f} (Δ={diff[3,0]:.2f})")
    
    # Save analysis
    with open(os.path.join(save_dir, f'{group_name}_transition_analysis.pkl'), 'wb') as f:
        pickle.dump(matrix_info, f)
    
    return matrix_info
        
def map_states_to_meditation_hierarchy(model, network_fields):
    """Map GLHMM states to hierarchical meditation states"""
    # Get number of states
    n_states = model.get_active_K()
    
    # Extract key properties
    means = np.array([model.get_mean(k) for k in range(n_states)])
    P = model.get_P()
    
    # Create scores for each state type
    state_scores = {}
    
    for state in range(n_states):
        # Get network activations
        dmn_idx = network_fields.index('DMN') if 'DMN' in network_fields else -1
        fpn_idx = network_fields.index('FPN') if 'FPN' in network_fields else -1 
        van_idx = network_fields.index('VAN') if 'VAN' in network_fields else -1
        dan_idx = network_fields.index('DAN') if 'DAN' in network_fields else -1
        smn_idx = network_fields.index('SMN') if 'SMN' in network_fields else -1
        
        # Get values (with safety checks)
        dmn_val = means[state, dmn_idx] if dmn_idx >= 0 else 0
        fpn_val = means[state, fpn_idx] if fpn_idx >= 0 else 0
        van_val = means[state, van_idx] if van_idx >= 0 else 0
        dan_val = means[state, dan_idx] if dan_idx >= 0 else 0
        smn_val = means[state, smn_idx] if smn_idx >= 0 else 0
        
        # Calculate scores for each meditation state type
        bf_score = (dan_val + smn_val - dmn_val) 
        mw_score = (dmn_val - fpn_val - dan_val)
        ma_score = van_val + 0.5*fpn_val  # VAN is critical for meta-awareness
        ra_score = fpn_val + dan_val - dmn_val
        
        # Calculate sustained vs transient score
        sustained_score = P[state, state]  # Self-transition probability
        
        state_scores[state] = {
            'BF_score': bf_score,
            'MW_score': mw_score,
            'MA_score': ma_score,
            'RA_score': ra_score,
            'sustained_score': sustained_score
        }
    
    # First divide into sustained vs transient states
    states_by_sustained = sorted(range(n_states), key=lambda x: state_scores[x]['sustained_score'], reverse=True)
    sustained_states = states_by_sustained[:2]  # 2 sustained states
    transient_states = states_by_sustained[2:]  # 2 transient states
    
    # Assign the sustained states (BF, MW)
    bf_scores = [(s, state_scores[s]['BF_score']) for s in sustained_states]
    mw_scores = [(s, state_scores[s]['MW_score']) for s in sustained_states]
    
    bf_state = max(bf_scores, key=lambda x: x[1])[0]
    mw_state = max(mw_scores, key=lambda x: x[1])[0]
    
    # Handle ties
    if bf_state == mw_state:
        # Use transition dynamics as tiebreaker - BF should have higher self-transition
        if P[bf_state, bf_state] > P[mw_state, mw_state]:
            mw_state = [s for s in sustained_states if s != bf_state][0]
        else:
            bf_state = [s for s in sustained_states if s != mw_state][0]
    
    # Assign the transient states (MA, RA)
    ma_scores = [(s, state_scores[s]['MA_score']) for s in transient_states]
    ra_scores = [(s, state_scores[s]['RA_score']) for s in transient_states]
    
    ma_state = max(ma_scores, key=lambda x: x[1])[0] if ma_scores else None
    ra_state = max(ra_scores, key=lambda x: x[1])[0] if ra_scores else None
    
    # Handle ties with transition dynamics
    if ma_state == ra_state and len(transient_states) > 1:
        # MA should have strong transitions from MW
        # RA should have strong transitions to BF
        ma_from_mw = [(s, P[mw_state, s]) for s in transient_states]
        ra_to_bf = [(s, P[s, bf_state]) for s in transient_states]
        
        ma_state = max(ma_from_mw, key=lambda x: x[1])[0]
        ra_state = max(ra_to_bf, key=lambda x: x[1])[0]
    
    # Create mapping
    mapping = {
        'BF': bf_state,
        'MW': mw_state,
        'MA': ma_state,
        'RA': ra_state
    }
    
    # Return mapping and hierarchical information
    return {
        'state_mapping': mapping,
        'hierarchy': {
            'sustained_states': [bf_state, mw_state],
            'transition_states': [ma_state, ra_state]
        },
        'state_scores': state_scores
    }

def analyze_meditation_cycle(model, state_mapping=None):
    """Analyze meditation cycle completion and triggers"""
    P = model.get_P()
    
    # If no mapping provided, use default state order
    if state_mapping is None:
        bf_idx, mw_idx, ma_idx, ra_idx = 0, 1, 2, 3
    else:
        # Get indices from mapping
        bf_idx = state_mapping['BF']
        mw_idx = state_mapping['MW']
        ma_idx = state_mapping['MA']
        ra_idx = state_mapping['RA']
    
    # Calculate key cycle metrics
    cycle_metrics = {
        # Individual transition probabilities
        'MW_to_MA': P[mw_idx, ma_idx],           # Detection probability
        'MA_to_RA': P[ma_idx, ra_idx],           # Trigger activation probability
        'RA_to_BF': P[ra_idx, bf_idx],           # Successful return probability
        
        # Failed transitions
        'MA_failed': P[ma_idx, mw_idx],          # Failed awareness
        'RA_failed': P[ra_idx, mw_idx],          # Failed redirection
        
        # Overall cycle properties
        'cycle_completion': P[mw_idx, ma_idx] * P[ma_idx, ra_idx] * P[ra_idx, bf_idx],
        'trigger_efficiency': P[ma_idx, ra_idx] / max(P[ma_idx, :].sum(), 1e-10),
        'detection_efficiency': P[mw_idx, ma_idx] / max(P[mw_idx, :].sum(), 1e-10)
    }
    
    return cycle_metrics

def compare_kmeans_to_glhmm(kmeans, model, network_fields):
    """Compare k-means centers with final GLHMM states"""
    kmeans_centers = kmeans.cluster_centers_
    glhmm_means = np.array([model.get_mean(k) for k in range(model.hyperparameters["K"])])
    
    # Calculate similarity between each k-means center and GLHMM state
    similarity_matrix = np.zeros((kmeans_centers.shape[0], glhmm_means.shape[0]))
    for i in range(kmeans_centers.shape[0]):
        for j in range(glhmm_means.shape[0]):
            # Use cosine similarity
            norm_product = np.linalg.norm(kmeans_centers[i]) * np.linalg.norm(glhmm_means[j])
            if norm_product > 0:
                similarity_matrix[i, j] = np.dot(kmeans_centers[i], glhmm_means[j]) / norm_product
            else:
                similarity_matrix[i, j] = 0
    
    # Find best matching states
    matches = []
    for i in range(kmeans_centers.shape[0]):
        best_match = np.argmax(similarity_matrix[i])
        similarity = similarity_matrix[i, best_match]
        matches.append((i, best_match, similarity))
    
    # Prepare comparison result
    comparison = {
        'kmeans_centers': kmeans_centers,
        'glhmm_means': glhmm_means,
        'similarity_matrix': similarity_matrix,
        'matches': matches
    }
    
    # Print comparison summary
    print("\nK-means to GLHMM state comparison:")
    for kmeans_idx, glhmm_idx, sim in matches:
        print(f"  K-means state {kmeans_idx+1} matches GLHMM state {glhmm_idx+1} with similarity {sim:.4f}")
    
    return comparison


def analyze_state_balance(model, Y, indices):
    """Analyze how well-balanced the states are in terms of occupancy"""
    # Get fractional occupancy (overall and per subject)
    Gamma = model.Gamma
    FO_all = np.mean(Gamma, axis=0)
    FO_subjects = utils.get_FO(Gamma, indices)
    
    # Calculate metrics
    # 1. Overall balance - how evenly distributed are the states?
    entropy = -np.sum(FO_all * np.log2(FO_all + 1e-10))
    max_entropy = -np.log2(1.0/len(FO_all))
    balance_score = entropy / max_entropy  # 1.0 = perfectly balanced
    
    # 2. Per-subject consistency - how consistent is state usage across subjects?
    consistency = np.std(FO_subjects, axis=0)
    avg_consistency = np.mean(consistency)
    
    print("\nState Balance Analysis:")
    print(f"  Overall balance score: {balance_score:.3f} (1.0 = perfectly balanced)")
    print(f"  Average cross-subject consistency: {avg_consistency:.3f} (lower = more consistent)")
    
    # State occupancy per subject
    print("\nState occupancy by subject:")
    for i, fo in enumerate(FO_subjects):
        print(f"  Subject {i+1}: {', '.join([f'State {j+1}: {v:.3f}' for j, v in enumerate(fo)])}")
    
    return {
        'balance_score': balance_score,
        'consistency': consistency,
        'avg_consistency': avg_consistency,
        'FO_all': FO_all,
        'FO_subjects': FO_subjects
    }
    
def main():
    # Load models
    print("Loading time-delay embedded models...")
    model_controls, model_meditators, network_fields = load_time_delay_models()
    
    # Run visualizations
    print("1. Visualizing transition matrices...")
    visualize_transition_matrices(model_controls, model_meditators, PLOT_DIR)
    
    print("2. Plotting network activation profiles...")
    plot_network_profiles(model_controls, model_meditators, network_fields, PLOT_DIR)
    
    # The following functions require the full data (X, Y, indices) which we don't have
    # They can be enabled later if needed by loading the original data
    
    # --- Meditation cycle analysis ---
    print("3. Analyzing meditation cycle patterns...")
    control_mapping = map_states_to_meditation_hierarchy(model_controls, network_fields)
    meditator_mapping = map_states_to_meditation_hierarchy(model_meditators, network_fields)
    
    print("\nControl state mapping:")
    for med_state, hmm_state in control_mapping['state_mapping'].items():
        print(f"  {med_state} → State {hmm_state+1}")
    
    print("\nMeditator state mapping:")
    for med_state, hmm_state in meditator_mapping['state_mapping'].items():
        print(f"  {med_state} → State {hmm_state+1}")
    
    # Analyze cycle metrics
    control_metrics = analyze_meditation_cycle(model_controls, control_mapping['state_mapping'])
    meditator_metrics = analyze_meditation_cycle(model_meditators, meditator_mapping['state_mapping'])
    
    print("\n--- Meditation Cycle Metrics ---")
    print("                   Controls    Meditators")
    for key in ['MW_to_MA', 'MA_to_RA', 'RA_to_BF', 'cycle_completion']:
        print(f"{key:20} {control_metrics[key]:.4f}    {meditator_metrics[key]:.4f}")
    
    print("\nState visualization complete. Plots saved to:", PLOT_DIR)
    
if __name__ == "__main__":
    main()