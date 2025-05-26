import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm, Normalize
import networkx as nx
import os, sys

import logging
# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup paths - matching the structure from hierarchical_standardization_reference_k4.py
ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')
REFERENCE_DIR = os.path.join(RESULTS_DIR, 'standardization_reference')
TRANSITIONS_DIR = os.path.join(RESULTS_DIR, 'transitions')
VIS_DIR = os.path.join(TRANSITIONS_DIR, 'visualizations')

# Create directories if they don't exist
os.makedirs(TRANSITIONS_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def visualize_succession_matrices(results):
    """
    Create visualizations of succession matrices for both groups,
    with proper state alignment and network annotations.
    """
    networks = results['networks']
    k = results['k']
    med_to_con_map = results['med_to_con_map']
    
    # Extract matrices
    med_probs = results['succession']['meditators']['probs']
    con_probs = results['succession']['controls']['probs']
    med_mapped_probs = results['succession']['meditators_mapped']['probs']
    diff_probs = results['succession']['differences']
    
    # Extract state information for labels
    med_state_info = results['state_info']['meditators']
    con_state_info = results['state_info']['controls']
    
    # Create figure with 2x2 grid (original matrices and difference matrix)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Custom labels with network information
    med_labels = [f"M{i}\n({med_state_info[i]['dominant_network'].split('_')[0]})" 
                 if i in med_state_info else f"M{i}" for i in range(k)]
    con_labels = [f"C{i}\n({con_state_info[i]['dominant_network'].split('_')[0]})" 
                 if i in con_state_info else f"C{i}" for i in range(k)]
    
    # Map meditator labels to corresponding control states
    med_mapped_labels = [f"M{i}→C{med_to_con_map[i]}" if i in med_to_con_map else f"M{i}" for i in range(k)]
    
    # Plot meditator succession matrix (original)
    sns.heatmap(med_probs, annot=True, fmt='.2f', cmap='viridis', 
               cbar=True, ax=axes[0, 0], 
               xticklabels=med_labels, yticklabels=med_labels,
               vmin=0, vmax=1)
    axes[0, 0].set_title('Meditator State Successions (Original States)')
    axes[0, 0].set_xlabel('To State')
    axes[0, 0].set_ylabel('From State')
    
    # Plot control succession matrix
    sns.heatmap(con_probs, annot=True, fmt='.2f', cmap='viridis', 
               cbar=True, ax=axes[0, 1], 
               xticklabels=con_labels, yticklabels=con_labels,
               vmin=0, vmax=1)
    axes[0, 1].set_title('Control State Successions')
    axes[0, 1].set_xlabel('To State')
    axes[0, 1].set_ylabel('From State')
    
    # Plot meditator mapped succession matrix
    sns.heatmap(med_mapped_probs, annot=True, fmt='.2f', cmap='viridis', 
               cbar=True, ax=axes[1, 0], 
               xticklabels=con_labels, yticklabels=con_labels,
               vmin=0, vmax=1)
    axes[1, 0].set_title('Meditator State Successions (Mapped to Control States)')
    axes[1, 0].set_xlabel('To State')
    axes[1, 0].set_ylabel('From State')
    
    # Plot difference matrix
    sns.heatmap(diff_probs, annot=True, fmt='.2f', cmap='coolwarm', 
               cbar=True, ax=axes[1, 1], 
               xticklabels=con_labels, yticklabels=con_labels,
               vmin=-0.3, vmax=0.3, center=0)
    axes[1, 1].set_title('Succession Differences (Mapped Meditators - Controls)')
    axes[1, 1].set_xlabel('To State')
    axes[1, 1].set_ylabel('From State')
    
    plt.suptitle(f'State Succession Analysis ({networks}-network, k={k})', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save figure
    fig_path = os.path.join(VIS_DIR, f'succession_matrices_{networks}networks_k{k}.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved succession matrices visualization to {fig_path}")

def visualize_transition_graph(results):
    """
    Create network graph visualization of state transitions,
    highlighting the most significant differences between groups.
    """
    networks = results['networks']
    k = results['k']
    
    # Extract matrices and mapping
    med_probs = results['succession']['meditators']['probs']
    con_probs = results['succession']['controls']['probs']
    diff_probs = results['succession']['differences']
    med_to_con_map = results['med_to_con_map']
    
    # Extract state information for node labels
    med_state_info = results['state_info']['meditators']
    con_state_info = results['state_info']['controls']
    
    # Create figure with 1x2 grid
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Generate graphs for both groups
    for idx, (group, probs) in enumerate([('Controls', con_probs), ('Meditators', diff_probs)]):
        ax = axes[idx]
        G = nx.DiGraph()
        
        # Calculate transition magnitude for edge thickness
        max_prob = np.max(probs) if group == 'Controls' else np.max(np.abs(diff_probs))
        
        # Add nodes and edges
        for i in range(k):
            # Add node with state info
            state_info = con_state_info[i] if i in con_state_info else {'dominant_network': 'Unknown'}
            G.add_node(i, network=state_info['dominant_network'])
            
            # Add edges based on transition probabilities
            for j in range(k):
                if group == 'Controls' and probs[i, j] > 0.05:
                    # Edge weight proportional to probability
                    G.add_edge(i, j, weight=probs[i, j], 
                              width=3 * probs[i, j] / max_prob)
                elif group == 'Meditators' and abs(probs[i, j]) > 0.05:
                    # Edge weight proportional to difference magnitude
                    # Positive differences (meditators > controls) in red
                    # Negative differences (controls > meditators) in blue
                    G.add_edge(i, j, weight=abs(probs[i, j]), 
                              width=3 * abs(probs[i, j]) / max_prob,
                              color='red' if probs[i, j] > 0 else 'blue')
        
        # Create layout
        pos = nx.circular_layout(G)
        
        # Draw the graph
        if group == 'Controls':
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, ax=ax, node_size=2000, 
                                  node_color=['lightblue' for _ in range(k)])
            
            # Draw edges
            for u, v, data in G.edges(data=True):
                nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(u, v)], 
                                    width=data['width'] * 3, 
                                    alpha=0.7,
                                    edge_color='navy',
                                    connectionstyle='arc3,rad=0.2',  # Increased curvature
                                    arrowsize=20,  # Larger arrows
                                    arrowstyle='->')  # Explicit arrow style
            
            # Draw labels
            labels = {i: f"C{i}\n({con_state_info[i]['dominant_network'].split('_')[0]})" 
                     if i in con_state_info else f"C{i}" for i in range(k)}
            nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=12)
            
            # Draw edge labels (probabilities)
            edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=10)
            
            ax.set_title(f'Control State Succession Graph', fontsize=14)
        else:
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, ax=ax, node_size=2000, 
                                  node_color=['lightblue' for _ in range(k)])
            
            # Draw edges with different colors for positive and negative differences
            for u, v, data in G.edges(data=True):
                nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(u, v)], 
                                      width=data['width'] * 3, 
                                      alpha=0.7,
                                      edge_color=data['color'],
                                      connectionstyle='arc3,rad=0.1')
            
            # Draw labels
            labels = {i: f"C{i}\n({con_state_info[i]['dominant_network'].split('_')[0]})" 
                     if i in con_state_info else f"C{i}" for i in range(k)}
            nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=12)
            
            # Draw edge labels (differences)
            edge_labels = {(u, v): f"{diff_probs[u, v]:.2f}" for u, v, d in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=10)
            
            ax.set_title(f'Meditation Effects on State Successions\n(Red = Meditators > Controls, Blue = Controls > Meditators)', fontsize=14)
        
        ax.set_axis_off()
    
    plt.suptitle(f'State Transition Graphs ({networks}-network, k={k})', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure
    fig_path = os.path.join(VIS_DIR, f'transition_graph_{networks}networks_k{k}.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved transition graph visualization to {fig_path}")
    
def calculate_state_durations(blocks, k):
    """Calculate average duration for each state."""
    state_durations = [[] for _ in range(k)]
    
    for block in blocks:
        state = block['state']
        if state < k:
            state_durations[state].append(block['length'])
    
    # Calculate average durations
    avg_durations = np.zeros(k)
    for s in range(k):
        if state_durations[s]:
            avg_durations[s] = np.mean(state_durations[s])
    
    return avg_durations
    
def visualize_state_durations(results):
    """
    Visualize state durations for both groups, with aligned states.
    """
    networks = results['networks']
    k = results['k']
    med_to_con_map = results['med_to_con_map']
    
    # Extract state information
    med_state_info = results['state_info']['meditators']
    con_state_info = results['state_info']['controls']
    
    # Calculate state durations
    med_blocks = results['blocks']['meditators']
    con_blocks = results['blocks']['controls']
    
    med_durations = calculate_state_durations(med_blocks, k)
    con_durations = calculate_state_durations(con_blocks, k)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set up bar positions
    x = np.arange(k)
    width = 0.35
    
    # Create bars
    con_bars = ax.bar(x - width/2, con_durations, width, label='Controls')
    
    # Align meditator states with controls
    med_aligned_durations = np.zeros(k)
    for med_state, con_state in med_to_con_map.items():
        med_aligned_durations[con_state] = med_durations[med_state]
    
    med_bars = ax.bar(x + width/2, med_aligned_durations, width, label='Meditators (aligned)')
    
    # Add labels and styling
    ax.set_xlabel('Control State')
    ax.set_ylabel('Average Duration (TRs)')
    ax.set_title(f'State Durations Comparison ({networks}-network, k={k})')
    ax.set_xticks(x)
    
    # Create custom labels with network information
    labels = [f"C{i}\n({con_state_info[i]['dominant_network'].split('_')[0]})" 
             if i in con_state_info else f"C{i}" for i in range(k)]
    ax.set_xticklabels(labels)
    
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(VIS_DIR, f'state_durations_{networks}networks_k{k}.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved state durations visualization to {fig_path}")
    
def debug_visualization_data(results):
    """Debug data issues causing empty plots."""
    # Check for NaN values
    matrices = [
        ('med_probs', results['succession']['meditators']['probs']),
        ('con_probs', results['succession']['controls']['probs']),
        ('diff_probs', results['succession']['differences'])
    ]
    
    for name, matrix in matrices:
        logger.info(f"{name} shape: {matrix.shape}")
        logger.info(f"{name} contains NaN: {np.isnan(matrix).any()}")
        logger.info(f"{name} non-zero elements: {np.count_nonzero(matrix)}")
        
    # Check state mapping completeness
    k = results['k']
    mapping = results['med_to_con_map']
    logger.info(f"Mapping covers {len(mapping)}/{k} states")
    
    # Log the actual values
    logger.info(f"First few values of diff_probs:\n{results['succession']['differences'][:2,:2]}")
    
