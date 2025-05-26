"""
Meditation State Visualizations

This script creates specialized visualizations mapping HMM-derived brain states
to Anapanasati meditation stages, focusing on:
1. Circular transition graphs with meditation stage labels
2. State occupancy comparisons between meditators and controls
3. Network activation signatures relevant to meditation
4. Meditation cycle visualization

The visualizations highlight how brain dynamics during meditation reflect the
traditional stages of Anapanasati practice.
"""

import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
import logging
from matplotlib.patches import Patch

# Setup paths
ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
TRANSITIONS_DIR = os.path.join(RESULTS_DIR, 'transitions')
STATE_PATTERNS_DIR = os.path.join(RESULTS_DIR, 'state_patterns')
MEDITATION_DIR = os.path.join(RESULTS_DIR, 'meditation_mapping')
VIS_DIR = os.path.join(MEDITATION_DIR, 'visualizations_May_14')

# Create directories if they don't exist
os.makedirs(VIS_DIR, exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define Anapanasati stages
ANAPANASATI_STAGES = {
    0: "Breath Body Awareness\n(Focused attention on breath sensations)",
    1: "Mind Wandering Awareness\n(Noticing & redirecting attention)",
    2: "Positive Experience\n(Pleasant sensations during focus)",
    3: "Physical Relaxation\n(Calming bodily tension)"
}

# Define non-meditation interpretations for controls
CONTROL_INTERPRETATIONS = {
    0: "Involuntary Attention Capture",
    1: "Task-Oriented Problem Solving",
    2: "Passive Visual Processing",
    3: "Emotional Processing"
}
# Define Anapanasati stages for meditators
MEDITATOR_STATE_LABELS = {
    0: "Emotional Regulation\n(Non-reactive breath observation)",
    1: "Introspection\n(Spontaneous thought)",
    2: "Sustained Attention\n(Focused breath awareness)",
    3: "Salience Detection\n(Noticing distractions)"
}

# Define non-meditation interpretations for controls
CONTROL_STATE_LABELS = {
    0: "Resting State\n(Low engagement)",
    1: "Passive Sensory Processing\n(Untrained sensory focus)",
    2: "Unregulated Introspection\n(Diffuse thought)",
    3: "Distraction Detection\n(Reactive salience)"
}

# Network colors for radar plots
NETWORK_COLORS = {
    'VIS': '#A153A2',   # Purple
    'SMN': '#6FABD2',   # Light Blue
    'DAN': '#2C8B4B',   # Green
    'VAN': '#B77FB4',   # Pink
    'LIM': '#E7B013',   # Yellow
    'FPN': '#E58429',   # Orange
    'DMN': '#CA3542'    # Red
}

def load_transition_data(networks=7, k=4):
    """Load transition data from pickle file."""
    transition_path = os.path.join(TRANSITIONS_DIR, f'aligned_transitions_{networks}networks_k{k}.pkl')
    
    if not os.path.exists(transition_path):
        logger.warning(f"Transition file not found: {transition_path}")
        return None
    
    with open(transition_path, 'rb') as f:
        transition_data = pickle.load(f)
    
    logger.info(f"Loaded transition data for {networks}-network, k={k}")
    return transition_data

def load_state_patterns(networks=7, k=4):
    """Load state activation patterns from CSV file."""
    patterns_path = os.path.join(STATE_PATTERNS_DIR, 'all_states_activation_patterns.csv')
    
    if not os.path.exists(patterns_path):
        logger.warning(f"State patterns file not found: {patterns_path}")
        return None
    
    patterns_df = pd.read_csv(patterns_path)
    
    # Filter to relevant configurations
    filtered_df = patterns_df[(patterns_df['networks'] == networks) & 
                             (patterns_df['k'] == k) &
                             (patterns_df['standardization'] == 'bygroup')]
    
    logger.info(f"Loaded state patterns for {networks}-network, k={k}")
    return filtered_df

def create_meditation_circular_graph(transition_data, group, save_path=None):
    """
    Create circular graph visualization of state transitions for a specific group,
    with states labeled according to neuroscience-aligned meditation stages.
    """
    k = transition_data['k']
    networks = transition_data['networks']
    
    # Determine which succession matrix and labels to use
    if group == 'meditators':
        probs = transition_data['succession']['meditators']['probs']
        state_info = transition_data['state_info']['meditators']
        state_mapping = MEDITATOR_STATE_LABELS
    else:
        probs = transition_data['succession']['controls']['probs']
        state_info = transition_data['state_info']['controls']
        state_mapping = CONTROL_STATE_LABELS
        
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes and edges
    for i in range(k):
        # Get dominant network for node label
        network = state_info[i]['dominant_network'].split('_')[0] if i in state_info else f"S{i}"
        # Add node with attributes
        G.add_node(i, 
                   network=network, 
                   meditation_stage=state_mapping[i])
        
        # Add edges
        for j in range(k):
            if probs[i, j] > 0:  # Only show edges above threshold
                G.add_edge(i, j, weight=probs[i, j], width=max(1, 8 * probs[i, j]))
    
    # Create figure
    plt.figure(figsize=(14, 14))
    
    # Create layout
    pos = nx.circular_layout(G)
    
    # Calculate node sizes based on FO with logarithmic scaling
    if group == 'meditators':
        fo_values = [transition_data['succession']['meditators']['counts'][i].sum() for i in range(k)]
    else:
        fo_values = [transition_data['succession']['controls']['counts'][i].sum() for i in range(k)]
    
    total = sum(fo_values)
    # Logarithmic scaling for node sizes (base size + scaled FO)
    node_sizes = [3000 * (np.log1p(v / total) / np.log1p(1)) for v in fo_values]
    
    # Define node colors based on network
    node_colors = [NETWORK_COLORS.get(G.nodes[n]['network'], '#CCCCCC') for n in G.nodes()]
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, 
                          alpha=0.8, edgecolors='black', linewidths=2)
    
    # Draw edges with varying thickness
    for u, v, data in G.edges(data=True):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=data['width'],
                              alpha=min(0.9, max(0.3, data['weight'])),
                              edge_color='navy',
                              connectionstyle='arc3,rad=0.2',
                              arrowsize=20)
    
    # Prepare custom labels with revised state names
    labels = {}
    for node in G.nodes():
        meditation_stage = G.nodes[node]['meditation_stage']
        network = G.nodes[node]['network']
        labels[node] = f"State {node}\n({network})\n\n{meditation_stage}"
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=12, font_family='sans-serif',
                           verticalalignment='center', horizontalalignment='center',
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.6'))
    
    # Add title
    plt.title(f"{group.capitalize()} State Transitions\n{networks}-network, k={k}", 
             fontsize=18, pad=20)
    
    # Add legend for node sizes
    handles = []
    for i in range(k):
        percent = 100 * (fo_values[i] / total)
        handles.append(Patch(color=node_colors[i], 
                            label=f"State {i} ({G.nodes[i]['network']}): {percent:.1f}%"))
    
    plt.legend(handles=handles, title="Fractional Occupancy", 
              loc='upper center', bbox_to_anchor=(0.5, -0.05),
              ncol=2, fancybox=True)
    
    plt.axis('off')
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved meditation circular graph to {save_path}")
    
    plt.close()

def visualize_state_occupancy(transition_data, patterns_df, save_path=None):
    """
    Visualize fractional occupancy for matched states (Sustained Attention,
    Introspection, Salience Detection, Emotional Regulation), with meditators
    and controls side-by-side using consistent colors from radar plot.
    """
    k = transition_data['k']
    networks = transition_data['networks']

    # Define consistent state order and labels
    conceptual_order = [2, 1, 3, 0]  # Sustained Attention, Introspection, Salience Detection, Emotional Regulation

    # State color mapping
    STATE_COLORS = {
        0: '#1f77b4',  # Emotional Regulation (blue)
        1: '#ff7f0e',  # Introspection (orange)
        2: '#2ca02c',  # Sustained Attention (green)
        3: '#d62728'   # Salience Detection (red)
    }

    # Control to meditator state mapping
    control_state_map = {1: 1, 2: 2, 3: 3, 0: 0}
    inverse_control_map = {v: k for k, v in control_state_map.items()}

    # Extract FO values for both groups
    med_fo = {row['state_idx']: row['fractional_occupancy'] 
              for _, row in patterns_df[patterns_df['group'] == 'meditators'].iterrows()}
    con_fo = {row['state_idx']: row['fractional_occupancy'] 
              for _, row in patterns_df[patterns_df['group'] == 'controls'].iterrows()}

    # Placeholder SE values (ideally use actual per-subject stats)
    med_se = {i: 0.02 for i in range(k)}
    con_se = {i: 0.03 for i in range(k)}

    # Prepare reordered values
    labels = []
    med_values, con_values = [], []
    med_errors, con_errors = [], []

    for concept_state in conceptual_order:
        med_idx = concept_state
        con_idx = inverse_control_map[concept_state]

        med_values.append(med_fo.get(med_idx, 0))
        con_values.append(con_fo.get(con_idx, 0))
        med_errors.append(med_se.get(med_idx, 0))
        con_errors.append(con_se.get(con_idx, 0))
        labels.append(MEDITATOR_STATE_LABELS[concept_state].splitlines()[0])

    # Plot settings
    x = np.arange(len(conceptual_order))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot bars
    ax.bar(x - width/2, med_values, width, yerr=med_errors, 
           label='Meditators', color=[STATE_COLORS[i] for i in conceptual_order],
           alpha=0.8, edgecolor='black', linewidth=1, capsize=4)

    ax.bar(x + width/2, con_values, width, yerr=con_errors, 
           label='Controls', color=[STATE_COLORS[i] for i in conceptual_order],
           alpha=0.4, edgecolor='black', linewidth=1, capsize=4, hatch='//')

    # Axis labeling
    ax.set_ylabel('Fractional Occupancy', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=11)
    ax.set_title(f'State Occupancy by Group ({networks}-network, k={k})', fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved state occupancy visualization to {save_path}")

    plt.close()

def visualize_network_radar(patterns_df, save_path=None):
    """
    Create radar plots showing network activation patterns for each state,
    ordered as Sustained Attention -> Introspection -> Salience Detection ->
    Emotional Regulation, with state-specific colors, group comparison overlays,
    and distinct fill patterns.
    """
    k = patterns_df['k'].iloc[0]
    networks_count = patterns_df['networks'].iloc[0]
    
    # Get network names
    network_names = ['VIS', 'SMN', 'DAN', 'VAN', 'LIM', 'FPN', 'DMN']
    if networks_count == 8:
        network_names.append('SUB')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, k, figsize=(16, 10), subplot_kw=dict(polar=True))
    
    # Flatten axes for easier iteration
    if k > 1:
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes[0], axes[1]]
    
    # Set angles for radar plot
    angles = np.linspace(0, 2*np.pi, len(network_names), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    # Define state-specific colors
    STATE_COLORS = {
        0: '#1f77b4',  # Emotional Regulation (blue)
        1: '#ff7f0e',  # Introspection (orange)
        2: '#2ca02c',  # Sustained Attention (green)
        3: '#d62728'   # Salience Detection (red)
    }
    
    # Define state order for meditators and controls
    state_order = [2, 1, 3, 0]  # Sustained Attention, Introspection, Salience Detection, Emotional Regulation
    control_order = [1, 2, 3, 0]  # Passive Sensory Processing, Unregulated Introspection, Distraction Detection, Resting State
    
    # Map control states to meditator states for color consistency
    control_state_map = {1: 2, 2: 1, 3: 3, 0: 0}  # Control State: Meditator State
    
    # Process each group
    for row, group in enumerate(['meditators', 'controls']):
        group_df = patterns_df[patterns_df['group'] == group]
        other_group = 'controls' if group == 'meditators' else 'meditators'
        other_group_df = patterns_df[patterns_df['group'] == other_group]
        
        # Process each state in specified order
        order = state_order if group == 'meditators' else control_order
        for idx, state in enumerate(order):
            ax_idx = row * k + idx
            ax = axes_flat[ax_idx]
            
            state_data = group_df[group_df['state_idx'] == state]
            if state_data.empty:
                continue
                
            # Get network activations
            values = [state_data[net].iloc[0] for net in network_names]
            
            # Get corresponding state from other group
            other_state_idx = control_order[idx] if group == 'meditators' else state_order[idx]
            other_state_data = other_group_df[other_group_df['state_idx'] == other_state_idx]
            other_values = [other_state_data[net].iloc[0] for net in network_names] if not other_state_data.empty else [0] * len(network_names)
            
            # Close polygons for plotting
            values_plot = values + [values[0]]
            other_values_plot = other_values + [other_values[0]]
            
            # Use state-specific color (correct indexing for controls)
            color_idx = state if group == 'meditators' else control_state_map[state]
            ax.plot(angles, values_plot, linewidth=2, linestyle='solid', 
                   color=STATE_COLORS[color_idx])
            ax.fill(angles, values_plot, alpha=0.25 if group == 'meditators' else 0.15,
                   color=STATE_COLORS[color_idx],
                   hatch='' if group == 'meditators' else '//')
            
            # Plot faint overlay for other group
            other_color_idx = state if group == 'controls' else control_state_map[other_state_idx]
            ax.plot(angles, other_values_plot, linewidth=1, linestyle='dashed', 
                   color=STATE_COLORS[other_color_idx], alpha=0.5)
            
            # Set network labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(network_names, fontsize=8)
            
            # Set y-axis limits
            ax.set_ylim(-1, 1)
            ax.set_yticks([-1, -0.5, 0, 0.5, 1])
            ax.set_yticklabels(['-1', '-0.5', '0', '0.5', '1'], fontsize=7)
            
            # Add title with revised state names only
            if row == 0:
                title = MEDITATOR_STATE_LABELS[state].splitlines()[0]
                ax.set_title(title, fontsize=11, pad=15)
            else:
                title = CONTROL_STATE_LABELS[state].splitlines()[0]
                ax.set_title(title, fontsize=11, pad=15)
            
            # Highlight zero line
            ax.plot(angles, [0]*len(angles), color='gray', linestyle='--', alpha=0.5)
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.5)
    
    # Add row labels
    fig.text(0.01, 0.75, 'Meditators', fontsize=14, rotation=90, ha='center', va='center')
    fig.text(0.01, 0.25, 'Controls', fontsize=14, rotation=90, ha='center', va='center')
    
    plt.tight_layout(rect=[0.03, 0, 1, 0.95])
    plt.suptitle(f'Network Activation Profiles for Meditation States', fontsize=16, y=0.98)
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved network radar visualization to {save_path}")
    
    plt.close()
    
def visualize_meditation_cycle(transition_data, save_path=None):
    """
    Visualize the Anapanasati meditation cycle with state transitions for meditators
    vs controls, using revised state names, styled arrows, distinct node shapes,
    and emphasized cycle path.
    """
    k = transition_data['k']
    networks = transition_data['networks']
    
    # Extract transition probabilities
    med_probs = transition_data['succession']['meditators']['probs']
    con_probs = transition_data['succession']['controls']['probs']
    
    # Get state info
    med_state_info = transition_data['state_info']['meditators']
    con_state_info = transition_data['state_info']['controls']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define positions for states in a cycle
    positions = {
        0: (0.2, 0.5),    # Left
        1: (0.5, 0.8),    # Top
        2: (0.8, 0.5),    # Right
        3: (0.5, 0.2)     # Bottom
    }
    
    # Define colors and FO for node sizes
    state_colors = {
        0: NETWORK_COLORS.get(med_state_info[0]['dominant_network'].split('_')[0], '#CCCCCC'),
        1: NETWORK_COLORS.get(med_state_info[1]['dominant_network'].split('_')[0], '#CCCCCC'),
        2: NETWORK_COLORS.get(med_state_info[2]['dominant_network'].split('_')[0], '#CCCCCC'),
        3: NETWORK_COLORS.get(med_state_info[3]['dominant_network'].split('_')[0], '#CCCCCC')
    }
    
    # Calculate FO for node sizes (log-scaled)
    fo_values = [transition_data['succession']['meditators']['counts'][i].sum() for i in range(k)]
    total = sum(fo_values)
    node_sizes = [1000 * (np.log1p(v / total) / np.log1p(1)) for v in fo_values]
    
    # Draw nodes with distinct shapes
    for state in range(k):
        x, y = positions[state]
        
        # Draw node (circle for meditators, as primary focus)
        shape = 'o'  # Fixed to circles for meditators’ states
        marker = plt.scatter(x, y, s=node_sizes[state], c=state_colors[state], 
                            alpha=0.8, edgecolors='black', linewidths=2, marker=shape)
        
        # Add state label
        med_network = med_state_info[state]['dominant_network'].split('_')[0] if state in med_state_info else "?"
        anapana_stage = MEDITATOR_STATE_LABELS[state].split('\n')[0]
        
        ax.text(x, y, f"S{state}\n({med_network})", 
            ha='center', va='center', fontsize=12, fontweight='bold',
            color='white', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
        
        ax.text(x, y-0.15, anapana_stage, 
            ha='center', va='center', fontsize=10, wrap=True,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    # Draw bold cycle path for meditators
    cycle_path = [(0, 2), (2, 1), (1, 3), (3, 0)]  # Emotional Regulation -> Sustained Attention -> Introspection -> Salience Detection -> Emotional Regulation
    for (i, j) in cycle_path:
        x1, y1 = positions[i]
        x2, y2 = positions[j]
        dx = x2 - x1
        dy = y2 - y1
        norm = np.sqrt(dx**2 + dy**2)
        dx = dx / norm
        dy = dy / norm
        start_x = x1 + dx * 0.1
        start_y = y1 + dy * 0.1
        end_x = x2 - dx * 0.1
        end_y = y2 - dy * 0.1
        ax.plot([start_x, end_x], [start_y, end_y], color='black', linewidth=3, alpha=0.2, zorder=0)
    
    # Draw arrows for transitions
    for i in range(k):
        for j in range(k):
            if i != j:  # Don't draw self-transitions
                # Get transition probabilities
                med_prob = med_probs[i, j]
                con_prob = con_probs[i, j]
                diff = med_prob - con_prob
                
                # Only draw arrows for probabilities above threshold
                if abs(med_prob) > 0.1 or abs(con_prob) > 0.1:
                    # Get positions
                    x1, y1 = positions[i]
                    x2, y2 = positions[j]
                    
                    # Calculate midpoint with offset for curved arrows
                    mid_x = (x1 + x2) / 2
                    mid_y = (y1 + y2) / 2
                    
                    # Add offset for curved arrow
                    dx = x2 - x1
                    dy = y2 - y1
                    offset = 0.1
                    offset_x = -dy * offset
                    offset_y = dx * offset
                    ctrl_x = mid_x + offset_x
                    ctrl_y = mid_y + offset_y
                    
                    # Calculate arrow direction vectors
                    norm = np.sqrt(dx**2 + dy**2)
                    dx = dx / norm
                    dy = dy / norm
                    
                    # Adjust start and end points
                    start_x = x1 + dx * 0.1
                    start_y = y1 + dy * 0.1
                    end_x = x2 - dx * 0.1
                    end_y = y2 - dy * 0.1
                    
                    # Choose color and style based on difference
                    if diff > 0:
                        color = 'red'  # Meditators > Controls
                        linestyle = 'solid'
                    else:
                        color = 'blue'  # Controls > Meditators
                        linestyle = 'dashed'
                    
                    # Draw arrow with thickness proportional to |diff|
                    arrow = ax.annotate("", 
                                      xy=(end_x, end_y), 
                                      xytext=(start_x, start_y),
                                      arrowprops=dict(
                                          arrowstyle="->",
                                          connectionstyle=f"arc3,rad={offset}",
                                          color=color,
                                          linestyle=linestyle,
                                          lw=max(1, 4 * abs(diff)),
                                          alpha=min(0.9, max(0.3, abs(diff)))
                                      ))
                    
                    # Add transition probability label
                    label = f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}"
                    ax.text(ctrl_x, ctrl_y, label, 
                           ha='center', va='center', fontsize=10, fontweight='bold',
                           color=color, 
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    # Add legend
    legend_elements = [
        Patch(facecolor='red', edgecolor='black', label='Meditators > Controls'),
        Patch(facecolor='blue', edgecolor='black', label='Controls > Meditators')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.05),
             ncol=2, fancybox=True)
    
    # Add title
    ax.set_title(f'Anapanasati Meditation Cycle: State Transitions\n{networks}-network, k={k}', 
                fontsize=16)
    
    # Set axis properties
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved meditation cycle visualization to {save_path}")
    
    plt.close()

def main():
    """Main function to create all meditation-focused visualizations."""
    logger.info("Starting meditation state visualization creation")
    
    # Set parameters
    networks = 7  # Using 7-network configuration
    k = 4         # Using k=4 states
    
    # Load data
    transition_data = load_transition_data(networks, k)
    pattern_data = load_state_patterns(networks, k)
    
    if transition_data is None or pattern_data is None:
        logger.error("Failed to load required data")
        return
    
    # Create visualizations
    
    # 1. Meditation-Stage Circular Graph for each group
    # create_meditation_circular_graph(
    #     transition_data, 
    #     'meditators',
    #     os.path.join(VIS_DIR, f'meditation_graph_meditators_{networks}networks_k{k}.png')
    # )
    
    # create_meditation_circular_graph(
    #     transition_data, 
    #     'controls',
    #     os.path.join(VIS_DIR, f'meditation_graph_controls_{networks}networks_k{k}.png')
    # )
    
    # 2. State Occupancy by Expertise
    visualize_state_occupancy(
        transition_data,
        pattern_data,
        os.path.join(VIS_DIR, f'state_occupancy_{networks}networks_k{k}.png')
    )
    
    #3. Network Activation Signatures
    visualize_network_radar(
        pattern_data,
        os.path.join(VIS_DIR, f'network_radar_{networks}networks_k{k}.png')
    )
    
    # 4. Meditation Cycle Visualization
    # visualize_meditation_cycle(
    #     transition_data,
    #     os.path.join(VIS_DIR, f'meditation_cycle_{networks}networks_k{k}.png')
    # )
    
    logger.info("Meditation state visualizations complete")

if __name__ == "__main__":
    main()