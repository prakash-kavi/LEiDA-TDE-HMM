"""
Generate Separate Cosine Similarity Heatmaps for State Interactions in Anapanasati Meditation

This script creates two separate 4x4 heatmaps showing cosine similarities between the four states
for meditators and controls (7-network, k=4, by-group standardization). It uses raw cosine similarity
(-1 to 1) with a divergent colormap to handle negative values, ensuring clear visualization.

Input:
- CSV file: G:\leida_hmm\python\results\state_patterns\all_states_activation_patterns.csv

Output:
- Heatmap figure: state_interaction_heatmap_separate_7networks_k4.png
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
CSV_PATH = r'G:\leida_hmm\python\results\state_patterns\all_states_activation_patterns.csv'
OUTPUT_DIR = r'G:\leida_hmm\python\results\visualizations'
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'state_interaction_heatmap_separate_7networks_k4.png')

# State names and dominant networks (from CSV activations)
MEDITATOR_STATES = {
    0: ('Emot. Reg.', 'LIM'),
    1: ('Introspection', 'DMN'),
    2: ('Sust. Attn.', 'VIS'),
    3: ('Sal. Detect.', 'SMN')
}
CONTROL_STATES = {
    0: ('Resting', 'LIM'),
    1: ('Pass. Sens.', 'VIS'),
    2: ('Distr. Detect.', 'VAN'),
    3: ('Unreg. Intro.', 'DMN')
}

# Load CSV data
try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    logger.error(f"CSV file not found at {CSV_PATH}")
    raise

# Filter for 7-network, k=4, by-group standardization
med_df = df[(df['group'] == 'meditators') & 
            (df['networks'] == 7) & 
            (df['k'] == 4) & 
            (df['standardization'] == 'bygroup')]
con_df = df[(df['group'] == 'controls') & 
            (df['networks'] == 7) & 
            (df['k'] == 4) & 
            (df['standardization'] == 'bygroup')]

if med_df.empty or con_df.empty:
    logger.error("Filtered data is empty. Check CSV for 7-network, k=4, by-group standardization.")
    raise ValueError("Empty filtered data")

# Network fields for activation vectors
network_fields = ['VIS', 'SMN', 'DAN', 'VAN', 'LIM', 'FPN', 'DMN']

# State order for radar plot alignment
MEDITATOR_STATE_ORDER = [2, 1, 3, 0]  # Sustained Attention, Introspection, Salience Detection, Emotional Regulation
CONTROL_STATE_ORDER = [1, 3, 2, 0]    # Passive Sensory Processing, Unregulated Introspection, Distraction Detection, Resting State

# Compute cosine similarity matrix with ordered states
def compute_cosine_similarity_matrix(group_df, network_fields, state_order):
    k = 4
    similarity_matrix = np.zeros((k, k))
    activations = group_df.sort_values('state_idx')[network_fields].values
    
    # Validate data
    if np.any(np.isnan(activations)) or np.any(np.isinf(activations)):
        logger.error("Invalid activation values (NaN or Inf) detected")
        raise ValueError("Invalid activation values")
    
    for i_idx, i in enumerate(state_order):
        for j_idx, j in enumerate(state_order):
            norm_i = np.linalg.norm(activations[i])
            norm_j = np.linalg.norm(activations[j])
            if norm_i == 0 or norm_j == 0:
                logger.warning(f"Zero norm for state {i} or {j}, setting similarity to 0")
                similarity = 0
            else:
                similarity = np.dot(activations[i], activations[j]) / (norm_i * norm_j)
                similarity = np.clip(similarity, -1, 1)
            similarity_matrix[i_idx, j_idx] = similarity
    
    return similarity_matrix

# Compute similarity matrices with ordered states
try:
    med_similarity = compute_cosine_similarity_matrix(med_df, network_fields, MEDITATOR_STATE_ORDER)
    con_similarity = compute_cosine_similarity_matrix(con_df, network_fields, CONTROL_STATE_ORDER)
except Exception as e:
    logger.error(f"Error computing similarity matrices: {str(e)}")
    raise

# Create figure with two separate subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 12))

# Meditator heatmap
med_labels = [f"{MEDITATOR_STATES[i][0]}\n({MEDITATOR_STATES[i][1]})" for i in MEDITATOR_STATE_ORDER]
sns.heatmap(med_similarity, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1,
            xticklabels=med_labels, yticklabels=med_labels, ax=ax1,
            cbar_kws={'label': 'Cosine Similarity'}, square=True)
ax1.set_title('Meditators: State Interactions', fontsize=12, pad=10)
ax1.set_xlabel('To State', fontsize=10)
ax1.set_ylabel('From State', fontsize=10)
for tick in ax1.get_xticklabels():
    tick.set_rotation(45)
    tick.set_ha('right')
    tick.set_fontsize(8)
for tick in ax1.get_yticklabels():
    tick.set_rotation(0)
    tick.set_fontsize(8)

# Control heatmap
con_labels = [f"{CONTROL_STATES[i][0]}\n({CONTROL_STATES[i][1]})" for i in CONTROL_STATE_ORDER]
sns.heatmap(con_similarity, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1,
            xticklabels=con_labels, yticklabels=con_labels, ax=ax2,
            cbar_kws={'label': 'Cosine Similarity'}, square=True)
ax2.set_title('Controls: State Interactions', fontsize=12, pad=10)
ax2.set_xlabel('To State', fontsize=10)
ax2.set_ylabel('From State', fontsize=10)
for tick in ax2.get_xticklabels():
    tick.set_rotation(45)
    tick.set_ha('right')
    tick.set_fontsize(8)
for tick in ax2.get_yticklabels():
    tick.set_rotation(0)
    tick.set_fontsize(8)

# Adjust layout
plt.tight_layout(h_pad=2)

# Save and close
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
plt.close()
logger.info(f"Saved heatmap to {OUTPUT_PATH}")