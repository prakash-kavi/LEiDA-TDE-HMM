import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from glhmm import graphics
from ..models.preprocess_tde import visualize_embedded_data, visualize_state_time_course

# Setup paths
ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
TDE_DIR = os.path.join(PROCESSED_DIR, 'tde')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results', 'visualizations', 'tde')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Select which file to visualize
standardize = "global"  # options: global, bygroup, persequence
networks = 8  # options: 7, 8

# Load the processed data
filename = f'tde_{networks}networks_{standardize}.pkl'
filepath = os.path.join(TDE_DIR, filename)
print(f"Loading {filepath}")

with open(filepath, 'rb') as f:
    processed_data = pickle.load(f)

# Load the original data to get raw sequences
with open(os.path.join(TDE_DIR, 'tde_8networks_persequence_old.pkl'), 'rb') as f:
    orig_data = pickle.load(f)

# Create a mock data structure for visualization
data = {
    'controls_sequences': orig_data['original_data']['controls_sequences'],
    'meditators_sequences': orig_data['original_data']['meditators_sequences'],
    'network_fields': processed_data['network_fields']
}

# Create mock group data structures for visualization
controls_tde = {
    'X_preproc': np.vstack(data['controls_sequences']),
    'X_embedded': np.vstack(processed_data['controls_sequences']),
    'indices': processed_data['tde_parameters']['controls_indices_original'],
    'indices_tde': processed_data['tde_parameters']['controls_indices_tde']
}

meditators_tde = {
    'X_preproc': np.vstack(data['meditators_sequences']),
    'X_embedded': np.vstack(processed_data['meditators_sequences']),
    'indices': processed_data['tde_parameters']['meditators_indices_original'],
    'indices_tde': processed_data['tde_parameters']['meditators_indices_tde']
}

# Generate visualizations for selected subjects
print("Generating visualizations...")
for subj_idx in [0, 5, 10, 15]:
    if subj_idx < len(data['controls_sequences']):
        visualize_embedded_data(data, controls_tde, 'controls', subj_idx)
        visualize_state_time_course(data, controls_tde, 'controls', subj_idx)
    
    if subj_idx < len(data['meditators_sequences']):
        visualize_embedded_data(data, meditators_tde, 'meditators', subj_idx)
        visualize_state_time_course(data, meditators_tde, 'meditators', subj_idx)

print(f"Visualizations saved to: {RESULTS_DIR}")