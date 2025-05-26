import numpy as np
import nibabel as nib
from nilearn import plotting, datasets, surface
from nilearn.regions import RegionExtractor
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from collections import OrderedDict

# Your provided mapping function
def define_network_mappings():
    """
    Define the 8 networks (7 Yeo + Subcortical)
    """
    network_fields = ['VIS', 'SMN', 'DAN', 'VAN', 'LIM', 'FPN', 'DMN', 'SUB']
    networks = OrderedDict()
    
    networks['VIS'] = {'name': 'Visual Network', 'rois': [i-1 for i in list(range(1, 10)) + list(range(59, 67))]}
    networks['SMN'] = {'name': 'Somatomotor Network', 'rois': [i-1 for i in list(range(10, 16)) + list(range(67, 75))]}
    networks['DAN'] = {'name': 'Dorsal Attention Network', 'rois': [i-1 for i in list(range(16, 24)) + list(range(75, 82))]}
    networks['VAN'] = {'name': 'Ventral Attention Network', 'rois': [i-1 for i in list(range(24, 31)) + list(range(82, 87))]}
    networks['LIM'] = {'name': 'Limbic Network', 'rois': [i-1 for i in list(range(31, 34)) + list(range(87, 89))]}
    networks['FPN'] = {'name': 'Frontoparietal Network', 'rois': [i-1 for i in list(range(34, 38)) + list(range(89, 98))]}
    networks['DMN'] = {'name': 'Default Mode Network', 'rois': [i-1 for i in list(range(38, 51)) + list(range(98, 109))]}
    networks['SUB'] = {'name': 'Subcortical Regions', 'rois': [i-1 for i in list(range(51, 59)) + list(range(109, 117))]}
    
    return networks, network_fields

# Load Schaefer 116 parcellation (replace with your file path)
schaefer_atlas = nib.load('Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.nii.gz')
schaefer_data = schaefer_atlas.get_fdata()

# Load state activation data (example: replace with your model.pkl)
with open('model_k4.pkl', 'rb') as f:
    k4_data = pickle.load(f)  # Shape: (4, 116) for k=4 states
with open('model_k5.pkl', 'rb') as f:
    k5_data = pickle.load(f)  # Shape: (5, 116) for k=5 states
with open('model_meditators.pkl', 'rb') as f:
    meditator_data = pickle.load(f)  # Shape: (n_subjects, n_states, 116)
with open('model_controls.pkl', 'rb') as f:
    control_data = pickle.load(f)  # Shape: (n_subjects, n_states, 116)
with open('model_experts.pkl', 'rb') as f:
    expert_data = pickle.load(f)  # Shape: (5, n_states, 116)

# Example succession probabilities (replace with your data)
succession_probs = {
    (1, 2): 0.3, (2, 3): 0.4, (3, 1): 0.25, (1, 4): 0.15
}  # e.g., P(2|1)=0.3

# Get network mappings
networks, network_fields = define_network_mappings()

# Fetch fsaverage surface for cortical rendering
fsaverage = datasets.fetch_surf_fsaverage()

# Function to create NIfTI image from state activation
def create_nifti_from_state(state_vector, atlas_data, affine):
    nifti_data = np.zeros_like(atlas_data)
    for i, val in enumerate(state_vector):
        nifti_data[atlas_data == (i+1)] = val  # Map activation to parcel
    return nib.Nifti1Image(nifti_data, affine)

# 1. Network-Level State Maps (k=4 and k=5)
for k, data in [(4, k4_data), (5, k5_data)]:
    fig = make_subplots(rows=1, cols=k, subplot_titles=[f'State {i+1}' for i in range(k)],
                        specs=[[{'type': 'surface'}]*k])
    for state_idx in range(k):
        state_vector = data[state_idx]  # Shape: (116,)
        nifti_img = create_nifti_from_state(state_vector, schaefer_data, schaefer_atlas.affine)
        
        # Plot cortical surface
        surf_map = surface.load_surf_data(fsaverage['pial_left'])  # Placeholder; map state_vector to surface
        plot = plotting.plot_surf_stat_map(
            fsaverage['pial_left'], surf_map, hemi='left', cmap='RdBu_r', threshold=0.1,
            bg_map=fsaverage['sulc_left'], output_file=None
        )
        fig.add_trace(plot.data[0], row=1, col=state_idx+1)
    
    fig.update_layout(title=f'Network-Level State Maps (k={k})', showlegend=False)
    fig.write_html(f'state_maps_k{k}.html')

# 2. Group Comparison Visualizations (Meditators vs. Controls)
for state_idx in range(4):  # Example for k=4
    meditator_mean = meditator_data[:, state_idx, :].mean(axis=0)  # Shape: (116,)
    control_mean = control_data[:, state_idx, :].mean(axis=0)  # Shape: (116,)
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Meditators', 'Controls'],
                        specs=[[{'type': 'surface'}, {'type': 'surface'}]])
    
    for group, data in [('Meditators', meditator_mean), ('Controls', control_mean)]:
        nifti_img = create_nifti_from_state(data, schaefer_data, schaefer_atlas.affine)
        surf_map = surface.load_surf_data(fsaverage['pial_left'])  # Placeholder
        plot = plotting.plot_surf_stat_map(
            fsaverage['pial_left'], surf_map, hemi='left', cmap='RdBu_r', threshold=0.1,
            bg_map=fsaverage['sulc_left'], output_file=None
        )
        col = 1 if group == 'Meditators' else 2
        fig.add_trace(plot.data[0], row=1, col=col)
    
    fig.update_layout(title=f'Group Comparison: State {state_idx+1}')
    fig.write_html(f'group_comparison_state{state_idx+1}.html')

# 3. Expert Meditator State Patterns
fig = make_subplots(rows=1, cols=5, subplot_titles=[f'Expert {i+1}' for i in range(5)],
                    specs=[[{'type': 'surface'}]*5])
for expert_idx in range(5):
    state_vector = expert_data[expert_idx, 0, :]  # Example: first state
    nifti_img = create_nifti_from_state(state_vector, schaefer_data, schaefer_atlas.affine)
    surf_map = surface.load_surf_data(fsaverage['pial_left'])  # Placeholder
    plot = plotting.plot_surf_stat_map(
        fsaverage['pial_left'], surf_map, hemi='left', cmap='RdBu_r', threshold=0.1,
        bg_map=fsaverage['sulc_left'], output_file=None
    )
    fig.add_trace(plot.data[0], row=1, col=expert_idx+1)

fig.update_layout(title='Expert Meditator State Patterns')
fig.write_html('expert_meditator_patterns.html')

# 4. State Succession Visualization
sequence = [(1, 2), (2, 3), (3, 1)]  # Example sequence
fig = plt.figure(figsize=(15, 5))
for idx, (current, next_state) in enumerate(sequence):
    state_vector = k4_data[current-1]  # State numbering starts at 1
    nifti_img = create_nifti_from_state(state_vector, schaefer_data, schaefer_atlas.affine)
    ax = fig.add_subplot(1, len(sequence), idx+1, projection='3d')
    plotting.plot_surf_stat_map(
        fsaverage['pial_left'], surf_map, hemi='left', cmap='RdBu_r', threshold=0.1,
        bg_map=fsaverage['sulc_left'], axes=ax
    )
    ax.set_title(f'State {current}')
    if idx < len(sequence)-1:
        plt.annotate('', xy=(0.95, 0.5), xytext=(1.05, 0.5),
                     xycoords='axes fraction', textcoords='axes fraction',
                     arrowprops=dict(arrowstyle='->', color='black'))
        plt.text(1.0, 0.55, f'P={succession_probs.get((current, next_state), 0):.2f}',
                 ha='center', va='center', transform=ax.transAxes)

plt.suptitle('State Succession Pattern')
plt.savefig('state_succession.png', dpi=300, bbox_inches='tight')
plt.close()

print("Visualizations saved as HTML and PNG files.")