import os
import pickle
import numpy as np
import pandas as pd

# Setup paths
ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
METRICS_DIR = os.path.join(ROOT_DIR, 'results', 'metrics')

def examine_metrics_structure(group, networks, k):
    """Examine the detailed structure of a metrics file."""
    metrics_path = os.path.join(METRICS_DIR, f'{networks}networks', group, f'k{k}_metrics.pkl')
    
    if not os.path.exists(metrics_path):
        print(f"File not found: {metrics_path}")
        return
    
    print(f"\n=== Detailed Structure: {group}, {networks}-network, k={k} ===")
    
    # Load the data
    with open(metrics_path, 'rb') as f:
        metrics_data = pickle.load(f)
    
    # Print basic info
    print(f"File size: {os.path.getsize(metrics_path)} bytes")
    print(f"Top-level keys: {list(metrics_data.keys())}")
    
    # Detailed examination of temporal metrics
    if 'temporal_metrics' in metrics_data:
        temp = metrics_data['temporal_metrics']
        print("\nTEMPORAL METRICS:")
        print(f"  Keys: {list(temp.keys())}")
        
        # Check main arrays
        for key in ['fractional_occupancy', 'lifetimes', 'state_intervals']:
            if key in temp:
                print(f"  {key}: type={type(temp[key])}, ", end="")
                if isinstance(temp[key], list):
                    print(f"length={len(temp[key])}")
                    if len(temp[key]) > 0:
                        print(f"    First item type: {type(temp[key][0])}")
                        if isinstance(temp[key][0], (list, np.ndarray)):
                            print(f"    First item length/shape: {len(temp[key][0]) if isinstance(temp[key][0], list) else temp[key][0].shape}")
                elif isinstance(temp[key], np.ndarray):
                    print(f"shape={temp[key].shape}")
    
    # Detailed examination of transition metrics
    if 'transition_metrics' in metrics_data:
        trans = metrics_data['transition_metrics']
        print("\nTRANSITION METRICS:")
        print(f"  Keys: {list(trans.keys())}")
        
        # Check main arrays
        for key in ['transition_matrices', 'succession_matrices']:
            if key in trans:
                print(f"  {key}: type={type(trans[key])}, ", end="")
                if isinstance(trans[key], list):
                    print(f"length={len(trans[key])}")
                    if len(trans[key]) > 0:
                        print(f"    First item type: {type(trans[key][0])}")
                        if isinstance(trans[key][0], (np.ndarray)):
                            print(f"    First item shape: {trans[key][0].shape}")
                elif isinstance(trans[key], np.ndarray):
                    print(f"shape={trans[key].shape}")
    
    # Detailed examination of network interactions
    if 'network_interactions' in metrics_data:
        net = metrics_data['network_interactions']
        print("\nNETWORK INTERACTIONS:")
        print(f"  Keys: {list(net.keys())}")
        
        # Check network means
        if 'state_network_means' in net:
            means = net['state_network_means']
            print(f"  state_network_means: type={type(means)}, ", end="")
            if isinstance(means, list):
                print(f"length={len(means)}")
                if len(means) > 0:
                    print(f"    First item type: {type(means[0])}")
                    if isinstance(means[0], np.ndarray):
                        print(f"    First item shape: {means[0].shape}")
            elif isinstance(means, np.ndarray):
                print(f"shape={means.shape}")
    
    print("\n")

# Run the detailed check
for group in ['meditators', 'controls']:
    for networks in [7, 8]:
        examine_metrics_structure(group, networks, 4)