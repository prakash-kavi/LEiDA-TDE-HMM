"""
Script to verify that TDE network submatrix extraction is correct.
This ensures that we're properly handling TDE expanded matrices.
"""

import os
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Add the project root to the path to ensure imports work
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Setup paths
DATA_DIR = os.path.join(project_root, 'data')
TRAINED_DIR = os.path.join(DATA_DIR, 'trained')

def load_model(group='meditators', k=5, networks=8):
    """Load a trained model."""
    model_path = os.path.join(TRAINED_DIR, f'selected_model_{networks}networks', 
                              group, f'k{k}', 'model.pkl')
    
    print(f"Loading model from: {model_path}")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # Add k to model_data if it doesn't exist
    if 'k' not in model_data:
        print(f"Adding missing 'k' value ({k}) to model_data")
        model_data['k'] = k
    
    return model_data

def verify_tde_extraction(model_data):
    """Verify TDE submatrix extraction."""
    hmm = model_data['hmm']
    network_fields = model_data['network_fields']
    network_count = len(network_fields)
    k = model_data['k']
    
    # Get covariance matrices
    print("\nExamining covariance matrices...")
    for state in range(k):
        cov = hmm.get_covariance_matrix(state)
        print(f"State {state+1} covariance shape: {cov.shape}")
        
        if cov.shape[0] > network_count:
            print(f"  TDE expanded matrix detected - needs {network_count}×{network_count} submatrix extraction")
            
            # Examine diagonal values
            diag_full = np.diag(cov)
            diag_networks = diag_full[:network_count]
            diag_tde = diag_full[network_count:]
            
            print(f"  Network diagonal values: {diag_networks}")
            print(f"  First few TDE lag values: {diag_tde[:5]}")
            
            # Compare with network values from X_preproc
            print("\nComparing with direct network data calculations...")
            X_preproc = model_data['X_preproc']
            vpath = model_data['vpath']
            
            # Convert vpath if needed
            if len(vpath.shape) == 2 and vpath.shape[1] > 1:
                vpath_1d = np.argmax(vpath, axis=1)
            else:
                vpath_1d = vpath
            
            # Handle length mismatch
            if len(vpath_1d) > X_preproc.shape[0]:
                vpath_1d = vpath_1d[:X_preproc.shape[0]]
            elif len(vpath_1d) < X_preproc.shape[0]:
                X_preproc = X_preproc[:len(vpath_1d), :]
            
            # Get network data only
            X_networks = X_preproc[:, :network_count]
            
            # Calculate direct covariance
            state_mask = vpath_1d == state
            if np.sum(state_mask) > 10:
                state_data = X_networks[state_mask]
                direct_cov = np.cov(state_data, rowvar=False)
                
                print(f"  Direct network covariance shape: {direct_cov.shape}")
                print(f"  Direct network diagonal values: {np.diag(direct_cov)}")
                
                # Compare submatrix with direct calculation
                submatrix = cov[:network_count, :network_count]
                diff = np.abs(submatrix - direct_cov)
                print(f"  Mean absolute difference: {np.mean(diff):.6f}")
                print(f"  Max absolute difference: {np.max(diff):.6f}")
                
                if np.mean(diff) < 0.01:
                    print("  ✓ Submatrix extraction matches direct calculation (differences < 0.01)")
                else:
                    print("  ✗ Submatrix extraction differs from direct calculation")
                    
                # Visualize comparison
                plt.figure(figsize=(15, 5))
                
                plt.subplot(131)
                plt.imshow(submatrix, cmap='coolwarm', vmin=-1, vmax=1)
                plt.colorbar()
                plt.title(f"State {state+1} TDE Submatrix")
                
                plt.subplot(132)
                plt.imshow(direct_cov, cmap='coolwarm', vmin=-1, vmax=1)
                plt.colorbar()
                plt.title(f"State {state+1} Direct Calculation")
                
                plt.subplot(133)
                plt.imshow(diff, cmap='Reds')
                plt.colorbar()
                plt.title(f"Absolute Difference")
                
                plt.tight_layout()
                plt.savefig(f'state{state+1}_network_submatrix_verification.png')
                plt.close()
            else:
                print(f"  Not enough timepoints ({np.sum(state_mask)}) for direct calculation")

def main():
    """Main function."""
    print("=== Verifying TDE Network Submatrix Extraction ===")
    
    # Load a model
    model_data = load_model()
    
    # Verify extraction
    verify_tde_extraction(model_data)

if __name__ == "__main__":
    main()