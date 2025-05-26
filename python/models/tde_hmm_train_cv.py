"""
GLHMM Training for Meditation States with Cross-Validation
- Trains separate models for controls and meditators
- Uses k-fold cross-validation across subjects for robust evaluation
- Tests expanded range of states (3-8)
- Uses full covariance models for optimal brain network interaction modeling
- Provides comprehensive state diagnostics using GLHMM utilities
- Incorporates time-delay embedding and PCA for temporal pattern detection

Usage:
  python tde_hmm_train_cv.py --standardize [global|bygroup|persequence] --networks [7|8]

References:
- Vidaurre et al. (2023). The Gaussian-linear hidden Markov model: A Python package.
  https://direct.mit.edu/imag/article/doi/10.1162/imag_a_00460/127499/The-Gaussian-Linear-Hidden-Markov-model-a-Python
"""

import os
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import KFold
import time
import logging
import argparse

# Import GLHMM modules
from glhmm import glhmm, preproc, utils, statistics, auxiliary

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup paths
ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
TDE_DIR = os.path.join(PROCESSED_DIR, 'tde')
TRAINED_DIR = os.path.join(DATA_DIR, 'trained', 'glhmm_tde')

# Create trained models directory
os.makedirs(TRAINED_DIR, exist_ok=True)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='TDE-HMM Training with Cross-Validation')
    
    parser.add_argument('--standardize', type=str, choices=['global', 'bygroup', 'persequence'],
                        default='persequence', 
                        help='Standardization method to use for training (default: persequence)')
    
    parser.add_argument('--networks', type=int, choices=[7, 8], default=8,
                        help='Number of networks in input data (default: 8)')
    
    args = parser.parse_args()
    return args

def load_tde_data(standardize_method='persequence', networks=8):
    """Load the preprocessed time-delay embedded data."""
    logger.info(f"Loading TDE preprocessed data (standardize={standardize_method}, networks={networks})...")
    
    # Define data path
    data_path = os.path.join(TDE_DIR, f'tde_{networks}networks_{standardize_method}.pkl')
    
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    try:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        logger.info(f"Loaded data with {len(data['controls_sequences'])} control subjects and "
                   f"{len(data['meditators_sequences'])} meditation subjects")
        logger.info(f"PCA components: {data['pca_components']}")
        logger.info(f"Lags used: {data['lags']}")
        
        return data
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def train_model_cv(sequences, lags, pca_components, indices_tde, n_states, n_folds=4):
    """Train a GLHMM using k-fold cross-validation across subjects."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_metrics = []
    
    # Get original network data for normalization
    n_regions = sequences[0].shape[1]
    
    # Concatenate all sequences for preprocessing
    X_concat = np.vstack(sequences)
    idx_data = statistics.get_indices_from_list(sequences)
    
    # Preprocess data using GLHMM utilities
    X_preproc = preproc.preprocess_data(X_concat, idx_data)[0]
    
    # Calculate network baselines for better group comparisons
    network_baselines = {}
    for net_idx in range(n_regions):
        network_baselines[net_idx] = {
            'mean': np.mean(X_preproc[:, net_idx]),
            'std': np.std(X_preproc[:, net_idx])
        }
    
    # Prepare TDE data
    X_embedded = np.vstack(sequences)
    idx_tde = indices_tde
    
    # Split subjects for cross-validation
    subject_indices = list(range(len(sequences)))
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(subject_indices)):
        logger.info(f"Training fold {fold_idx + 1}/{n_folds}")
        
        # Extract training segments and recalculate indices
        X_train_segments = []
        train_indices_adjusted = []
        current_pos = 0
        
        for i in train_idx:
            start, end = idx_tde[i]
            segment = X_embedded[start:end]
            segment_len = end - start
            
            X_train_segments.append(segment)
            train_indices_adjusted.append([current_pos, current_pos + segment_len])
            current_pos += segment_len
        
        # Create properly indexed training data
        X_train = np.vstack(X_train_segments)
        train_indices_adjusted = np.array(train_indices_adjusted)
        
        # Train the model
        model = glhmm.glhmm(
            K=n_states,
            covtype='full',
            model_beta='no',
            model_mean='state'
        )
        
        start_time = time.time()
        # Train on the subset with adjusted indices
        Gamma, Xi, fe = model.train(X=None, Y=X_train, indices=train_indices_adjusted, 
                                    options={'initmethod': 'random'})
        train_time = time.time() - start_time
        
        # Extract test segments and recalculate indices
        X_test_segments = []
        test_indices_adjusted = []
        current_pos = 0
        
        for i in test_idx:
            start, end = idx_tde[i]
            segment = X_embedded[start:end]
            segment_len = end - start
            
            X_test_segments.append(segment)
            test_indices_adjusted.append([current_pos, current_pos + segment_len])
            current_pos += segment_len
        
        # Create properly indexed test data
        X_test = np.vstack(X_test_segments)
        test_indices_adjusted = np.array(test_indices_adjusted)
        
        # Calculate test metrics using GLHMM functions
        test_free_energy = 0
        test_log_likelihood = 0
        
        for i, (start, end) in enumerate(test_indices_adjusted):
            test_data = X_test[start:end]
            gamma_i, xi_i, scale_i = model.decode(X=None, Y=test_data)
            test_log_likelihood += np.sum(np.log(scale_i))
            test_free_energy += np.sum(model.get_fe(X=None, Y=test_data, 
                                                   Gamma=gamma_i, Xi=xi_i, scale=scale_i))
        
        # Store metrics
        metrics = {
            'fold': fold_idx + 1,
            'test_log_likelihood': test_log_likelihood,
            'test_free_energy': test_free_energy,
            'training_time': train_time,
            'active_states': model.get_active_K()
        }
        
        logger.info(f"  Log-likelihood: {test_log_likelihood:.2f}")
        logger.info(f"  Free Energy: {test_free_energy:.2f}")
        fold_metrics.append(metrics)
    
    # Calculate average metrics
    average_metrics = {
        key: np.mean([m[key] for m in fold_metrics])
        for key in ['test_log_likelihood', 'test_free_energy', 'training_time']
    }
    average_metrics['active_states'] = [m['active_states'] for m in fold_metrics]
    
    # Train final model on all data
    final_model = glhmm.glhmm(
        K=n_states,
        covtype='full',
        model_beta='no',
        model_mean='state'
    )
    
    final_model.train(X=None, Y=X_embedded, indices=idx_tde, 
                     options={'initmethod': 'random'})
    
    # Calculate metrics using GLHMM utility functions
    vpath = final_model.decode(X=None, Y=X_embedded, indices=idx_tde, viterbi=True)
    
    # Get proper metrics using GLHMM utilities
    FO = utils.get_FO(vpath, indices=idx_tde)
    SR = utils.get_switching_rate(vpath, indices=idx_tde)
    LT_mean, LT_med, LT_max = utils.get_life_times(vpath, indices=idx_tde)
    
    # Create model data dictionary
    model_data = {
        'model': final_model,
        'vpath': vpath,
        'indices': idx_tde,
        'X_preproc': X_preproc,
        'X_embedded': X_embedded,
        'lags': lags,  
        'FO': FO,
        'SR': SR,
        'LT_mean': LT_mean,
        'network_baselines': network_baselines,
        'cv_metrics': average_metrics,
        'P': final_model.get_P(),
        'k': n_states,
        'pca_components': pca_components 
    }
    
    return model_data

def main():
    """Main function to run the GLHMM training pipeline with cross-validation"""
    # Parse command line arguments
    args = parse_arguments()
    
    print("=== GLHMM Training for Meditation State Analysis (TDE-PCA-CV) ===")
    print(f"Standardization method: {args.standardize}")
    print(f"Network configuration: {args.networks} networks")
    
    # Setup paths and load data
    data = load_tde_data(standardize_method=args.standardize, networks=args.networks)
    
    # Model configurations
    state_options = [3, 4, 5, 6, 7, 8]
    groups = ['controls', 'meditators']
    results = {}
    
    # Create output directory structure that includes standardization and network info
    standardized_dir = os.path.join(TRAINED_DIR, f"{args.standardize}_{args.networks}networks")
    
    for group in groups:
        print(f"\n=== Training {group.capitalize()} Models ===")
        results[group] = {}
        sequences = data[f'{group}_sequences']
        
        for n_states in state_options:
            print(f"\nTraining {n_states}-state model...")
            
            model_data = train_model_cv(
                sequences=sequences,
                lags=data['lags'],                     
                pca_components=data['pca_components'],  
                indices_tde=data['tde_parameters'][f'{group}_indices_tde'],
                n_states=n_states,
                n_folds=4
            )
            
            # Save results
            config = f"{n_states}states_tde_pca"
            results[group][config] = {
                'avg_metrics': model_data['cv_metrics']
            }
                        
            # Save model with updated directory structure
            model_dir = os.path.join(standardized_dir, group, f"k{n_states}")
            os.makedirs(model_dir, exist_ok=True)
            
            with open(os.path.join(model_dir, 'model.pkl'), 'wb') as f:
                pickle.dump(model_data, f)
    
    # Create summary report
    records = []
    for group in results:
        for config, result in results[group].items():
            n_states = int(config.split('states')[0])
            avg_metrics = result['avg_metrics']
            
            records.append({
                'group': group,
                'n_states': n_states,
                'standardization': args.standardize,
                'networks': args.networks,
                'test_free_energy': avg_metrics['test_free_energy'],
                'test_log_likelihood': avg_metrics['test_log_likelihood']
            })
    
    results_df = pd.DataFrame(records)
    summary_dir = os.path.join(os.path.dirname(TRAINED_DIR))
    os.makedirs(summary_dir, exist_ok=True)
    
    # Update CSV filename to include standardization and network info
    csv_filename = f'tde_model_comparison_{args.standardize}_{args.networks}networks.csv'
    results_df.to_csv(os.path.join(summary_dir, csv_filename), index=False)
    
    print("\n=== Training Complete ===")
    print(f"Results saved to: {os.path.join(summary_dir, csv_filename)}")
    print(f"Models saved to: {standardized_dir}")
    
if __name__ == "__main__":
    main()