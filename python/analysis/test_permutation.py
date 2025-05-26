"""
Test script for state_permutation_testing.py

This script runs a small-scale test of the permutation testing with 
only 5 permutations for k=4 to quickly validate functionality.
"""

import os
import sys
import time
import logging
from state_permutation_testing import (
    load_succession_data, 
    permutation_test_successions, 
    visualize_permutation_results, 
    save_permutation_results
)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_quick_test(networks=7, k=4, n_permutations=5):
    """Run a quick test of permutation testing with minimal iterations."""
    logger.info(f"===== TESTING PERMUTATION ANALYSIS WITH {n_permutations} PERMUTATIONS =====")
    logger.info(f"Testing on {networks}-network configuration, k={k}")
    
    start_time = time.time()
    
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(script_dir)
    RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
    METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')
    TRANSITIONS_DIR = os.path.join(METRICS_DIR, 'transitions')
    
    # Create test output directory
    test_dir = os.path.join(TRANSITIONS_DIR, 'test_permutation')
    os.makedirs(test_dir, exist_ok=True)
    
    try:
        # 1. Load data
        logger.info("Loading succession data...")
        succession_data = load_succession_data(networks, k)
        
        # 2. Run minimal permutation test
        logger.info(f"Running quick permutation test with {n_permutations} permutations...")
        perm_results = permutation_test_successions(succession_data, n_permutations)
        
        # 3. Visualize results
        logger.info("Creating visualizations...")
        plots_dir = visualize_permutation_results(perm_results, k, test_dir, networks)
        
        # 4. Save summary
        logger.info("Saving results summary...")
        summary_path = save_permutation_results(perm_results, k, test_dir, networks)
        
        elapsed_time = (time.time() - start_time)
        logger.info(f"Test completed in {elapsed_time:.2f} seconds")
        logger.info(f"Results saved to: {test_dir}")
        
        return test_dir
    
    except Exception as e:
        logger.error(f"Error during test: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Run tests for both network configurations
    result_dir = run_quick_test(networks=7, k=4, n_permutations=5)
    
    if result_dir:
        logger.info(f"Test successful! Check results in {result_dir}")
    else:
        logger.error("Test failed.")