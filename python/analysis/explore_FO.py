"""
Simple script to explore the state patterns data structure.
This will help understand what columns are available for permutation testing.
"""

import os
import pandas as pd
import numpy as np

# Setup paths - same as in the permutation script
ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
STATE_PATTERNS_DIR = os.path.join(RESULTS_DIR, 'state_patterns')

# Load data
def explore_data(networks=7, k=4):
    patterns_path = os.path.join(STATE_PATTERNS_DIR, 'all_states_activation_patterns.csv')
    print(f"Loading data from: {patterns_path}")
    
    if not os.path.exists(patterns_path):
        print(f"ERROR: File not found: {patterns_path}")
        return
    
    # Load the dataframe
    patterns_df = pd.read_csv(patterns_path)
    
    # Filter to relevant configurations
    filtered_df = patterns_df[(patterns_df['networks'] == networks) & 
                            (patterns_df['k'] == k) &
                            (patterns_df['standardization'] == 'bygroup')]
    
    # Print general dataframe info
    print("\n===== DATAFRAME INFORMATION =====")
    print(f"Full dataframe shape: {patterns_df.shape}")
    print(f"Filtered dataframe shape: {filtered_df.shape}")
    
    # Print column names
    print("\n===== COLUMN NAMES =====")
    print(filtered_df.columns.tolist())
    
    # Print data types
    print("\n===== DATA TYPES =====")
    print(filtered_df.dtypes)
    
    # Print unique values for categorical columns
    print("\n===== UNIQUE VALUES =====")
    categorical_cols = ['networks', 'k', 'standardization', 'group', 'state_idx']
    for col in categorical_cols:
        if col in filtered_df.columns:
            print(f"{col}: {filtered_df[col].unique()}")
    
    # Show first few rows
    print("\n===== SAMPLE DATA =====")
    print(filtered_df.head())
    
    # Print group info
    print("\n===== GROUP SUMMARY =====")
    if 'group' in filtered_df.columns:
        group_counts = filtered_df['group'].value_counts()
        print(group_counts)
    
    # Check if subject information exists
    print("\n===== SUBJECT INFORMATION =====")
    subject_cols = [col for col in filtered_df.columns if 'subject' in col.lower()]
    if subject_cols:
        print(f"Subject columns found: {subject_cols}")
        for col in subject_cols:
            print(f"\nUnique values in {col}: {filtered_df[col].nunique()}")
            print(filtered_df[col].head())
    else:
        print("No subject columns found.")
    
    # Print FO summary by group and state
    print("\n===== FRACTIONAL OCCUPANCY SUMMARY =====")
    if 'fractional_occupancy' in filtered_df.columns:
        fo_summary = filtered_df.groupby(['group', 'state_idx'])['fractional_occupancy'].mean()
        print(fo_summary)
    
    return filtered_df

if __name__ == "__main__":
    print("Exploring state patterns data for 7-network, k=4 configuration...")
    data = explore_data(networks=7, k=4)
    print("\nData exploration complete!")