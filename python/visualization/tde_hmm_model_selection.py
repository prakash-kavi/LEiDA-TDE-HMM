import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Setup paths
ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
TRAINED_DIR = os.path.join(DATA_DIR, 'trained')
MODELS_DIR = os.path.join(TRAINED_DIR, 'glhmm_tde')  # Where model files are stored
CSV_DIR = TRAINED_DIR  # Where CSV comparison files are stored
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
MODEL_SELECTION_DIR = os.path.join(RESULTS_DIR, 'model_selection')

# Create necessary directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(MODEL_SELECTION_DIR, exist_ok=True)

# Define all TDE configurations to process
network_configs = ['7networks', '8networks']
std_methods = ['global', 'bygroup', 'persequence']

# Initialize variables to track global min/max values for consistent axes
global_fe_min, global_fe_max = float('inf'), float('-inf')
global_ll_min, global_ll_max = float('inf'), float('-inf')

# First pass: determine global min/max values across all configurations
all_dfs = []
for network_config in network_configs:
    for std_method in std_methods:
        config_dir = f"{std_method}_{network_config}"
        results_dir = os.path.join(MODELS_DIR, config_dir)        
        # Skip if directory doesn't exist
        if not os.path.exists(results_dir):
            print(f"Warning: Results directory not found: {results_dir}")
            continue
            
        # Load results DataFrame
        comparison_path = os.path.join(CSV_DIR, f'tde_model_comparison_{std_method}_{network_config}.csv')
        if not os.path.exists(comparison_path):
            print(f"Warning: Comparison file not found: {comparison_path}")
            continue
            
        results_df = pd.read_csv(comparison_path)
        all_dfs.append(results_df)
        
        # Add error handling if no data files were found
        if not all_dfs:
            print("No data files were found. Please check CSV path and file naming.")
            exit()
        
        # Update global min/max values
        global_fe_min = min(global_fe_min, results_df['test_free_energy'].min())
        global_fe_max = max(global_fe_max, results_df['test_free_energy'].max())
        global_ll_min = min(global_ll_min, results_df['test_log_likelihood'].min())
        global_ll_max = max(global_ll_max, results_df['test_log_likelihood'].max())

# Add padding (5%)
fe_pad = (global_fe_max - global_fe_min) * 0.05
ll_pad = (global_ll_max - global_ll_min) * 0.05

global_fe_min -= fe_pad
global_fe_max += fe_pad
global_ll_min -= ll_pad
global_ll_max += ll_pad

# Second pass: create standardized plots for each configuration
for network_config in network_configs:
    for std_method in std_methods:
        config_dir = f"{std_method}_{network_config}"
        results_dir = os.path.join(MODELS_DIR, config_dir)        
        # Skip if directory doesn't exist
        if not os.path.exists(results_dir):
            continue
            
        # Load results DataFrame
        comparison_path = os.path.join(CSV_DIR, f'tde_model_comparison_{std_method}_{network_config}.csv')
        if not os.path.exists(comparison_path):
            continue
            
        results_df = pd.read_csv(comparison_path)
        
        print(f"Creating plot for: {config_dir}")
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Define groups and colors
        groups = ['controls', 'meditators']
        colors = {'free_energy': '#1f77b4', 'log_likelihood': '#d62728'}
        
        # Store twin axes for later legend creation
        twin_axes = []
        
        for i, group in enumerate(groups):
            # Filter data for the group
            group_data = results_df[results_df['group'] == group]
            
            # Sort by number of states
            group_data = group_data.sort_values('n_states')
            
            # Create twin axis for each subplot
            ax1 = axes[i]
            ax2 = ax1.twinx()
            twin_axes.append(ax2)  # Store the twin axis reference
            
            # Plot Free Energy on left axis
            ax1.plot(group_data['n_states'], group_data['test_free_energy'], 
                     marker='o', linestyle='-', linewidth=2.5, color=colors['free_energy'],
                     label='Test Free Energy')
            
            # Plot Log-Likelihood on right axis
            ax2.plot(group_data['n_states'], group_data['test_log_likelihood'],
                     marker='s', linestyle='--', linewidth=2.5, color=colors['log_likelihood'],
                     label='Test Log-Likelihood')
            
            # Set titles and labels
            ax1.set_title(f'{group.capitalize()} Group', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Number of States (K)', fontsize=12)
            ax1.set_xticks(group_data['n_states'])
            
            # Set y-axis labels
            ax1.set_ylabel('Free Energy (lower is better)', fontsize=12, color=colors['free_energy'])
            ax2.set_ylabel('Log-Likelihood (higher is better)', fontsize=12, color=colors['log_likelihood'])
            
            # Set colors for axis ticks
            ax1.tick_params(axis='y', labelcolor=colors['free_energy'])
            ax2.tick_params(axis='y', labelcolor=colors['log_likelihood'])
            
            # Add grid for readability
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Apply standardized axis limits
            ax1.set_ylim([global_fe_min, global_fe_max])
            ax2.set_ylim([global_ll_min, global_ll_max])
        
        # Add a common legend
        handles1, labels1 = axes[0].get_legend_handles_labels()
        handles2, labels2 = twin_axes[0].get_legend_handles_labels()
        fig.legend(handles1 + handles2, labels1 + labels2, loc='upper center', 
                   bbox_to_anchor=(0.5, 0.05), ncol=2, fontsize=12)
        
        # Add overall title with configuration details
        std_display = std_method.replace('bygroup', 'By Group').replace('persequence', 'Per Sequence')
        plt.suptitle(f'GLHMM Model Selection: {network_config} with {std_display} Standardization', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        
        # Save figure
        save_name = f"model_selection_plot_{network_config}_{std_method}"
        save_path = os.path.join(MODEL_SELECTION_DIR, f"{save_name}.png")        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
        
        print(f"Plot saved to: {save_path}")
        plt.close()
        
        
# Third pass: create consolidated comparison plots for each network config
os.makedirs(MODEL_SELECTION_DIR, exist_ok=True)

# Define standardization method line styles for comparison plots
std_styles = {
    'global': {'linestyle': '-', 'marker': 'o'},       # solid line with circle
    'bygroup': {'linestyle': '--', 'marker': 's'},     # dashed line with square
    'persequence': {'linestyle': ':', 'marker': '^'}   # dotted line with triangle
}

std_display_names = {
    'global': 'Global',
    'bygroup': 'By Group',
    'persequence': 'Per Sequence'
}

# Create one comparison plot for each network configuration
for network_config in network_configs:
    print(f"\nCreating comparison plot for {network_config}...")
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Define groups and colors (same as before)
    groups = ['controls', 'meditators']
    colors = {'free_energy': '#1f77b4', 'log_likelihood': '#d62728'}
    
    # Store twin axes for later legend creation
    twin_axes = []
    
    for i, group in enumerate(groups):
        # Create twin axis for each subplot
        ax1 = axes[i]
        ax2 = ax1.twinx()
        twin_axes.append(ax2)
        
        # Plot each standardization method on the same axes
        for std_method in std_methods:
            config_dir = f"{std_method}_{network_config}"
            results_dir = os.path.join(MODELS_DIR, config_dir)            
            # Skip if directory doesn't exist
            if not os.path.exists(results_dir):
                continue
                
            # Load results DataFrame
            comparison_path = os.path.join(CSV_DIR, f'tde_model_comparison_{std_method}_{network_config}.csv')
            if not os.path.exists(comparison_path):
                continue
                
            results_df = pd.read_csv(comparison_path)
            
            # Filter data for the group
            group_data = results_df[results_df['group'] == group]
            
            # Sort by number of states
            group_data = group_data.sort_values('n_states')
            
            # Plot Free Energy with different line styles and markers
            ax1.plot(group_data['n_states'], group_data['test_free_energy'], 
                    marker=std_styles[std_method]['marker'], 
                    linestyle=std_styles[std_method]['linestyle'], 
                    linewidth=2.5, 
                    color=colors['free_energy'],
                    label=f"{std_display_names[std_method]} (FE)")
            
            # Plot Log-Likelihood with different line styles and markers
            ax2.plot(group_data['n_states'], group_data['test_log_likelihood'],
                    marker=std_styles[std_method]['marker'], 
                    linestyle=std_styles[std_method]['linestyle'], 
                    linewidth=2.5, 
                    color=colors['log_likelihood'],
                    label=f"{std_display_names[std_method]} (LL)")
        
        # Set titles and labels (same as before)
        ax1.set_title(f'{group.capitalize()} Group', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Number of States (K)', fontsize=12)
        ax1.set_xticks(np.arange(3, 9))  # Assuming states 3-8
        
        # Set y-axis labels
        ax1.set_ylabel('Free Energy (lower is better)', fontsize=12, color=colors['free_energy'])
        ax2.set_ylabel('Log-Likelihood (higher is better)', fontsize=12, color=colors['log_likelihood'])
        
        # Set colors for axis ticks
        ax1.tick_params(axis='y', labelcolor=colors['free_energy'])
        ax2.tick_params(axis='y', labelcolor=colors['log_likelihood'])
        
        # Add grid for readability
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Apply standardized axis limits
        ax1.set_ylim([global_fe_min, global_fe_max])
        ax2.set_ylim([global_ll_min, global_ll_max])
    
        # Get the handles and labels from both axes
        handles1, labels1 = axes[0].get_legend_handles_labels()
        handles2, labels2 = twin_axes[0].get_legend_handles_labels()
        
        # Create clearer labels that emphasize the standardization method
        modified_labels1 = []
        modified_labels2 = []
        
        for label in labels1:
            method = label.split(" ")[0]  # Extract the method name
            modified_labels1.append(f"{method} (Free Energy)")
        
        for label in labels2:
            method = label.split(" ")[0]  # Extract the method name
            modified_labels2.append(f"{method} (Log-Likelihood)")
        
        # Create a custom legend with all 6 combinations
        from matplotlib.lines import Line2D
        custom_lines = []
        custom_labels = []
        
        # Add Free Energy entries
        for method in std_methods:
            custom_line = Line2D([0], [0], 
                                color=colors['free_energy'],
                                marker=std_styles[method]['marker'],
                                linestyle=std_styles[method]['linestyle'],
                                linewidth=2.5)
            custom_lines.append(custom_line)
            custom_labels.append(f"{std_display_names[method]} (Free Energy)")
        
        # Add Log-Likelihood entries
        for method in std_methods:
            custom_line = Line2D([0], [0], 
                                color=colors['log_likelihood'],
                                marker=std_styles[method]['marker'],
                                linestyle=std_styles[method]['linestyle'],
                                linewidth=2.5)
            custom_lines.append(custom_line)
            custom_labels.append(f"{std_display_names[method]} (Log-Likelihood)")
        
        # Create legend with custom entries
        fig.legend(custom_lines, custom_labels, 
                loc='upper center', 
                bbox_to_anchor=(0.5, 0.05), 
                ncol=2,  # Changed from 3 to 2 (as per Javier's request), 
                fontsize=11,
                frameon=True)
    
    # Add overall title with configuration details
    plt.suptitle(f'GLHMM Model Selection Comparison: {network_config}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    # Save figure
    save_name = f"model_selection_comparison_{network_config}"
    save_path = os.path.join(MODEL_SELECTION_DIR, f"{save_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"Comparison plot saved to: {save_path}")
    plt.close()

# Fourth pass: create a network comparison plot for per-sequence standardization
print("\nCreating network comparison plot for per-sequence standardization...")

# Define network config line styles for comparison plots
network_styles = {
    '7networks': {'linestyle': '-', 'marker': 'o'},     # solid line with circle
    '8networks': {'linestyle': '--', 'marker': 's'}     # dashed line with square
}

network_display_names = {
    '7networks': '7 Networks (No SUB)',
    '8networks': '8 Networks (With SUB)'
}

# Target standardization method
std_method = 'bygroup'

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Define groups and colors
groups = ['controls', 'meditators']
colors = {'free_energy': '#1f77b4', 'log_likelihood': '#d62728'}

# Store twin axes for later legend creation
twin_axes = []

for i, group in enumerate(groups):
    # Create twin axis for each subplot
    ax1 = axes[i]
    ax2 = ax1.twinx()
    twin_axes.append(ax2)
    
    # Plot each network configuration on the same axes
    for network_config in network_configs:
        config_dir = f"{std_method}_{network_config}"
        results_dir = os.path.join(MODELS_DIR, config_dir)        
        # Skip if directory doesn't exist
        if not os.path.exists(results_dir):
            continue
            
        # Load results DataFrame
        comparison_path = os.path.join(CSV_DIR, f'tde_model_comparison_{std_method}_{network_config}.csv')
        if not os.path.exists(comparison_path):
            continue
            
        results_df = pd.read_csv(comparison_path)
        
        # Filter data for the group
        group_data = results_df[results_df['group'] == group]
        
        # Sort by number of states
        group_data = group_data.sort_values('n_states')
        
        # Plot Free Energy with different line styles and markers
        ax1.plot(group_data['n_states'], group_data['test_free_energy'], 
                marker=network_styles[network_config]['marker'], 
                linestyle=network_styles[network_config]['linestyle'], 
                linewidth=2.5, 
                color=colors['free_energy'],
                label=f"{network_display_names[network_config]} (FE)")
        
        # Plot Log-Likelihood with different line styles and markers
        ax2.plot(group_data['n_states'], group_data['test_log_likelihood'],
                marker=network_styles[network_config]['marker'], 
                linestyle=network_styles[network_config]['linestyle'], 
                linewidth=2.5, 
                color=colors['log_likelihood'],
                label=f"{network_display_names[network_config]} (LL)")
    
    # Set titles and labels
    ax1.set_title(f'{group.capitalize()} Group', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Number of States (K)', fontsize=12)
    ax1.set_xticks(np.arange(3, 9))  # Assuming states 3-8
    
    # Set y-axis labels
    ax1.set_ylabel('Free Energy (lower is better)', fontsize=12, color=colors['free_energy'])
    ax2.set_ylabel('Log-Likelihood (higher is better)', fontsize=12, color=colors['log_likelihood'])
    
    # Set colors for axis ticks
    ax1.tick_params(axis='y', labelcolor=colors['free_energy'])
    ax2.tick_params(axis='y', labelcolor=colors['log_likelihood'])
    
    # Add grid for readability
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Apply standardized axis limits
    ax1.set_ylim([global_fe_min, global_fe_max])
    ax2.set_ylim([global_ll_min, global_ll_max])

# Create a custom legend with all combinations
from matplotlib.lines import Line2D
custom_lines = []
custom_labels = []

# Add Free Energy entries
for network_config in network_configs:
    custom_line = Line2D([0], [0], 
                       color=colors['free_energy'],
                       marker=network_styles[network_config]['marker'],
                       linestyle=network_styles[network_config]['linestyle'],
                       linewidth=2.5)
    custom_lines.append(custom_line)
    custom_labels.append(f"{network_display_names[network_config]} (Free Energy)")

# Add Log-Likelihood entries
for network_config in network_configs:
    custom_line = Line2D([0], [0], 
                       color=colors['log_likelihood'],
                       marker=network_styles[network_config]['marker'],
                       linestyle=network_styles[network_config]['linestyle'],
                       linewidth=2.5)
    custom_lines.append(custom_line)
    custom_labels.append(f"{network_display_names[network_config]} (Log-Likelihood)")

# Create legend with custom entries
fig.legend(custom_lines, custom_labels, 
         loc='upper center', 
         bbox_to_anchor=(0.5, 0.05), 
         ncol=2, 
         fontsize=11,
         frameon=True)

# Add overall title with configuration details
plt.suptitle(f'GLHMM Model Selection: 7 vs 8 Networks Comparison with By-Group Standardization', 
           fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0.08, 1, 0.95])

# Save figure
save_name = "model_selection_network_comparison_bygroup"
save_path = os.path.join(MODEL_SELECTION_DIR, f"{save_name}.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')

print(f"Network comparison plot saved to: {save_path}")
plt.close()

print("All model selection plots created successfully!")