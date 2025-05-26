# TDE-HMM Analysis Workflow for Meditation Research

This document outlines the complete workflow for analyzing fMRI data using Time-Delay Embedded Hidden Markov Models (TDE-HMM) to identify brain state dynamics during meditation.

## Overview

This analysis pipeline identifies recurring brain states, their temporal characteristics, and transition patterns that distinguish experienced meditators from controls, providing insights into neural mechanisms underlying meditation practices.

## 1. Data Preparation

### 1.1 Prerequisites
- Python 3.8+
- GLHMM package (Vidaurre et al., 2025)
- NumPy, pandas, matplotlib, seaborn
- scikit-learn for dimensionality reduction
- NetworkX for network analysis

### 1.2 Data Organization
```
python/
├── data/
│   ├── raw/                 # Original preprocessed fMRI data
│   └── trained/             # Trained HMM models
├── models/                  # Model training scripts
├── analysis/                # Analysis scripts
└── results/                 # Analysis outputs
    ├── metrics/             # Model metrics
    ├── visualizations/      # Figures
    └── transitions/         # State transition analyses
```


### 1.3 Standard File Paths

All scripts use the following standardized paths relative to the root project directory:

```python
# Base directories
ROOT_DIR = Path(__file__).parent.parent.absolute()  # Always two levels up from analysis scripts
DATA_DIR = ROOT_DIR / 'data'
RESULTS_DIR = ROOT_DIR / 'results'

# Input data directories
RAW_DATA_DIR = DATA_DIR / 'raw'
TRAINED_DIR = DATA_DIR / 'trained'

# Results directories
METRICS_DIR = RESULTS_DIR / 'metrics'
TRANSITIONS_DIR = METRICS_DIR / 'transitions'
VIS_DIR = RESULTS_DIR / 'visualizations'
REFERENCE_DIR = RESULTS_DIR / 'standardization_reference'

# Standardization-specific paths
GLOBAL_DIR = lambda networks, group, k: METRICS_DIR / 'global' / f'{networks}networks' / group / f'k{k}'
BYGROUP_DIR = lambda networks, group, k: METRICS_DIR / 'bygroup' / f'{networks}networks' / group / f'k{k}'

# Key Input/Output Files

The following table outlines the key input and output files for each stage of the analysis pipeline:

| Analysis Stage            | Input Files                                                                 | Output Files                                                                                     |
|---------------------------|-----------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| Model Training            | `RAW_DATA_DIR/*.npy`                                                        | `TRAINED_DIR/{std}/{networks}networks/{group}/k{k}/model.pkl`                                     |
| Metrics Calculation       | `TRAINED_DIR/{std}/.../model.pkl`                                           | `METRICS_DIR/{std}/{networks}networks/{group}/k{k}_metrics.pkl`                                   |
| Standardization Reference | `METRICS_DIR/{std}/.../k{k}_metrics.pkl`                                    | `REFERENCE_DIR/k{k}/{group}_{networks}networks_*.pkl`                                             |
| State Transitions         | `REFERENCE_DIR/k{k}/*_correspondence.pkl`                                   | `TRANSITIONS_DIR/{networks}networks/k{k}_succession_data.pkl`                                     |
| Permutation Testing       | `TRANSITIONS_DIR/{networks}networks/k{k}_succession_data.pkl`                | `TRANSITIONS_DIR/glhmm_analysis/{networks}networks/*`                                             |

## 2. Model Training and Selection

The model training process involves three key steps: cross-validation across multiple configurations, selection of optimal configuration, and final model training.

### 2.1 Cross-Validation Across Configurations

Run cross-validation to evaluate different model configurations:

```bash
python models/tde_hmm_train_cv.py
```

This script:
- Applies time-delay embedding to fMRI time series
- Performs dimensionality reduction using PCA
- Trains models across 6 different configurations:
  - Network parcellations: 7-network vs. 8-network (with subcortical)
  - Standardization methods: global, by-group, and per-sequence
- Tests multiple state numbers (k=3 to k=8) for each configuration
- Uses 4-fold cross-validation to ensure generalizability
- Produces separate model_comparison_cv.csv files for each configuration

### 2.2 Configuration Selection

Visualize and compare the performance of different configurations:

```bash
python visualization/tde_hmm_model_selection.py
```

This script:
- Creates standardized plots comparing model performance
- Generates one plot per configuration showing test free energy and log-likelihood
- Creates comparison plots across standardization methods
- Produces network comparison visualizations
- Saves results to the model_selection directory
- Assists in identifying the optimal configuration based on free energy

Manual inspection of these plots helps determine:
1. The optimal network parcellation (7 vs. 8 networks)
2. The best standardization approach (global vs. by-group vs. per-sequence)
3. The ideal number of states (k) for each group

### 2.3 Final Model Training

After identifying the optimal configuration and k values, train final models on the full dataset:

```bash
python models/tde_hmm_train.py
```
The TDE implementation specifically:
- Uses 21 lags (-10 to +10) to create a full embedding space of ~168 dimensions
- Applies PCA to reduce to 16 dimensions (2× the original network count)
- Trains the HMM in this reduced TDE space

Furthermore, this script:
- Trains models with k= 4,5 states using the standardization methods: global, by-group, and per-sequence
- Creates separate models for meditators and controls
- Saves complete model parameters for downstream analysis
- Generates Viterbi paths for state sequences
- Produces state covariance and mean activation patterns
- Stores results in the `data/trained/glhmm_tde/` directory

The final trained models serve as input for all subsequent analyses, including metrics calculation, state transition analysis, and visualization.

## 3. Metrics Calculation and State Correspondence Mapping

### 3.1 Metrics Calculation

Calculate comprehensive metrics from trained models for standardization comparision: 
global, by-group, and per-sequence, for k= 4,5 states

```bash
python analysis/tde_hmm_metrics.py
```
For network-level metrics, the script deliberately uses only the 
original network dimensions (~8), not the TDE-expanded or PCA-reduced spaces. 
Validation tests show up to 30× scale differences between direct calculation 
and TDE submatrix extraction. 

This script handles several important aspects:

- Aligns model components (X_preproc and vpath) to ensure consistent dimensions
- Calculates temporal metrics (FO, lifetimes, switching rates)
- Computes state-specific network activations
- Calculates direct network-to-network interactions from original dimensions
- Computes meditation-relevant metrics:
  - DMN anticorrelation with task-positive networks
  - DMN-LIM correlation (emotional regulation)
  - SMN correlations with attention networks (breath awareness)

### 3.2 Hierarchical Standardization Comparison
After calculating metrics for all standardization approaches, perform hierarchical comparison analysis:

```bash
python analysis/hierarchical_standardization_reference_k4.py
```
This script establishes objective benchmarks for comparing standardization approaches, focusing on k=4 models:

State correspondence analysis:
- Implements Hungarian algorithm to align states across standardization methods
- Uses cosine similarity between state patterns for optimal matching
- Quantifies alignment quality with overlap matrices and mean scores
- Creates separate mappings for each group and network configuration

Between-standardization comparisons:
- Maps global → bygroup states with customizable similarity thresholds
- Calculates metric deviations between matched states
- Quantifies how key metrics change across standardization methods
- Provides statistical benchmarks for expected variability

Between-group correspondences:
- Aligns meditator and control states for both standardization approaches
- Uses stricter thresholds for between-group matching (similarity ≥ 0.2)
- Generates comprehensive overlap matrices for all state pairs
- Produces visualizations showing corresponding states

Visualizations and outputs:
- Correspondence matrices with matching states highlighted
- Network activation patterns for corresponding states
- k-means and hierarchical clustering visualizations for state relationships
- Detailed deviation statistics for meditation-relevant metrics
- Summary CSV files with standardization stability metrics

This reference framework provides a systematic approach to understanding how standardization choices impact interpretation of brain states, ensuring robust comparisons between meditation practitioners and controls.

## 4. State Transition Analysis

### 4.1 Basic State Succession Analysis

Analyze state transitions and block-level successions:

```bash
python analysis/state_transition_analysis.py
```

This script:
- Extracts continuous state blocks from Viterbi-decoded paths
- Calculates succession probabilities between states
- Identifies common succession patterns
- Calculates entropy measures of state predictability (Kringelbach & Deco, 2020)
- Saves results to TRANSITIONS_DIR/{networks}networks/k{k}_succession_data.pkl

### 4.2 Permutation Testing (Compute-Intensive)

For statistical validation of succession differences, run permutation testing on a high-performance computer:

```bash
python analysis/permutation_tests_glhmm.py --networks 7 8 --k-values 4 5 --permutations 5000
```
This script:

- Loads succession data from TRANSITIONS_DIR/{networks}networks/k{k}_succession_data.pkl
- Uses GLHMM's statistical framework for permutation testing
- Performs 5000 permutations of group labels (--permutations parameter)
- Calculates empirical p-values for succession differences
- Applies FDR correction for multiple comparisons
- Tests both transition probabilities and fractional occupancy differences
- Generates comprehensive visualizations of results
- Saves statistics and figures to TRANSITIONS_DIR/glhmm_analysis/{networks}networks/
- Optimized for parallel processing 

## 5. Comparative Analysis

Compare brain dynamics between meditators and controls:

```bash
python analysis/comparative_analysis.py
```

This script produces:
- Network interaction comparisons
- Temporal dynamics differences

## 6. Meditation-Specific Analysis

Analyze meditation-specific neural signatures:

```bash
python analysis/meditation_signatures.py
```

This generates:
- Integration metrics (mutual information, global efficiency)
- Small-worldness and graph theoretical measures (Cabral et al., 2017)
- Mapping inspired from theoretical meditation states (Hasenkamp et al., 2012). Anapanasati meditation mapping, not Vipassana. 

## 7. Visualization

Generate standard visualizations for publication:

```bash
python models/standard_glhmm_visualizations.py
```

This creates figures for:
- Fractional occupancy
- Switching rates
- State lifetimes
- Network profiles
- Transition matrices
- Functional connectivity

## 8. Troubleshooting

### 8.1 Data Alignment Issues

When processing TDE-HMM models, dimension mismatches can occur between:
- Original data and TDE-expanded data
- Viterbi paths and preprocessed data

The pipeline includes automatic alignment to ensure:
- X_preproc and vpath have compatible dimensions
- Network metrics are calculated from aligned data
- Minimal data loss (typically <6% from sequence ends)

### 8.2 Validation Metrics

For model validation, check:
- Fractional occupancy should sum to approximately 1.0
- Network variances should be reasonably scaled
- Correlation matrices should contain values in [-1, 1]

## Key References

- Vidaurre, D., Smith, S. M., & Woolrich, M. W. (2017). Brain network dynamics are hierarchically organized in time. Proceedings of the National Academy of Sciences, 114(48), 12827-12832.
- Vidaurre, D., et al. (2023). The Gaussian-linear hidden Markov model: A Python package. Imaging Neuroscience, 1, 127499.
- Cabral, J., Kringelbach, M. L., & Deco, G. (2017). Functional connectivity dynamically evolves on multiple time-scales over a static structural connectome: Models and mechanisms. NeuroImage, 160, 84-96.
- Kringelbach, M. L., & Deco, G. (2020). Brain states and transitions: insights from computational neuroscience. Cell Reports, 32(10), 108128.
- Hasenkamp, W., Wilson-Mendenhall, C. D., Duncan, E., & Barsalou, L. W. (2012). Mind wandering and attention during focused meditation: A fine-grained temporal analysis of fluctuating cognitive states. NeuroImage, 59(1), 750-760.
- Lurie, D. J., et al. (2020). Questions and controversies in the study of time-varying functional connectivity in resting fMRI. Network Neuroscience, 4(1), 30-69.
- Stevner, A. B. A., et al. (2019). Discovery of key whole-brain transitions and dynamics during human wakefulness and non-REM sleep. Nature Communications, 10(1), 1035.
- Baker, A. P., et al. (2014). Fast transient networks in spontaneous human brain activity. eLife, 3, e01867.