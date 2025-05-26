"""
Clustering for Meditation Brain State Visualization

This script performs k-means clustering (for PCA and t-SNE visualizations) and hierarchical
clustering on preprocessed eigenvector data from eigenvectors_data_yeo.mat (7-network, k=4,
by-group standardization). It generates three plots: PCA scatter, t-SNE scatter, and dendrograms,
with meditators on the left and controls on the right, using a consistent coloring scheme.

Plots are designed to be visually appealing with large fonts, distinct colors, and reduced clutter.
No state or network labeling is included, and no additional analyses (e.g., cluster matching) are performed.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import pickle
import logging
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, set_link_color_palette
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.io import loadmat

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Path Definitions ---
ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'preprocessed', 'eigenvectors_data_yeo.mat')
CLUSTER_DIR = os.path.join(ROOT_DIR, 'results', 'clustering')
VIS_DIR = os.path.join(CLUSTER_DIR, 'visualizations_Ma14')

os.makedirs(CLUSTER_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

# --- Constants ---
K = 4  # Number of clusters
NETWORKS = 7  # 7-network configuration
COLORS = plt.cm.tab10(np.linspace(0, 1, K))  # Consistent color scheme for clusters

def load_eigenvector_data(data_path):
    """Load preprocessed eigenvector data from MAT file and apply by-group standardization."""
    logger.info(f"Loading eigenvector data from {data_path}")
    
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    try:
        mat_data = loadmat(data_path, squeeze_me=True, struct_as_record=False)
        eigenvectors_data = mat_data['eigenvectors_data_yeo']
        
        data = {
            'controls_sequences': [],
            'meditators_sequences': [],
            'network_fields': list(eigenvectors_data.network_fields),
            'timepoints_per_subject': eigenvectors_data.info.timepoints_per_subject
        }
        
        # Load controls data
        for i in range(eigenvectors_data.controls.n_subjects):
            data['controls_sequences'].append(eigenvectors_data.controls.subjects[i])
        
        # Load meditators data
        for i in range(eigenvectors_data.meditators.n_subjects):
            data['meditators_sequences'].append(eigenvectors_data.meditators.subjects[i])
        
        # Apply by-group standardization
        for group in ['controls', 'meditators']:
            sequences = data[f'{group}_sequences']
            all_data = np.vstack(sequences)
            group_mean = np.mean(all_data, axis=0)
            group_std = np.std(all_data, axis=0) + 1e-6
            for i in range(len(sequences)):
                sequences[i] = (sequences[i] - group_mean) / group_std
        
        logger.info(f"Loaded data: Controls ({len(data['controls_sequences'])} subjects), "
                    f"Meditators ({len(data['meditators_sequences'])} subjects)")
        logger.info(f"Networks: {data['network_fields']}")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def perform_hierarchical_clustering(data, n_clusters=K, method='ward', metric='cosine'):
    """Perform hierarchical clustering on eigenvector data."""
    logger.info(f"Performing hierarchical clustering with k={n_clusters}")
    dist_matrix = pdist(data, metric=metric)
    Z = linkage(dist_matrix, method=method)
    labels = fcluster(Z, n_clusters, criterion='maxclust') - 1
    return {'labels': labels, 'Z': Z}

def perform_kmeans_clustering(data, n_clusters=K):
    """Perform k-means clustering."""
    logger.info(f"Performing K-means with k={n_clusters}")
    if data.shape[0] < n_clusters:
        logger.warning(f"Samples ({data.shape[0]}) < n_clusters ({n_clusters})")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(data)
    return {'labels': labels}

def visualize_dendrograms(control_Z, meditator_Z, n_clusters, output_path):
    """Create combined dendrogram visualization for meditators (left) and controls (right)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    color_palette = [matplotlib.colors.rgb2hex(COLORS[i][:3]) for i in range(n_clusters)]

    # Meditator Dendrogram
    meditator_threshold = meditator_Z[-(n_clusters-1), 2] if len(meditator_Z) >= n_clusters-1 else np.max(meditator_Z[:,2]) * 0.7
    set_link_color_palette(color_palette)
    dendrogram(meditator_Z, truncate_mode='lastp', p=n_clusters, color_threshold=meditator_threshold, 
               no_labels=True, orientation='top', ax=ax1, above_threshold_color='grey')
    ax1.axhline(y=meditator_threshold, c='k', ls='--', lw=1)
    ax1.set_title('Meditators', fontsize=14)
    ax1.set_xlabel('Data Points', fontsize=12)
    ax1.set_ylabel('Cosine Distance', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Control Dendrogram
    control_threshold = control_Z[-(n_clusters-1), 2] if len(control_Z) >= n_clusters-1 else np.max(control_Z[:,2]) * 0.7
    set_link_color_palette(color_palette)
    dendrogram(control_Z, truncate_mode='lastp', p=n_clusters, color_threshold=control_threshold, 
               no_labels=True, orientation='top', ax=ax2, above_threshold_color='grey')
    ax2.axhline(y=control_threshold, c='k', ls='--', lw=1)
    ax2.set_title('Controls', fontsize=14)
    ax2.set_xlabel('Data Points', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)

    set_link_color_palette(None)
    handles = [plt.Line2D([0], [0], color=c, lw=4) for c in color_palette]
    fig.legend(handles, [f'Cluster {i}' for i in range(n_clusters)], loc='lower center', ncol=n_clusters, fontsize=12)
    plt.suptitle(f'Hierarchical Clustering Dendrograms (k={n_clusters})', fontsize=16)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved dendrograms to {output_path}")

def visualize_pca_clusters(control_data, meditator_data, control_labels, meditator_labels, n_clusters, output_path):
    """Visualize k-means clusters in 2D PCA space, meditators left, controls right."""
    pca = PCA(n_components=2)
    combined_data = np.vstack([control_data, meditator_data]) if control_data.size and meditator_data.size else control_data if control_data.size else meditator_data
    if combined_data.size == 0:
        logger.warning("No data for PCA visualization")
        return
    
    pca.fit(combined_data)
    var_exp = pca.explained_variance_ratio_

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Subsample data (~100 points per subject, assuming ~20 subjects)
    subsample_size = 100 * 20 // max(1, len(control_data) // 1000 + len(meditator_data) // 1000)

    # Meditator PCA
    if meditator_data.size:
        med_pca = pca.transform(meditator_data)
        indices = np.random.choice(len(med_pca), min(subsample_size, len(med_pca)), replace=False)
        for i in range(n_clusters):
            mask = meditator_labels[indices] == i
            ax1.scatter(med_pca[indices][mask, 0], med_pca[indices][mask, 1], s=30, alpha=0.7, color=COLORS[i], label=f'Cluster {i}')
    ax1.set_title('Meditators', fontsize=14)
    ax1.set_xlabel(f'PC1 ({var_exp[0]*100:.1f}%)', fontsize=12)
    ax1.set_ylabel(f'PC2 ({var_exp[1]*100:.1f}%)', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.3)

    # Control PCA
    if control_data.size:
        con_pca = pca.transform(control_data)
        indices = np.random.choice(len(con_pca), min(subsample_size, len(con_pca)), replace=False)
        for i in range(n_clusters):
            mask = control_labels[indices] == i
            ax2.scatter(con_pca[indices][mask, 1], con_pca[indices][mask, 1], s=30, alpha=0.7, color=COLORS[i], label=f'Cluster {i}')
    ax2.set_title('Controls', fontsize=14)
    ax2.set_xlabel(f'PC1 ({var_exp[0]*100:.1f}%)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.3)

    plt.suptitle(f'K-means Clusters in PCA Space (k={n_clusters})', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved PCA clusters to {output_path}")

def visualize_tsne_clusters(control_data, meditator_data, control_labels, meditator_labels, n_clusters, output_path):
    """Visualize k-means clusters in 2D t-SNE space, meditators left, controls right."""
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    combined_data = np.vstack([control_data, meditator_data]) if control_data.size and meditator_data.size else control_data if control_data.size else meditator_data
    if combined_data.size == 0:
        logger.warning("No data for t-SNE visualization")
        return
    
    tsne_data = tsne.fit_transform(combined_data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Subsample data
    subsample_size = 100 * 20 // max(1, len(control_data) // 1000 + len(meditator_data) // 1000)
    control_offset = len(control_data)

    # Meditator t-SNE
    if meditator_data.size:
        med_tsne = tsne_data[control_offset:control_offset+len(meditator_data)]
        indices = np.random.choice(len(med_tsne), min(subsample_size, len(med_tsne)), replace=False)
        for i in range(n_clusters):
            mask = meditator_labels[indices] == i
            ax1.scatter(med_tsne[indices][mask, 0], med_tsne[indices][mask, 1], s=30, alpha=0.7, color=COLORS[i], label=f'Cluster {i}')
    ax1.set_title('Meditators', fontsize=14)
    ax1.set_xlabel('t-SNE 1', fontsize=12)
    ax1.set_ylabel('t-SNE 2', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.3)

    # Control t-SNE
    if control_data.size:
        con_tsne = tsne_data[:control_offset]
        indices = np.random.choice(len(con_tsne), min(subsample_size, len(con_tsne)), replace=False)
        for i in range(n_clusters):
            mask = control_labels[indices] == i
            ax2.scatter(con_tsne[indices][mask, 0], con_tsne[indices][mask, 1], s=30, alpha=0.7, color=COLORS[i], label=f'Cluster {i}')
    ax2.set_title('Controls', fontsize=14)
    ax2.set_xlabel('t-SNE 1', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.3)

    plt.suptitle(f'K-means Clusters in t-SNE Space (k={n_clusters})', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved t-SNE clusters to {output_path}")

def main():
    """Run PCA, t-SNE, and hierarchical clustering for k=4."""
    logger.info("Starting Clustering Analysis")
    
    # Load preprocessed eigenvector data
    try:
        data = load_eigenvector_data(RAW_DATA_PATH)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # Extract and concatenate sequences
    all_controls = np.vstack(data['controls_sequences'])
    all_meditators = np.vstack(data['meditators_sequences'])
    
    if all_controls.size == 0 or all_meditators.size == 0:
        logger.error("Control or meditator data is empty")
        return
    
    logger.info(f"Control data shape: {all_controls.shape}")
    logger.info(f"Meditator data shape: {all_meditators.shape}")

    # Perform clustering
    # Hierarchical
    control_hier = perform_hierarchical_clustering(all_controls, n_clusters=K)
    meditator_hier = perform_hierarchical_clustering(all_meditators, n_clusters=K)
    
    # K-means for PCA and t-SNE
    control_kmeans = perform_kmeans_clustering(all_controls, n_clusters=K)
    meditator_kmeans = perform_kmeans_clustering(all_meditators, n_clusters=K)

    # Visualize
    dendro_path = os.path.join(VIS_DIR, f'dendrograms_k{K}.png')
    visualize_dendrograms(control_hier['Z'], meditator_hier['Z'], K, dendro_path)
    
    pca_path = os.path.join(VIS_DIR, f'pca_clusters_k{K}.png')
    visualize_pca_clusters(all_controls, all_meditators, control_kmeans['labels'], meditator_kmeans['labels'], K, pca_path)
    
    tsne_path = os.path.join(VIS_DIR, f'tsne_clusters_k{K}.png')
    visualize_tsne_clusters(all_controls, all_meditators, control_kmeans['labels'], meditator_kmeans['labels'], K, tsne_path)

    logger.info("Clustering analysis completed")

if __name__ == "__main__":
    main()