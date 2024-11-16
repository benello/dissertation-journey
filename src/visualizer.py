import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import logging

logger = logging.getLogger(__name__)

class FeatureVisualizer:
    """Handles feature visualization using PCA and t-SNE."""
    
    def __init__(self, model, data_loader, config):
        """
        Initialize the visualizer.
        
        Args:
            model: Trained neural network model
            data_loader: DataLoader for feature extraction
            config: Configuration dictionary
        """
        self.model = model
        self.data_loader = data_loader
        self.config = config
        self.device = model.device

    def collect_features(self):
        """Collect features from the model's intermediate layer."""
        logger.info("Collecting features...")
        self.model.eval()
        features_list = []
        labels_list = []
        samples_collected = 0
        
        with torch.no_grad():
            for data, target in self.data_loader:
                if samples_collected >= self.config['visualization']['num_samples']:
                    break
                
                data = data.to(self.device)
                _, features = self.model(data)
                features_list.append(features.cpu().numpy())
                labels_list.append(target.numpy())
                samples_collected += data.size(0)
        
        self.features = np.vstack(features_list)
        self.labels = np.concatenate(labels_list)
        logger.info(f"Collected {len(self.features)} samples")
        
    def visualize(self):
        """Create visualizations using PCA and t-SNE."""
        logger.info("Creating visualizations...")
        
        # Perform PCA
        pca = PCA()
        features_pca = pca.fit_transform(self.features)
        
        # Create figure
        fig_size = self.config['visualization']['fig_size']
        plt.figure(figsize=tuple(fig_size))
        
        # Plot 1: Explained variance ratio
        plt.subplot(1, 3, 1)
        cumulative_var = np.cumsum(pca.explained_variance_ratio_)
        components = range(1, len(pca.explained_variance_ratio_) + 1)
        
        plt.plot(components, cumulative_var, 'bo-')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Explained Variance')
        plt.grid(True)
        
        # Plot 2: First two PCA components
        plt.subplot(1, 3, 2)
        scatter = plt.scatter(
            features_pca[:, 0], 
            features_pca[:, 1],
            c=self.labels, 
            cmap='tab10', 
            alpha=0.6
        )
        plt.xlabel('First PCA Component')
        plt.ylabel('Second PCA Component')
        plt.title('PCA Components Visualization')
        plt.colorbar(scatter, label='Digit Class')
        
        # Plot 3: t-SNE visualization
        plt.subplot(1, 3, 3)
        tsne = TSNE(
            n_components=2,
            random_state=self.config['visualization']['tsne_random_state'],
            perplexity=self.config['visualization']['tsne_perplexity']
        )
        features_tsne = tsne.fit_transform(self.features)
        
        scatter = plt.scatter(
            features_tsne[:, 0], 
            features_tsne[:, 1],
            c=self.labels, 
            cmap='tab10', 
            alpha=0.6
        )
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title('t-SNE Visualization')
        plt.colorbar(scatter, label='Digit Class')
        
        plt.tight_layout()
        
        # Print PCA statistics
        logger.info("\nPCA Analysis Summary:")
        logger.info(f"Total explained variance: {sum(pca.explained_variance_ratio_):.4f}")
        logger.info(f"Number of components for 90% variance: "
                   f"{len([x for x in cumulative_var if x <= 0.9]) + 1}")
        
        return plt.gcf()  # Return the figure for saving if needed