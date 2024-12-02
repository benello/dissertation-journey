import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import logging

logger = logging.getLogger(__name__)

class DimensionalityAnalyser:
    """Handles dimensionality analysis using PCA, Parallel Analysis and Kaiser-Harris criteria."""

    def __init__(self, model, data_loader, config):
        """
                Initialize the unified analyzer.

                Args:
                    model: Trained neural network model
                    data_loader: DataLoader for feature extraction
                    config: Configuration dictionary
                """
        # Feature visualization attributes
        self.model = model
        self.data_loader = data_loader
        self.config = config['visualization']
        self.device = model.device

        # Shared attributes
        self.labels = None
        self.features = None

    def collect_features(self):
        """Collect features from the model's intermediate layer."""
        logger.info("Collecting features...")
        self.model.eval()
        features_list = []
        labels_list = []
        samples_collected = 0

        with torch.no_grad():
            for data, target in self.data_loader:
                if samples_collected >= self.config['num_samples']:
                    break

                data = data.to(self.device)
                _, features = self.model(data)
                features_list.append(features.cpu().numpy())
                labels_list.append(target.numpy())
                samples_collected += data.size(0)

        self.features = np.vstack(features_list)
        self.labels = np.concatenate(labels_list)
        logger.info(f"Collected {len(self.features)} samples")

    def parallel_analysis(self):
        """
        Perform parallel analysis to determine number of components to retain.

        Returns:
            tuple: (n_components, actual_eigenvalues, threshold_eigenvalues)
        """
        if self.features is None:
            raise ValueError("Features not collected. Call collect_features() first.")

        n_samples, n_features = self.features.shape

        # Calculate eigenvalues of actual data
        pca = PCA(n_components=n_features)
        pca.fit(self.features)
        actual_eigenvalues = pca.explained_variance_

        # Generate random eigenvalues
        random_eigenvalues = np.zeros((self.config['iterations'], n_features))
        for i in range(self.config['iterations']):
            random_data = np.random.normal(size=(n_samples, n_features))
            pca.fit(random_data)
            random_eigenvalues[i, :] = pca.explained_variance_

        threshold_eigenvalues = np.percentile(random_eigenvalues, self.config['percentile'], axis=0)
        n_components = sum(actual_eigenvalues > threshold_eigenvalues)

        return n_components, actual_eigenvalues, threshold_eigenvalues

    def kaiser_harris(self):
        """
        Apply Kaiser-Harris criterion to determine number of components.

        Returns:
            tuple: (n_components, eigenvalues)
        """
        if self.features is None:
            raise ValueError("Features not collected. Call collect_features() first.")

        pca = PCA(n_components=self.features.shape[1])
        pca.fit(self.features)
        eigenvalues = pca.explained_variance_
        n_components = sum(eigenvalues > 1.0)

        return n_components, eigenvalues

    def analyze_dimensionality(self):
        """
        Perform comprehensive dimensionality analysis using multiple methods.

        Returns:
            dict: Dictionary containing analysis results
        """
        # Perform parallel analysis
        pa_components, actual_eig, random_eig = self.parallel_analysis()

        # Perform Kaiser-Harris analysis
        kh_components, kh_eig = self.kaiser_harris()

        return {
            'pa_components': pa_components,
            'kh_components': kh_components,
            'actual_eigenvalues': actual_eig,
            'random_eigenvalues': random_eig,
            'kh_eigenvalues': kh_eig
        }

    def visualize(self):
        """
        Create comprehensive visualizations including PCA, t-SNE, and dimensionality analysis.

        Returns:
            tuple: (fig_pca, fig_analysis) containing the matplotlib figures
        """
        if self.features is None:
            raise ValueError("Features not collected. Call collect_features() first.")

        # Perform PCA
        pca = PCA()
        features_pca = pca.fit_transform(self.features)

        # Get dimensionality analysis results
        analysis_results = self.analyze_dimensionality()

        # Create PCA visualization figure
        fig_pca = plt.figure(figsize=tuple(self.config['fig_size']))

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
        plt.colorbar(scatter, label='Class')

        # Plot 3: t-SNE visualization
        plt.subplot(1, 3, 3)
        tsne = TSNE(n_components=2, random_state=42)
        features_tsne = tsne.fit_transform(self.features)
        scatter_tsne = plt.scatter(
            features_tsne[:, 0],
            features_tsne[:, 1],
            c=self.labels,
            cmap='tab10',
            alpha=0.6
        )
        plt.xlabel('First t-SNE Component')
        plt.ylabel('Second t-SNE Component')
        plt.title('t-SNE Visualization')
        plt.colorbar(scatter_tsne, label='Class')

        plt.tight_layout()

        # Create dimensionality analysis figure
        fig_analysis = plt.figure(figsize=tuple(self.config['fig_size']))

        # Plot eigenvalues and thresholds
        components = range(1, len(analysis_results['actual_eigenvalues']) + 1)
        plt.plot(components, analysis_results['actual_eigenvalues'], 'b-',
                 label='Actual Data')
        plt.plot(components, analysis_results['random_eigenvalues'], 'r--',
                 label='Parallel Analysis Threshold')
        plt.axhline(y=1.0, color='g', linestyle='--',
                    label='Kaiser-Harris Threshold')

        plt.xlabel('Component Number')
        plt.ylabel('Eigenvalue')
        plt.title('Dimensionality Analysis')
        plt.legend()
        plt.grid(True)

        # Add vertical lines for recommended components
        plt.axvline(x=analysis_results['pa_components'], color='r', alpha=0.3,
                    label='PA cutoff')
        plt.axvline(x=analysis_results['kh_components'], color='g', alpha=0.3,
                    label='KH cutoff')

        plt.tight_layout()

        # Print PCA statistics
        logger.info("\nAnalysis Summary:")
        logger.info(f"Total explained variance: {sum(pca.explained_variance_ratio_):.4f}")
        logger.info(f"Number of components for 90% variance: "
                    f"{len([x for x in cumulative_var if x <= 0.9]) + 1}")
        logger.info(f"PA recommended components: {analysis_results['pa_components']}")
        logger.info(f"KH recommended components: {analysis_results['kh_components']}")

        return fig_pca, fig_analysis

    def save_visualizations(self, output_dir):
        """
        Save all visualizations to files.

        Args:
            output_dir (str): Directory to save the visualization files
        """
        fig_pca, fig_analysis = self.visualize()

        # Save PCA and t-SNE visualizations
        fig_pca.savefig(f"{output_dir}/feature_visualization.png")

        # Save dimensionality analysis
        fig_analysis.savefig(f"{output_dir}/dimensionality_analysis.png")

        plt.close('all')
        logger.info(f"Saved visualizations to {output_dir}")