import multiprocessing
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import logging

from torch import nn
from tqdm import tqdm
from joblib import Parallel, delayed

from src.activation_tracking import ActivationTracker

logger = logging.getLogger(__name__)


class DimensionalityAnalyser:
    def __init__(self, model, data_loader, config):
        self.model = model
        self.data_loader = data_loader
        self.config = config['visualization']
        self.device = model.device
        self.labels = None
        self.features = None

    def collect_features(self):
        """Example of using tracker for feature collection."""
        tracker = ActivationTracker("outputs")
        tracker.set_layers_to_track([nn.Conv2d, nn.ReLU])

        labels_list = []
        samples_collected = 0

        with torch.no_grad(), tracker.track(self.model, 'collect_features') as tracked_model:
            for data, target in self.data_loader:
                if samples_collected >= self.config['num_samples']:
                    break

                _ = tracked_model(data.to(self.device))
                labels_list.append(target.numpy())
                samples_collected += data.size(0)

            # Get and process features
            features = tracker.get_activations('conv_layers.0')
            features = features.reshape(features.size(0), -1)
            labels = np.concatenate(labels_list)

            return features, labels

    def parallel_analysis(self):
        """
        Perform parallel analysis to determine number of components to retain.

        Returns:
            tuple: (n_components, actual_eigenvalues, threshold_eigenvalues)
        """
        if self.features is None:
            raise ValueError("Features not collected. Call collect_features() first.")

        n_samples, _ = self.features.shape
        logger.info(f"Running parallel analysis with {self.config['iterations']} iterations...")

        # Calculate eigenvalues of actual data
        pca = PCA(n_components=n_samples)
        pca.fit(self.features)
        actual_eigenvalues = pca.explained_variance_
        # Generate random eigenvalues
        pa = ParallelAnalysis(self.config['iterations'], n_jobs=4)
        threshold_eigenvalues = pa.fit(self.features)

        # Determine number of components to retain
        n_components = sum(actual_eigenvalues > threshold_eigenvalues)

        logger.info(f"Parallel Analysis suggests {n_components} components")
        return n_components, actual_eigenvalues, threshold_eigenvalues

    def kaiser_harris(self):
        """
        Apply Kaiser-Harris criterion to determine number of components.

        Returns:
            tuple: (n_components, eigenvalues)
        """
        if self.features is None:
            raise ValueError("Features not collected. Call collect_features() first.")

        n_samples, _ = self.features.shape

        # Perform PCA
        pca = PCA(n_components=n_samples)
        pca.fit(self.features)
        eigenvalues = pca.explained_variance_
        n_components = sum(eigenvalues > 1.0)

        logger.info(f"Kaiser-Harris criterion suggests {n_components} components")
        return n_components, eigenvalues

    def analyze_dimensionality(self):
        """
        Perform comprehensive dimensionality analysis using multiple methods.

        Returns:
            dict: Dictionary containing analysis results and recommended dimensions
        """
        # Perform parallel analysis
        pa_components, actual_eig, random_eig = self.parallel_analysis()

        # Perform Kaiser-Harris analysis
        kh_components, kh_eig = self.kaiser_harris()

        # Calculate PCA explained variance for comparison
        pca = PCA()
        pca.fit(self.features)
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)

        # Number of components for different variance thresholds
        variance_components = {
            '90%': len([x for x in cumulative_variance if x <= 0.9]) + 1,
            '95%': len([x for x in cumulative_variance if x <= 0.95]) + 1,
            '99%': len([x for x in cumulative_variance if x <= 0.99]) + 1
        }

        return {
            'pa_components': pa_components,
            'kh_components': kh_components,
            'actual_eigenvalues': actual_eig,
            'random_eigenvalues': random_eig,
            'kh_eigenvalues': kh_eig,
            'explained_variance_ratio': explained_variance_ratio,
            'cumulative_variance': cumulative_variance,
            'variance_components': variance_components
        }

    def visualize_analysis(self, results):
        """
        Create comprehensive visualization of dimensionality analysis results.

        Args:
            results: Dictionary containing analysis results from analyze_dimensionality()
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Eigenvalue comparison
        components = range(1, len(results['actual_eigenvalues']) + 1)
        ax1.plot(components, results['actual_eigenvalues'], 'b-',
                 label='Actual Data', linewidth=2)
        ax1.plot(components, results['random_eigenvalues'], 'r--',
                 label='PA Threshold', linewidth=1.5)
        ax1.axhline(y=1.0, color='g', linestyle='--',
                    label='Kaiser-Harris Threshold', linewidth=1.5)

        # Add vertical lines for recommended components
        ax1.axvline(x=results['pa_components'], color='r', alpha=0.3,
                    label=f'PA cutoff ({results["pa_components"]})')
        ax1.axvline(x=results['kh_components'], color='g', alpha=0.3,
                    label=f'KH cutoff ({results["kh_components"]})')

        ax1.set_xlabel('Component Number')
        ax1.set_ylabel('Eigenvalue')
        ax1.set_title('Dimensionality Analysis: Eigenvalue Comparison')
        ax1.legend()
        ax1.grid(True)

        # Plot 2: Cumulative explained variance
        components = range(1, len(results['explained_variance_ratio']) + 1)
        ax2.plot(components, results['cumulative_variance'], 'b-', linewidth=2)

        # Add markers for variance thresholds
        for threshold, n_comp in results['variance_components'].items():
            ax2.axvline(x=n_comp, color='r', alpha=0.3,
                        label=f'{threshold} var ({n_comp} components)')

        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.set_title('PCA Explained Variance')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        return fig

    def run_analysis(self):
        """Run complete dimensionality analysis and save results."""
        # Ensure features are collected
        if self.features is None:
            self.features, self.labels = self.collect_features()

        # Run analysis
        results = self.analyze_dimensionality()

        # Create visualization
        fig = self.visualize_analysis(results)

        return results, fig


class ParallelAnalysis:
    def __init__(self, n_iterations=100, n_jobs=multiprocessing.cpu_count()):
        self.n_iterations = n_iterations
        self.n_jobs = n_jobs

    def _single_iteration(self, data_shape, random_seed=None):
        """Run a single iteration of parallel analysis"""
        if random_seed is not None:
            np.random.seed(random_seed)

        # Generate random normal data
        random_data = np.random.normal(size=data_shape)

        # Fit PCA and return eigenvalues
        pca = PCA(n_components=min(data_shape))
        pca.fit(random_data)
        return pca.explained_variance_

    def fit(self, features):
        """Run parallel analysis with parallel processing"""
        data_shape = features.shape

        # Generate random seeds for reproducibility
        random_seeds = np.random.randint(0, np.iinfo(np.int32).max, size=self.n_iterations)

        # Run iterations in parallel
        random_eigenvalues = Parallel(n_jobs=self.n_jobs)(
            delayed(self._single_iteration)(data_shape, seed)
            for seed in tqdm(random_seeds, desc="Running PA iterations")
        )

        # Stack results
        random_eigenvalues = np.array(random_eigenvalues)

        # Calculate percentiles
        eigenvalue_95 = np.percentile(random_eigenvalues, 95, axis=0)

        return eigenvalue_95