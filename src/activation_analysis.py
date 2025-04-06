import logging
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import pairwise_distances
from torch.utils.data import Dataset

from src.activation_tracking import ActivationTracker
from src.novel_loader import NovelDataset

logger = logging.getLogger(__name__)

class ActivationAnalysis:
    """
    A class to extract features from a CNN model and perform various analyses
    including PCA, visualization, and out-of-distribution (OOD) analysis.
    """

    def __init__(self, model: nn.Module, config):
        """
        Initialize the FeatureAnalyzer.

        Args:
            model (nn.Module): Trained CNN model.
            config (Any): Model configuration (e.g., hyperparameters, paths).
        """
        self.model = model
        self.device = model.device
        self.config = config

    def perform_pca_analysis(self, tracked_features: ActivationTracker, labels: np.ndarray):
        pca_models = {}
        n_components = self.config['analysis']['pca_n_components']

        global_pca = IncrementalPCA(n_components=n_components)

        for batch in tracked_features.get_activations(None):
            feature_batch = batch.reshape(batch.size(0), -1)
            global_pca.partial_fit(feature_batch)

        transformed_activations = []
        for batch in tracked_features.get_activations(None):
            feature_batch = batch.reshape(batch.size(0), -1)
            transformed_activations.append(global_pca.transform(feature_batch))

        transformed_data = np.concatenate(transformed_activations, axis=0)

        title = f'{tracked_features.probe_layers[0]}(Global PCA)'
        self.plot_pca_3d(transformed_data, labels, title)
        self.plot_pca_pairs(transformed_data, labels, title, n_pairs=2)

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=labels, cmap='viridis', alpha=0.6)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title(title)
        plt.colorbar(scatter, label='Class')
        plt.show()

        pca_models[-1] = global_pca, transformed_data

        # Per-class PCA
        unique_classes = np.unique(labels)

        for cls in unique_classes:
            pca_models[cls] = IncrementalPCA(n_components=n_components), None # need to keep certain batches to increase n_components

        label_idx = 0
        buffers = defaultdict(list)
        for batch in tracked_features.get_activations(None):
            feature_batch = batch.reshape(batch.size(0), -1)
            batch_size = feature_batch.shape[0]
            # Get the corresponding labels for the current batch.
            batch_labels = labels[label_idx:label_idx + batch_size]
            for cls in unique_classes:
                feature_class = np.where(batch_labels == cls)[0]
                if np.any(feature_class):
                    feats = feature_batch[feature_class]
                    buffers[cls].append(feats)

                    cat_buffer = torch.cat(buffers[cls], dim=0)
                    # Check if there is enough samples.
                    if cat_buffer.shape[0] < n_components:
                        continue

                    pca_models[cls][0].partial_fit(cat_buffer)
                    buffers[cls] = []
            label_idx += batch_size

        # Second pass: transform the activations for each class.
        transformed_per_class = {cls: [] for cls in unique_classes}
        label_idx = 0
        for batch in tracked_features.get_activations(None):
            feature_batch = batch.reshape(batch.size(0), -1).cpu().numpy()
            batch_size = feature_batch.shape[0]
            batch_labels = labels[label_idx:label_idx + batch_size]
            for cls in unique_classes:
                feature_class = np.where(batch_labels == cls)[0]
                if np.any(feature_class):
                    test = feature_batch[feature_class]
                    transformed_batch = pca_models[cls][0].transform(feature_batch[feature_class])
                    transformed_per_class[cls].append(transformed_batch)
            label_idx += batch_size

        # First two principal components of each class
        for cls in unique_classes:
            if transformed_per_class[cls]:
                title =  f'{tracked_features.probe_layers[0]}(Class {cls})'
                cls_transformed = np.concatenate(transformed_per_class[cls], axis=0)
                plt.figure(figsize=(6, 5))
                plt.scatter(cls_transformed[:, 0], cls_transformed[:, 1], alpha=0.6)
                plt.xlabel('PC1')
                plt.ylabel('PC2')
                plt.title(title)
                plt.show()

                pca_models[cls] = pca_models[cls][0], cls_transformed

        return pca_models

    @staticmethod
    def plot_pca_3d(transformed: np.ndarray, labels: np.ndarray, title: str):
        """
        Plot a 3D scatter plot of the first three principal components.

        Args:
            transformed (np.ndarray): PCA-transformed data with at least 3 components.
            labels (np.ndarray): Class labels for coloring the points.
            title (str): Plot title.
        """
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(transformed[:, 0], transformed[:, 1], transformed[:, 2],
                             c=labels, cmap='viridis', alpha=0.6)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title(title)
        fig.colorbar(scatter, label='Class')
        plt.show()

    @staticmethod
    def plot_pca_pairs(transformed: np.ndarray, labels: np.ndarray, title: str, n_pairs: int = 3):
        """
        Plot multiple 2D scatter plots for pairs of principal components.

        By default, plots PC1 vs PC2, PC1 vs PC3, and PC2 vs PC3.

        Args:
            transformed (np.ndarray): PCA-transformed data.
            labels (np.ndarray): Class labels.
            title (str): Base title for the plots.
            n_pairs (int): Number of PC pairs to plot (max 3).
        """
        # Define the default pairs
        pairs = [(0, 1), (0, 2), (1, 2)]
        n_pairs = min(n_pairs, len(pairs))
        fig, axes = plt.subplots(1, n_pairs, figsize=(6 * n_pairs, 5))

        for ax, (i, j) in zip(axes, pairs[:n_pairs]):
            sc = ax.scatter(transformed[:, i], transformed[:, j], c=labels, cmap='viridis', alpha=0.6)
            ax.set_xlabel(f'PC{i + 1}')
            ax.set_ylabel(f'PC{j + 1}')
            ax.set_title(f'{title} (PC{i + 1} vs PC{j + 1})')

        plt.tight_layout()
        plt.show()

    def analyse_samples(self, pca_models,
                        labels: torch.Tensor,
                        data_loader: Dataset,
                        key: int = None):
        """
        Analyze a sample (either a normal or OOD sample) using the global PCA model.

        The method computes Mahalanobis distances (in the PCA space) between the sample’s
        fc1_relu features and those of the training set, then visualizes the projection and
        nearest neighbors.
        """
        # Determine sample type and obtain the image
        if not isinstance(data_loader, NovelDataset):
            sample_type = 'Normal'
        else:
            sample_type = 'OOD'

        random_index = random.randint(0, len(data_loader) - 1)
        iterator = iter(data_loader)
        for _ in range(random_index-1):
            next(iterator)

        data_input, label = next(iterator)
        sample_image = data_input[0].unsqueeze(0)

        # Prepare image and extract features
        self.analyse_sample(pca_models, labels, sample_image, sample_type, key)


    def analyse_sample(self, pca_models, labels, sample_image, sample_type, key: int = None):
        plt.imshow(sample_image.squeeze().squeeze().cpu(), cmap='gray')
        plt.title(f'{sample_type} Sample Signal')
        plt.show()

        # Prepare image and extract features
        sample_image = sample_image.to(self.device)

        tracker = ActivationTracker('outputs', self.config)
        with torch.no_grad() , tracker.track(self.model) as tracked_model:
            outputs, _ = tracked_model(sample_image)
            _, pred = outputs.max(1)
            pred_class = pred.item()
        logger.info(f'\nAnalyzing {sample_type} sample (predicted as class {pred_class})')

        sample_feat = torch.cat(tracker.activations[tracker.probe_layers[0]], dim=0)
        sample_feat = sample_feat.reshape(sample_feat.size(0), -1)

        if key is None:
            for key, (pca, transformed_data) in pca_models.items():
                sample_pca = pca.transform(sample_feat)

                self._generate_sample_analysis_image(transformed_data, sample_pca, sample_type, key, labels)
        else:
            pca, train_pca = pca_models[key]
            sample_pca = pca.transform(sample_feat)

            self._generate_sample_analysis_image(train_pca, sample_pca, sample_type, key, labels)

    def _generate_sample_analysis_image(self, transformed_data, sample_data, sample_type, key, labels=None):
        # Set labels to None if key is -1 (global PCA)
        labels = None if key != -1 else labels

        # Calculate Mahalanobis distances
        cov_matrix = np.cov(transformed_data, rowvar=False)
        VI = np.linalg.inv(cov_matrix)
        dists = pairwise_distances(sample_data, transformed_data, metric='mahalanobis', VI=VI)[0]

        # Get k nearest neighbors
        k = self.config['analysis']['k_neighbour']
        nearest_idx = np.argsort(dists)[:k]
        nearest_dists = dists[nearest_idx]

        # Create visualization with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(
            1, 3,
            figsize=(18, 6),
            gridspec_kw={'width_ratios': [2, 1, 1]}
        )

        # Main scatter plot with PCA projection
        if labels is not None:
            unique_classes = np.unique(labels)
            cmap = plt.cm.get_cmap('viridis', len(unique_classes))

            for i, cls in enumerate(unique_classes):
                mask = labels == cls
                ax1.scatter(transformed_data[mask, 0], transformed_data[mask, 1],
                            color=cmap(i), alpha=0.5, label=f'Class {cls}')
        else:
            ax1.scatter(transformed_data[:, 0], transformed_data[:, 1], alpha=0.4, label='Training Data')

        # Highlight the sample point
        ax1.scatter(sample_data[0, 0], sample_data[0, 1], color='red',
                    marker='*', s=200, label=f'{sample_type} Sample')

        # Highlight nearest neighbors
        nn_classes = []
        for i, idx in enumerate(nearest_idx):
            if labels is not None:
                nn_classes.append(int(labels[idx]))
                nn_color = cmap(int(labels[idx]))
            else:
                nn_color = 'green'

            # Plot connection lines to nearest neighbors
            ax1.plot([sample_data[0, 0], transformed_data[idx, 0]],
                     [sample_data[0, 1], transformed_data[idx, 1]],
                     'k--', alpha=0.4)

            # Mark nearest neighbors with numbered points
            ax1.scatter(transformed_data[idx, 0], transformed_data[idx, 1],
                        color=nn_color, edgecolor='black', s=100, zorder=10)
            ax1.text(transformed_data[idx, 0], transformed_data[idx, 1], str(i + 1),
                     ha='center', va='center', fontsize=8, fontweight='bold')

        ax1.set_xlabel('PC1', fontsize=12)
        ax1.set_ylabel('PC2', fontsize=12)
        ax1.set_title(f'{sample_type} Sample Analysis with {'Global' if key == -1 else f'Class {key}'} PCA',
                      fontsize=14)
        ax1.legend(loc='upper right')

        # Plot nearest neighbor distances
        bars = ax2.barh(range(k), nearest_dists[::-1], color='skyblue')
        ax2.set_yticks(range(k))
        ax2.set_yticklabels([f'#{k - i}' for i in range(k)])
        ax2.set_xlabel('Mahalanobis Distance', fontsize=12)
        ax2.set_ylabel('Nearest Neighbor', fontsize=12)
        ax2.set_title('Nearest Neighbor Distances', fontsize=14)

        # Add class labels to bars if available
        if labels is not None:
            for i, (bar, idx) in enumerate(zip(bars, nearest_idx[::-1])):
                ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                         f'Class {labels[nearest_idx[k - i - 1]]}',
                         va='center', fontsize=10)

        ax3.hist(dists, bins=30, color='lightgreen', alpha=0.7, edgecolor='black')
        ax3.set_title('Distribution of Distances', fontsize=14)
        ax3.set_xlabel('Mahalanobis Distance', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)

        plt.tight_layout()
        plt.show()

        # Log structured information about nearest neighbors
        logger.info(f'\n{'=' * 50}')
        logger.info(
            f'NEAREST NEIGHBORS ANALYSIS ({sample_type} Sample, {'Global' if key == -1 else f'Class {key}'} PCA)')
        logger.info(f'{'=' * 50}')
        logger.info(f'{'#':<4}{'Distance':<12}{'Class':<8}')
        logger.info(f'{'-' * 30}')

        for i, (idx, dist) in enumerate(zip(nearest_idx, nearest_dists), 1):
            class_label = labels[idx] if labels is not None else key
            logger.info(f'{i:<4}{dist:<12.4f}{class_label:<8}')
        logger.info(f'{'=' * 50}\n')