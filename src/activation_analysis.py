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

    def perform_pca_analysis(self, tracked_features: ActivationTracker, labels: np.ndarray, title: str):
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

        self.plot_pca_3d(transformed_data, labels, title)
        self.plot_pca_pairs(transformed_data, labels, title, n_pairs=2)

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=labels, cmap='viridis', alpha=0.6)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(f"{title} (Global PCA)")
        plt.colorbar(scatter, label="Class")
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
                cls_transformed = np.concatenate(transformed_per_class[cls], axis=0)
                plt.figure(figsize=(6, 5))
                plt.scatter(cls_transformed[:, 0], cls_transformed[:, 1], alpha=0.6)
                plt.xlabel("PC1")
                plt.ylabel("PC2")
                plt.title(f"{title} (Class {cls})")
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
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title(title)
        fig.colorbar(scatter, label="Class")
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
            ax.set_xlabel(f"PC{i + 1}")
            ax.set_ylabel(f"PC{j + 1}")
            ax.set_title(f"{title} (PC{i + 1} vs PC{j + 1})")

        plt.tight_layout()
        plt.show()

    def analyse_sample(self, pca_models,
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
            sample_type = "Normal"
        else:
            sample_type = "OOD"

        random_index = random.randint(0, len(data_loader) - 1)
        iterator = iter(data_loader)
        for _ in range(random_index-1):
            next(iterator)

        data_input, label = next(iterator)
        sample_image = data_input[0]

        plt.imshow(sample_image.squeeze().cpu(), cmap='gray')
        plt.title(f"{sample_type} Sample Signal")
        plt.show()

        # Prepare image and extract features
        sample_image = sample_image.unsqueeze(0).to(self.device)

        tracker = ActivationTracker('outputs', self.config)
        with torch.no_grad() , tracker.track(self.model) as tracked_model:
            outputs, _ = tracked_model(sample_image)
            _, pred = outputs.max(1)
            pred_class = pred.item()
        logger.info(f"\nAnalyzing {sample_type} sample (predicted as class {pred_class})")

        sample_feat = torch.cat(tracker.activations[tracker.probe_layers[0]], dim=0)
        sample_feat = sample_feat.reshape(sample_feat.size(0), -1)

        if key is None:
            for key, (pca, transformed_data) in pca_models.items():
                sample_pca = pca.transform(sample_feat)

                self._generate_sample_analysis_image(transformed_data, sample_pca, sample_type, key)
        else:
            pca, train_pca = pca_models[key]
            sample_pca = pca.transform(sample_feat)

            self._generate_sample_analysis_image(train_pca, sample_pca, sample_type, key)

    def _generate_sample_analysis_image(self, train_pca, sample_pca, sample_type, key):
        cov_matrix = np.cov(train_pca, rowvar=False)
        VI = np.linalg.inv(cov_matrix)
        dists = pairwise_distances(sample_pca, train_pca, metric='mahalanobis', VI=VI)[0]
        k = self.config['analysis']['k_neighbour']  # number of nearest neighbors
        nearest_idx = np.argsort(dists)[:k]
        nearest_dists = dists[nearest_idx]

        plt.figure(figsize=(8, 6))
        plt.scatter(train_pca[:, 0], train_pca[:, 1], label="Training Samples", alpha=0.6)
        plt.scatter(sample_pca[0, 0], sample_pca[0, 1], color='red', label=f"{sample_type} Sample", s=100)
        for idx in nearest_idx:
            plt.plot([sample_pca[0, 0], train_pca[idx, 0]],
                     [sample_pca[0, 1], train_pca[idx, 1]],
                     'k--', alpha=0.5)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(f"{sample_type} Sample Analysis with {key} PCA")
        plt.legend()
        plt.show()

        logger.info(f"Mahalanobis Distances to {k} nearest neighbors:{nearest_dists}")