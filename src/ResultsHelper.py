import logging

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import pairwise_distances

from src.activation_tracking import ActivationTracker

def compute_results(sample_image, true_label, model, global_model, test_labels, config):
    # Use ActivationTracker to extract features
    tracker = ActivationTracker('outputs', config)
    model.eval()
    with torch.no_grad(), tracker.track(model) as tracked_model:
        outputs, _ = tracked_model(sample_image.to('cuda'))
        _, pred = outputs.max(1)
        pred_class = pred.item()

    # Concatenate activations from the first probe layer and reshape to 2D
    sample_feat = torch.cat(tracker.activations[tracker.probe_layers[0]], dim=0)
    sample_feat = sample_feat.reshape(sample_feat.size(0), -1)
    sample_feat_np = sample_feat.cpu().numpy()

    pca_model, train_pca = global_model
    # Transform features using the PCA model
    sample_pca = pca_model.transform(sample_feat_np)
    # Compute covariance and its inverse (for Mahalanobis distance)
    cov_matrix = np.cov(train_pca, rowvar=False)
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    dists = pairwise_distances(sample_pca, train_pca, metric='mahalanobis', VI=inv_cov_matrix)[0]
    k = config['analysis']['k_neighbour']
    nearest_idx = np.argsort(dists)[:k]
    nearest_dists = dists[nearest_idx]
    mean_distance = float(np.mean(nearest_dists))

    # Compute entropy of the k nearest neighbors' class labels
    nearest_labels = test_labels[nearest_idx].cpu().numpy()
    unique_classes = np.unique(test_labels.cpu().numpy())
    counts = np.zeros(len(unique_classes))
    for i, unique_cls in enumerate(unique_classes):
        counts[i] = np.sum(nearest_labels == unique_cls)
    probs = counts / k
    probs = probs[probs > 0]  # Filter out zero probabilities
    entropy = -np.sum(probs * np.log2(probs))

    # Compute mean distances for neighbors with correct and incorrect classes
    correct_mask = nearest_labels == true_label
    incorrect_mask = nearest_labels != true_label

    if np.any(correct_mask):
        mean_distance_correct = float(np.mean(nearest_dists[correct_mask]))
    else:
        mean_distance_correct = float('nan')

    if np.any(incorrect_mask):
        mean_distance_incorrect = float(np.mean(nearest_dists[incorrect_mask]))
    else:
        mean_distance_incorrect = float('nan')

    classification_outcome = "Correct" if pred_class == true_label else "Mislabeled"
    threshold = config['analysis'].get('novel_threshold', 8.0)
    novelty_indicator = mean_distance > threshold
    additional_remarks = (
        "Ambiguous neighborhood" if entropy > 1.0 else "Consistent with cluster"
    )

    return {
        "Data Point ID": None,  # to be filled with sample index
        "Original Class": true_label,
        "Predicted Class": pred_class,
        "Nearest Neighbors Classes": ", ".join(str(cls) for cls in nearest_labels),
        "Nearest Neighbors Distances": ", ".join(f"{d:.4f}" for d in nearest_dists),
        "Mean Distance": mean_distance,
        "Mean Distance (Correct Neighbors)": mean_distance_correct,
        "Mean Distance (Incorrect Neighbors)": mean_distance_incorrect,
        "Entropy Value": entropy,
        "Classification Outcome": classification_outcome,
        "Novelty Indicator": novelty_indicator,
        "Additional Remarks": additional_remarks,
    }

def export_csv_data(trainer, model, config, pca_models, test_labels, output_dir):
    metrics_list = []
    sample_index = 0
    for sample in trainer.test_loader:
        sample_images, labels = sample
        for i in range(len(labels)):
            image = sample_images[i].unsqueeze(0)  # Process one image with a batch dimension
            label_val = labels[i].item() if isinstance(labels, torch.Tensor) else labels[i]
            metrics = compute_results(image, label_val, model, pca_models[-1], test_labels, config)
            metrics["Data Point ID"] = sample_index
            metrics_list.append(metrics)
            sample_index += 1

    df = pd.DataFrame(metrics_list)
    csv_path = output_dir / "analysis_data.csv"
    df.to_csv(csv_path, index=False)
    logging.getLogger(__name__).info(f"Exported CSV data to {csv_path}")
