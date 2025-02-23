import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Tuple
from pathlib import Path

from torch import nn

from src.activation_tracking import ActivationTracker

logger = logging.getLogger(__name__)

class ActivationVisualizer:
    """Visualizes activations of neurons throughout the network."""
    
    def __init__(self, model):
        """
        Initialize the activation visualizer.
        
        Args:
            model: The neural network model
        """
        self.model = model
        self.device = model.device
        self.model.to(self.device)
        self.tracker = ActivationTracker('outputs')
        self.tracker.set_layers_to_track([nn.Conv2d, nn.ReLU])
    
    def visualize_feature_evolution(self, image: torch.Tensor, digit_label: int = None,
                                  selected_channels: List[int] = None) -> plt.Figure:
        """
        Visualize how features evolve through the network for specific channels.
        
        Args:
            image: Input image tensor
            digit_label: Optional label for the digit
            selected_channels: List of channel indices to visualize
            
        Returns:
            matplotlib Figure object
        """
        # Clear previous activations
        self.tracker.clear()
        
        # Forward pass
        with torch.no_grad(), self.tracker.track(self.model, None, False) as tracked_model:
            tracked_model.eval()
            prediction, _ = tracked_model(image.to(self.device))
        if selected_channels is None:
            # Default to first few channels
            selected_channels = list(range(min(4, 5)))
        
        # Create figure
        n_channels = len(selected_channels)
        n_layers = len(self.tracker)
        fig = plt.figure(figsize=(3 * n_layers, 3 * n_channels))

        # Plot evolution of each selected channel
        for i, channel_idx in enumerate(selected_channels):
            # Plot input image in the first column
            plt.subplot(n_channels, n_layers + 1, i * (n_layers + 1) + 1)
            plt.imshow(image.squeeze(), cmap='gray', vmin=0, vmax=1)
            if i == 0:  # Only add title for the first row
                plt.title('Input')
            plt.axis('off')

            # Plot activations for each layer starting from second column
            for j, (name, acts) in enumerate(self.tracker):
                # Calculate the correct subplot index (j+1 because input image takes first column)
                acts = acts.squeeze()   # Only one image is being processed so remove extra dimension
                subplot_idx = i * (n_layers + 1) + (j + 2)
                plt.subplot(n_channels, n_layers + 1, subplot_idx)

                if channel_idx < acts.shape[0]:
                    plt.imshow(acts[channel_idx], cmap='viridis')
                    plt.title(f'{name}\nChannel {channel_idx}')
                else:
                    plt.text(0.5, 0.5, 'Channel\nnot available',
                             ha='center', va='center')
                plt.axis('off')
        
        plt.tight_layout()
        return fig
    
    def get_most_activated_channels(self, image: torch.Tensor, n_channels: int = 5) -> Dict[str, List[Tuple[int, float]]]:
        """
        Find the channels that are most activated by the input image.
        
        Args:
            image: Input image tensor
            n_channels: Number of top channels to return per layer
            
        Returns:
            Dictionary mapping layer names to lists of (channel_idx, activation_value) tuples
        """
        self.tracker.clear()
        
        # Forward pass
        self.model.eval()
        with torch.no_grad(), self.tracker.track(self.model, None, False) as tracked_model:
            _ = tracked_model(image.to(self.device))
        
        results = {}
        for name, acts in self.tracker:
            if 'conv' in name.lower():
                # Calculate mean activation for each channel
                channel_means = acts.mean(dim=(2, 3)).numpy().squeeze()
                # Get indices of top n_channels
                top_channels = np.argpartition(channel_means, -n_channels)[-n_channels:]
                # Sort by activation value
                top_channels = sorted([(idx, channel_means[idx]) 
                                    for idx in top_channels],
                                   key=lambda x: x[1], reverse=True)
                results[name] = top_channels
        
        return results

    def find_inactive_channels(self, image: torch.Tensor, threshold: float = 0) -> Dict[str, List[int]]:
        """
        Find channels that have no significant activations (effectively "blank" feature maps).

        Args:
            image: Input image tensor
            threshold: Threshold for considering a channel inactive

        Returns:
            Dictionary mapping layer names to lists of inactive channel indices
        """
        self.tracker.clear()

        # Forward pass
        self.model.eval()
        with torch.no_grad(), self.tracker.track(self.model, None, False) as tracked_model:
            _ = tracked_model(image.to(self.device))

        inactive_channels = {}
        for name, acts in self.tracker:
            if 'conv' not in name.lower():
                continue

            acts = acts.squeeze()  # Remove batch dimension

            # Find inactive channels
            inactive = []
            for channel_idx in range(acts.shape[0]):
                feature_map = acts[channel_idx]

                # A channel is considered inactive if:
                # 1. Maximum activation is below threshold OR
                # 2. Standard deviation is below threshold (uniform activation)
                if (feature_map.abs().max() < threshold or
                        torch.std(feature_map) < threshold):
                    inactive.append(channel_idx)

            if len(inactive) > 0:
                inactive_channels[name] = inactive

        return inactive_channels

    def visualize_inactive_channels(self, image: torch.Tensor, threshold: float = 1e-6) -> plt.Figure:
        """
        Create a visualization highlighting inactive channels.

        Args:
            image: Input image tensor
            threshold: Threshold for considering a channel inactive

        Returns:
            matplotlib Figure showing inactive channels
        """
        inactive_channels = self.find_inactive_channels(image, threshold)

        # Count total conv layers and maximum channels for subplot layout
        n_conv_layers = sum(1 for name, _ in self.tracker if 'conv' in name.lower())

        fig = plt.figure(figsize=(4 * n_conv_layers, 4))

        plot_idx = 1
        for name, acts in self.tracker:
            if 'conv' not in name.lower():
                continue

            acts = acts.squeeze()
            inactive = inactive_channels.get(name, [])

            if inactive:
                plt.subplot(1, n_conv_layers, plot_idx)
                plt.title(f'{name}\n{len(inactive)} inactive channels')

                # heatmap channel
                channel_activity = acts.abs().mean(dim=(1, 2))
                plt.imshow(channel_activity.reshape(-1, 1), aspect='auto', cmap='viridis')

                # Highlight inactive
                for idx in inactive:
                    plt.axhline(y=idx, color='r', alpha=0.3)

                plt.colorbar(label='Mean activation')
                plt.xlabel('Channel index')

            plot_idx += 1

        plt.tight_layout()
        return fig