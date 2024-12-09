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
            _ = tracked_model(image.unsqueeze(0).to(self.device))

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
            _ = tracked_model(image.unsqueeze(0).to(self.device))
        
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