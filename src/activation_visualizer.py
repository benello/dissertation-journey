import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Tuple
from pathlib import Path

from src.activation_holder import ActivationHolder

logger = logging.getLogger(__name__)

class ActivationVisualizer:
    """Visualizes activations of neurons throughout the network."""
    
    def __init__(self, model, activations: ActivationHolder):
        """
        Initialize the activation visualizer.
        
        Args:
            model: The neural network model
        """
        self.model = model
        self.device = model.device
        self.model.to(self.device)
        self.activation_holder = activations
    
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
        self.activation_holder.clear()
        
        # Forward pass
        self.model.eval()
        with torch.no_grad():
            _ = self.model(image.unsqueeze(0).to(self.device))

        if selected_channels is None:
            # Default to first few channels
            selected_channels = list(range(min(4, len(self.activation_holder))))
        
        # Create figure
        n_channels = len(selected_channels)
        n_layers = len(self.activation_holder)
        fig = plt.figure(figsize=(3 * n_layers, 3 * n_channels))
        
        # Plot evolution of each selected channel
        for i, channel_idx in enumerate(selected_channels):
            for j, (name, acts) in enumerate(self.activation_holder):
                plt.subplot(n_channels, n_layers + 1, i * (n_layers + 1) + j + 1)
                
                # Get activation for specific channel
                if j == 0:  # First column shows input
                    plt.imshow(image.squeeze(), cmap='gray')
                    plt.title('Input')
                else:
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
        self.activation_holder.clear()
        
        # Forward pass
        self.model.eval()
        with torch.no_grad():
            _ = self.model(image.unsqueeze(0).to(self.device))
        
        results = {}
        for name, acts in self.activation_holder:
            if 'conv' in name.lower():
                # Calculate mean activation for each channel
                channel_means = acts.mean(dim=(1, 2)).numpy()
                # Get indices of top n_channels
                top_channels = np.argpartition(channel_means, -n_channels)[-n_channels:]
                # Sort by activation value
                top_channels = sorted([(idx, channel_means[idx]) 
                                    for idx in top_channels],
                                   key=lambda x: x[1], reverse=True)
                results[name] = top_channels
        
        return results

activations_path = 'activations'

class ActivationSaver:
    """Handles saving and loading of neural network activations."""

    def __init__(self, base_dir: Path):
        """
        Initialize the activation saver.

        Args:
            base_dir: Base directory where activation files will be saved
        """
        self.base_dir = base_dir / activations_path

    def save_activations(self, activation_holder: ActivationHolder,
                         input_name: str = "default"):
        """
        Save activations from each layer to separate files.

        Args:
            activation_holder: Dictionary mapping layer names to activation tensors
            input_name: Name identifier for the inputs being processed
        """
        # Create a subdirectory for this image
        activations_dir = self.base_dir / input_name
        activations_dir.mkdir(parents=True, exist_ok=True)

        # Save activations for each layer
        for layer_name, acts in activation_holder:
            # Clean layer name for filename
            clean_name = self._clean_name(layer_name)

            # Convert activation to numpy and save
            acts_np = acts.numpy()

            # Save metadata about the activation
            metadata = activation_holder.get_metadata(layer_name)

            # Save both the activation array and its metadata
            np.save(activations_dir / f"{clean_name}_activation.npy", acts_np)
            np.save(activations_dir / f"{clean_name}_metadata.npy", metadata)

            logger.info(f"Saved activation for layer {layer_name} with shape {acts_np.shape}")

    def load_activation(self, layer_name: str,
                        input_name: str = "default") -> Tuple[np.ndarray, Dict]:
        """
        Load activation data for a specific layer.

        Args:
            layer_name: Name of the layer whose activation to load
            input_name: Name identifier of the inputs

        Returns:
            Tuple of (activation array, metadata dictionary)
        """
        activations_dir = self.base_dir / input_name
        clean_name = self._clean_name(layer_name)

        # Load activation array and metadata
        act_path = activations_dir / f"{clean_name}_activation.npy"
        metadata_path = activations_dir / f"{clean_name}_metadata.npy"

        if not act_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Activation files for layer {layer_name} not found")

        activation = np.load(act_path)
        metadata = np.load(metadata_path, allow_pickle=True).item()

        return activation, metadata

    @staticmethod
    def _clean_name(name: str) -> str:
        """Clean a layer name for use in filenames."""
        return "".join(c if c.isalnum() else "_" for c in name)