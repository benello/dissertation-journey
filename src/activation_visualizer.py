import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Tuple
from collections import defaultdict
from pathlib import Path

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
        self.activations = defaultdict(list)
        self._attach_hooks()
    
    def _attach_hooks(self):
        """Attach forward hooks to all convolutional and ReLU layers."""
        def hook_fn(name):
            def hook(module, input, output):
                self.activations[name].append(output.detach().cpu())
            return hook

        # Attach hooks to each layer we want to visualize
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.ReLU)):
                module.register_forward_hook(hook_fn(name))
    
    def clear_activations(self):
        """Clear stored activations."""
        self.activations.clear()
    
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
        self.clear_activations()
        
        # Forward pass
        self.model.eval()
        with torch.no_grad():
            _ = self.model(image.unsqueeze(0).to(self.device))
        
        # Filter conv layers only
        conv_activations = {name: acts for name, acts in self.activations.items() 
                          if 'conv' in name.lower()}
        
        if selected_channels is None:
            # Default to first few channels
            selected_channels = list(range(min(4, len(conv_activations))))
        
        # Create figure
        n_channels = len(selected_channels)
        n_layers = len(conv_activations)
        fig = plt.figure(figsize=(3 * n_layers, 3 * n_channels))
        
        # Plot evolution of each selected channel
        for i, channel_idx in enumerate(selected_channels):
            for j, (name, acts) in enumerate(conv_activations.items()):
                plt.subplot(n_channels, n_layers + 1, i * (n_layers + 1) + j + 1)
                
                # Get activation for specific channel
                if j == 0:  # First column shows input
                    plt.imshow(image.squeeze().cpu(), cmap='gray')
                    plt.title('Input')
                else:
                    act = acts[0][0]  # Get first batch
                    if channel_idx < act.shape[0]:
                        plt.imshow(act[channel_idx].cpu(), cmap='viridis')
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
        self.clear_activations()
        
        # Forward pass
        self.model.eval()
        with torch.no_grad():
            _ = self.model(image.unsqueeze(0).to(self.device))
        
        results = {}
        for name, acts in self.activations.items():
            if 'conv' in name.lower():
                act = acts[0][0]  # Get first batch
                # Calculate mean activation for each channel
                channel_means = act.mean(dim=(1, 2)).numpy()
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

    def save_activations(self, activations: Dict[str, List[torch.Tensor]],
                         image_name: str = "default"):
        """
        Save activations from each layer to separate files.

        Args:
            activations: Dictionary mapping layer names to activation tensors
            image_name: Name identifier for the input image being processed
        """
        # Create a subdirectory for this image
        image_dir = self.base_dir / image_name
        image_dir.mkdir(parents=True, exist_ok=True)

        # Save activations for each layer
        for layer_name, acts in activations.items():
            # Clean layer name for filename
            clean_name = self._clean_name(layer_name)

            # Convert activation to numpy and save
            act_array = acts[0].numpy()  # Get first (and only) batch

            # Save metadata about the activation
            metadata = {
                "layer_name": layer_name,
                "shape": act_array.shape,
                "mean": float(np.mean(act_array)),
                "std": float(np.std(act_array)),
                "max": float(np.max(act_array)),
                "min": float(np.min(act_array))
            }

            # Save both the activation array and its metadata
            np.save(image_dir / f"{clean_name}_activation.npy", act_array)
            np.save(image_dir / f"{clean_name}_metadata.npy", metadata)

            logger.info(f"Saved activation for layer {layer_name} with shape {act_array.shape}")

    def load_activation(self, layer_name: str,
                        image_name: str = "default") -> Tuple[np.ndarray, Dict]:
        """
        Load activation data for a specific layer.

        Args:
            layer_name: Name of the layer whose activation to load
            image_name: Name identifier of the input image

        Returns:
            Tuple of (activation array, metadata dictionary)
        """
        image_dir = self.base_dir / image_name
        clean_name = self._clean_name(layer_name)

        # Load activation array and metadata
        act_path = image_dir / f"{clean_name}_activation.npy"
        metadata_path = image_dir / f"{clean_name}_metadata.npy"

        if not act_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Activation files for layer {layer_name} not found")

        activation = np.load(act_path)
        metadata = np.load(metadata_path, allow_pickle=True).item()

        return activation, metadata

    def list_saved_images(self) -> List[str]:
        """List all image directories in the base directory."""
        return [d.name for d in self.base_dir.iterdir() if d.is_dir()]

    def list_saved_layers(self, image_name: str) -> List[str]:
        """List all layers saved for a specific image."""
        image_dir = self.base_dir / image_name
        if not image_dir.exists():
            raise FileNotFoundError(f"No saved activations found for image {image_name}")

        return [f.stem.replace('_activation', '')
                for f in image_dir.glob('*_activation.npy')]

    @staticmethod
    def _clean_name(name: str) -> str:
        """Clean a layer name for use in filenames."""
        return "".join(c if c.isalnum() else "_" for c in name)