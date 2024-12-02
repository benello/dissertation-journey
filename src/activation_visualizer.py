import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Tuple
from pathlib import Path
from numpy.lib import format
from sympy import ceiling

logger = logging.getLogger(__name__)

class ActivationVisualizer:
    """Visualizes activations of neurons throughout the network."""
    
    def __init__(self, model, activations):
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
        self._file_handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Close all handles when done
        if self._file_handle is not None:
            self._file_handle.close()

    def save_activations(self, activation_holder, input_name: str = 'default'):
        """Save activations in chunks with proper file handling."""
        activations_dir = self.base_dir / input_name
        activations_dir.mkdir(parents=True, exist_ok=True)

        h5_path = activations_dir / 'activations.h5'
        if self._file_handle is None:
            self._file_handle = h5py.File(h5_path, 'w')

        for layer_name, acts in activation_holder:
            clean_name = self._clean_name(layer_name)
            acts_np = acts.numpy()

            # Create or resize dataset
            if clean_name in self._file_handle:
                dset = self._file_handle[clean_name]
                current_size = dset.shape[0]
                dset.resize((current_size + acts_np.shape[0],) + acts_np.shape[1:])
            else:
                maxshape = (None,) + acts_np.shape[1:]  # Allows first dimension to be expandable
                dset = self._file_handle.create_dataset(
                    clean_name,
                    data=acts_np,
                    maxshape=maxshape,
                    chunks=True
                )

            # Write new data
            if clean_name in self._file_handle:
                current_size = dset.shape[0] - acts_np.shape[0]
                dset[current_size:] = acts_np

            # Save metadata
            metadata = activation_holder.get_metadata(layer_name)
            dset.attrs['metadata'] = str(metadata)

    def load_activations(self, input_name: str = 'default', layer_name: str = None, batch_size: int = 200):
        """Load activations in chunks with memory mapping."""
        activations_dir = self.base_dir / input_name
        h5_path = activations_dir / 'activations.h5'

        with h5py.File(h5_path, 'r') as f:
            clean_name = self._clean_name(layer_name)
            # Filter layers if specified
            layers = [clean_name] if layer_name else list(f.keys())

            for layer in layers:
                dset = f[layer]
                total_size = dset.shape[0]
                metadata = dset.attrs.get('metadata')

                # Yield batches
                for batch_idx in range(0, total_size, batch_size):
                    end_idx = min(batch_idx + batch_size, total_size)
                    yield {
                        'layer_name': layer,
                        'data': dset[batch_idx:end_idx],
                        'metadata': metadata,
                        'batch_info': {
                            'start': batch_idx,
                            'end': end_idx,
                            'total': total_size
                        }
                    }

    @staticmethod
    def _clean_name(name: str) -> str:
        """Clean a layer name for use in filenames."""
        return "".join(c if c.isalnum() else "_" for c in name)