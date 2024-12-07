from contextlib import contextmanager
from pathlib import Path

import torch
from typing import Optional
from collections import defaultdict
import logging

from src.activation_visualizer import ActivationSaver
from src.model import FeatureVisualizerCNN

logger = logging.getLogger(__name__)

class ActivationHolder:
    """Holds and manages neural network layer activations with efficient storage and retrieval."""
    
    def __init__(self, model: FeatureVisualizerCNN, save_dir: Path):
        self._activations = defaultdict(list)
        self._metadata = {}

        self.hooks = []
        self.max_batch_size = 200
        self.current_batch_size = 0
        self.input_name = None
        self.saver = ActivationSaver(save_dir)

        self._attach_hooks(model)

    def __len__(self) -> int:
        """Get number of layers with stored activations."""
        return len(self._activations)

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        self.store(key, value)

    def __iter__(self):
        """
        Iterate over (layer_name, activation) pairs.
        Yields tuples of (layer_name, latest_activation_tensor).
        """
        for layer_name in self._activations:
            if 'conv' in layer_name.lower():    # Return conv layers only
                yield layer_name, self.get(layer_name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()
        self.clear()
        self.saver.__exit__(exc_type, exc_val, exc_tb)

    def _attach_hooks(self, model: FeatureVisualizerCNN):
        """Attach forward hooks to all convolutional and ReLU layers."""
        def hook_fn(name):
            def hook(module, input, output):
                self.store(name, output.squeeze())

            return hook

        # Attach hooks to each layer we want to visualize
        t = model.named_modules()
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                handle = module.register_forward_hook(hook_fn(name))
                self.hooks.append(handle)
            if isinstance(module, torch.nn.ReLU):
                handle = module.register_forward_hook(hook_fn(f'{name}-relu'))
                self.hooks.append(handle)


    def remove_hooks(self):
        """Remove all hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
    
    def store(self, layer_name: str, activation: torch.Tensor) -> None:
        """
        Store activation for a layer.
        
        Args:
            layer_name: Name of the layer
            activation: Activation tensor to store
        """
        # Detach and move to CPU for storage
        activation_cpu = activation.detach().cpu()

        # Store activation
        self._activations[layer_name].append(activation_cpu)
        
        # Update metadata
        if layer_name not in self._metadata:
            self._metadata[layer_name] = {
                'shape': tuple(activation.shape),
                'dtype': str(activation.dtype),
                'mean': 0.0,
                'std': 0.0,
                'min': float('inf'),
                'max': float('-inf'),
            }

        with torch.no_grad():
            current_stats = self._metadata[layer_name]
            current_stats['mean'] += float(activation.mean())
            current_stats['std'] += float(activation.std())
            current_stats['min'] = min(current_stats['min'], float(activation.min()))
            current_stats['max'] = max(current_stats['max'], float(activation.max()))

        logger.debug(f"Stored activation for layer {layer_name} with shape {activation.shape}")

        self.current_batch_size += 1

        # Check if we need to flush to disk
        if self.current_batch_size >= self.max_batch_size:
            self._flush_to_disk(self.input_name)
            self._activations.clear()
            self.current_batch_size = 0

    def get(self, layer_name: str, batch_idx: int = -1) -> torch.Tensor:
        """
        Retrieve activation for a specific layer.
        
        Args:
            layer_name: Name of the layer
            batch_idx: Index of the batch to retrieve (-1 for latest)

        Returns:
            Activation tensor
        """
        if layer_name not in self._activations:
            raise KeyError(f"No activation stored for layer {layer_name}")
            
        activation = self._activations[layer_name][batch_idx].detach().cpu()

        return activation

    def get_channel(self, layer_name: str, channel_idx: int, batch_idx: int = -1) -> torch.Tensor:
        """
        Get activation for a specific channel in a layer.

        Args:
            layer_name: Name of the layer
            channel_idx: Index of the channel to retrieve
            batch_idx: Index of the batch to retrieve (-1 for latest)

        Returns:
            Channel activation tensor
        """
        activation = self.get(layer_name, batch_idx)
        if channel_idx >= activation.shape[1]:
            raise IndexError(f"Channel index {channel_idx} out of bounds for layer {layer_name}")
        return activation[:, channel_idx]

    def clear(self, layer_name: Optional[str] = None) -> None:
        """
        Clear stored activations.
        
        Args:
            layer_name: If provided, clear only this layer's activations
        """
        if layer_name is None:
            self._activations.clear()
            self._metadata.clear()
        else:
            if layer_name in self._activations:
                del self._activations[layer_name]
            if layer_name in self._metadata:
                del self._metadata[layer_name]
    
    def get_metadata(self, layer_name: Optional[str] = None):
        """
        Get metadata for stored activations.
        
        Args:
            layer_name: If provided, get metadata only for this layer
            
        Returns:
            Dictionary of layer metadata
        """
        if layer_name is not None:
            if layer_name not in self._metadata:
                raise KeyError(f"No metadata available for layer {layer_name}")

            return self._metadata[layer_name]

        return self._metadata

    @contextmanager
    def batch_context(self, input_name: str = "default"):
        """Context manager for handling batches of activations."""
        self.input_name = input_name
        try:
            yield self
        finally:
            # Flush any remaining activations
            if self.current_batch_size > 0:
                self._flush_to_disk(input_name)
            self._activations.clear()
            self.input_name = None

    def _flush_to_disk(self, input_name: str = "default"):
        if input_name is None:
            input_name = 'default'

        self.saver.save_activations(self, input_name)
        logger.debug(f"Flushed batch of size {self.current_batch_size} to disk")