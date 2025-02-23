from collections import defaultdict
from pathlib import Path
from contextlib import contextmanager
import logging

import numpy as np
import torch
from datetime import datetime

logger = logging.getLogger(__name__)

activations_path = 'activations'

class ActivationTracker:
    """Simple activation tracking system"""

    def __init__(self, base_dir, batch_size=10):
        self.save_dir = Path(base_dir) / activations_path
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.batch_size = batch_size
        self.activations = defaultdict(list)
        self.hooks = []
        self.track_layers = None
        self.sub_path = None

    def __len__(self) -> int:
        """Get number of layers with stored activations."""
        return len(self.activations)

    def __getitem__(self, key):
        return self.get_activations(key)

    def __iter__(self):
        """
        Iterate over (layer_name, activation) pairs.
        Yields tuples of (layer_name, latest_activation_tensor).
        """
        for layer_name in self.activations:
            if 'conv' in layer_name.lower():    # Return conv layers only
                yield layer_name, self.get_activations(layer_name)

    def set_layers_to_track(self, layer_types):
        """Set which types of layers to track."""
        self.track_layers = layer_types

    def _create_hook(self, layer_name):
        """Create a hook for capturing layer activations."""

        def hook(module, input, output):
            # Store tensor directly after detaching and moving to CPU
            tensor = output.detach().cpu()
            self.activations[layer_name].append(tensor)
            # save if batch size is reached
            if len(self.activations[layer_name]) >= self.batch_size:
                self._save_batch()

        return hook

    @contextmanager
    def track(self, model, filename: str, save_batch: bool = True):
        """Context manager for tracking model activations."""
        if self.track_layers is None:
            raise ValueError("Must call set_layers_to_track before tracking")

        self.sub_path = filename

        if save_batch:
            (self.save_dir / self.sub_path).mkdir(parents=True, exist_ok=True)

        try:
            # Setup tracking
            for name, module in model.named_modules():
                if any(isinstance(module, layer_type) for layer_type in self.track_layers):
                    layer_name = name
                    # Name is the same for activation function
                    if isinstance(module, torch.nn.ReLU):
                        layer_name = f'{name}-relu'

                    handle = module.register_forward_hook(self._create_hook(layer_name))
                    self.hooks.append(handle)

            yield model

        finally:
            # Cleanup
            for hook in self.hooks:
                hook.remove()
            self.hooks = []

            if save_batch:
                self._save_batch()

            self.sub_path = None

    def get_activations(self, layer_name):
        """Get concatenated activations for a layer."""
        if layer_name not in self.activations or not self.activations[layer_name]:
            raise KeyError(f"No activations found for layer: {layer_name}")

        return torch.cat(self.activations[layer_name], dim=0)

    def get_stats(self, layer_name):
        """Get statistics for a layer's activations."""
        tensors = self.get_activations(layer_name)

        with torch.no_grad():
            return {
                'min': float(tensors.min()),
                'max': float(tensors.max()),
            }

    def save(self, experiment_name):
        """Save current activations to disk."""
        sanitized_name = self._clean_name(experiment_name)
        save_path = self.save_dir / self.sub_path / f"{sanitized_name}.pt"
        torch.save(self.activations, save_path)
        logger.info(f"Saved activations to {save_path}")

    def load(self, experiment_name):
        """Load activations from disk."""
        sanitized_name = self._clean_name(experiment_name)
        load_path = self.save_dir / f"{sanitized_name}.pt"
        self.activations = torch.load(load_path)
        logger.info(f"Loaded activations from {load_path}")

    def clear(self):
        """Clear all stored activations."""
        self.activations.clear()

    def _save_batch(self):
        """Internal method to save and clear current batch."""
        if self.activations:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save(f"batch_{timestamp}")
            self.clear()

    @staticmethod
    def _clean_name(name: str) -> str:
        """Clean a layer name for use in filenames."""
        return "".join(c if c.isalnum() else "_" for c in name)