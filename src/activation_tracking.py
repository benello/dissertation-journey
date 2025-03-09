import shutil
from collections import defaultdict
from pathlib import Path
from contextlib import contextmanager
import logging

import torch
from datetime import datetime

logger = logging.getLogger(__name__)

activations_path = 'activations'

class ActivationTracker:
    """Simple activation tracking system"""

    def __init__(self, base_dir, config, batch_size=40):
        self.save_dir = Path(base_dir) / activations_path
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.batch_size = batch_size
        self.activations = defaultdict(list)
        self.hooks = []
        self.probe_layers = config['tracker']['probe_layers']
        self.sub_path = None

    def __iter__(self):
        """
        Iterate over (layer_name, activation) pairs.
        Yields tuples of (layer_name, latest_activation_tensor).
        """
        if self.sub_path is not None:
            raise Exception("Tracker has been called with saving mode")
        for layer_name in self.activations:
            if 'conv' in layer_name.lower():  # Return conv layers only
                yield layer_name, torch.cat(self.activations[layer_name], dim=0)

    def set_layers_to_probe(self, layer_types):
        """Override which layers to track."""
        self.probe_layers = layer_types

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
    def track(self, model, filename: str = None):
        """Context manager for tracking model activations."""
        if self.probe_layers is None:
            raise ValueError("Must call set_layers_to_track before tracking")

        self.sub_path = filename

        if filename is not None:
            dir = (self.save_dir / self.sub_path)

            if dir.exists():
                shutil.rmtree(dir)

            dir.mkdir(parents=True, exist_ok=True)

        try:
            # Setup tracking
            for name, module in model.named_modules():
                if name in self.probe_layers:
                    if isinstance(module, torch.nn.ReLU):
                        name = f'{name}_relu'
                    handle = module.register_forward_hook(self._create_hook(name))
                    self.hooks.append(handle)

            yield model

        finally:
            # Cleanup
            for hook in self.hooks:
                hook.remove()
            self.hooks = []

            if filename is not None:
                self._save_batch()

    def get_activations(self, layer_name: str|None):
        """Generator that yields concatenated activations for a layer from disk one batch at a time."""
        if layer_name is None:
            layer_name = self.probe_layers[0]

        # Look for all saved batch files recursively in the save_dir
        batch_files = sorted(self.save_dir.rglob("*.pt"))
        if not batch_files and layer_name not in self.activations :
            raise KeyError(f"No saved activation files found for layer: {layer_name}")

        for file in batch_files:
            # Load the saved batch
            batch = torch.load(file, weights_only=False)    # Not safe for real world use
            if layer_name in batch.keys():
                # Yield the concatenated tensor for this batch
                yield torch.cat(batch[layer_name], dim=0)

        # Activations that have not been saved yet
        if self.activations[layer_name] is not None and len(self.activations[layer_name]) > 0:
            yield torch.cat(self.activations[layer_name], dim=0)

    def save(self, name):
        """Save current activations to disk."""
        sanitized_name = self._sanitize_name(name)
        save_path = self.save_dir / self.sub_path / f"{sanitized_name}.pt"
        torch.save(self.activations, save_path)
        logger.info(f"Saved activations to {save_path}")

    def clear(self):
        """Clear all stored activations."""
        self.activations.clear()

    def _save_batch(self):
        """Internal method to save and clear current batch."""
        if self.activations:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self.save(f"batch_{timestamp}")
            self.clear()

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Clean a layer name for use in filenames."""
        return "".join(c if c.isalnum() else "_" for c in name)