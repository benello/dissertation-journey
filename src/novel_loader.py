import torch
from torch.utils.data import Dataset
import struct
import numpy as np

from pathlib import Path
from src.novel_generator import image_filename, label_filename


class NovelDataset(Dataset):
    def __init__(self, data_dir):
        """
        Args:
            data_dir (string): Directory with the ubyte files
        """
        self.data_dir = Path(data_dir)

        # Load images
        with open(self.data_dir / image_filename, 'rb') as f:
            magic, size, rows, cols = struct.unpack('>IIII', f.read(16))
            self.images = np.frombuffer(f.read(), dtype=np.uint8,).reshape(size, rows, cols).copy()  # np.frombuffer() returns an immutable array

        # Load labels
        with open(self.data_dir / label_filename, 'rb') as f:
            magic, size = struct.unpack('>II', f.read(8))
            self.labels = np.frombuffer(f.read(), dtype=np.uint8).copy()   # np.frombuffer() returns an immutable array

        self.images = self.images.astype(np.float32) / 255.0  # Normalize to [0, 1]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Convert to tensor
        image = torch.from_numpy(image)

        # Add channel dimension
        image = image.unsqueeze(0)  # Shape becomes [1, H, W]

        return image, label