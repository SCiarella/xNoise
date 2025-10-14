"""
Dataset class for CLIP-embedded image data.
"""

import csv
import h5py
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable


class CLIPDataset(Dataset):
    """
    PyTorch Dataset for images with precomputed CLIP embeddings from HDF5 files.

    Loads images from HDF5 format (created by scripts/images_to_hdf5.py) and
    CLIP embeddings from a CSV file.

    Args:
        h5_path: Path to HDF5 file containing images and labels
        clip_csv_path: Path to CSV file containing CLIP embeddings
        img_transforms: Additional transforms to apply (optional, images are already normalized)
        random_transforms: Random augmentation transforms to apply on __getitem__
        clip_features: Dimension of CLIP embeddings (default: 512)
        device: Device to load tensors onto (use 'cpu' for multi-worker DataLoader)
    """

    def __init__(self,
                 h5_path: str,
                 clip_csv_path: str,
                 img_transforms: Optional[Callable] = None,
                 random_transforms: Optional[Callable] = None,
                 clip_features: int = 512,
                 device: str = 'cpu'):

        self.h5_path = h5_path
        self.img_transforms = img_transforms
        self.random_transforms = random_transforms
        self.device = device

        # Open HDF5 file to get metadata
        with h5py.File(h5_path, 'r') as h5f:
            self.n_images = h5f['images'].shape[0]
            self.image_shape = h5f['images'].shape[1:]
            self.n_classes = h5f.attrs['num_classes']
            self.split = h5f.attrs['split']
            self.class_names = [
                name.decode('utf-8') for name in h5f['class_names'][:]
            ]

        print(f'[CLIPDataset] Loaded HDF5 from {h5_path}')
        print(f'  Split: {self.split}')
        print(f'  Images: {self.n_images}')
        print(f'  Classes: {self.n_classes}')
        print(f'  Image shape: {self.image_shape}')

        # Load CLIP embeddings from CSV
        self.clip_embeddings = []
        print(f'[CLIPDataset] Loading CLIP embeddings from {clip_csv_path}')

        with open(clip_csv_path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                # Parse embedding (skip first column which is path)
                embedding = [float(x) for x in row[1:]]
                self.clip_embeddings.append(embedding)

        # Convert to tensor for fast access
        self.clip_embeddings = torch.tensor(self.clip_embeddings,
                                            dtype=torch.float32,
                                            device=device)

        print(f'  Loaded {len(self.clip_embeddings)} CLIP embeddings')

        if len(self.clip_embeddings) != self.n_images:
            print(
                f'  WARNING: Number of embeddings ({len(self.clip_embeddings)}) '
                f'!= number of images ({self.n_images})')
            print(
                f'  Using minimum: {min(len(self.clip_embeddings), self.n_images)}'
            )
            self.n_images = min(len(self.clip_embeddings), self.n_images)

        # Each worker needs its own file handle
        self.h5_file = None

    def _get_h5_file(self):
        """Get or create HDF5 file handle for this worker."""
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
        return self.h5_file

    def __getitem__(self, idx: int):
        # Get HDF5 file handle
        h5f = self._get_h5_file()

        # Load image from HDF5 (uint8 [H, W, C])
        img_array = h5f['images'][idx]

        # Convert to float tensor and normalize to [-1, 1]
        img = torch.from_numpy(img_array).float() / 255.0
        img = img.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
        img = (img * 2) - 1  # [0, 1] -> [-1, 1]

        # Apply additional transforms if provided
        if self.img_transforms is not None:
            img = self.img_transforms(img)

        # Apply random augmentations
        if self.random_transforms is not None:
            img = self.random_transforms(img)

        # Get CLIP embedding
        label = self.clip_embeddings[idx]

        return img, label

    def __len__(self) -> int:
        return self.n_images

    def __del__(self):
        """Close HDF5 file when dataset is deleted."""
        if self.h5_file is not None:
            self.h5_file.close()
