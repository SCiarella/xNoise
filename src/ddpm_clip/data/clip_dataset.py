"""
Dataset class for CLIP-embedded image data.
"""

import csv
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional, Callable, Any


class CLIPDataset(Dataset):
    """
    PyTorch Dataset for images with precomputed CLIP embeddings.

    Efficient version: loads images on-the-fly instead of preloading all into memory.

    Args:
        csv_path: Path to CSV file containing image paths and CLIP embeddings
        img_transforms: Transforms to apply to images when loading
        random_transforms: Random augmentation transforms to apply on __getitem__
        clip_features: Dimension of CLIP embeddings (default: 512)
        preprocessed_clip: Whether CLIP embeddings are preprocessed in CSV
        clip_model: CLIP model for on-the-fly encoding (required if preprocessed_clip=False)
        clip_preprocess: CLIP preprocessing (required if preprocessed_clip=False)
        device: Device to load tensors onto (use 'cpu' for multi-worker DataLoader)
    """

    def __init__(self,
                 csv_path: str,
                 img_transforms: Optional[Callable] = None,
                 random_transforms: Optional[Callable] = None,
                 clip_features: int = 512,
                 preprocessed_clip: bool = True,
                 clip_model: Optional[Any] = None,
                 clip_preprocess: Optional[Callable] = None,
                 device: str = 'cpu'):
        self.img_transforms = img_transforms
        self.random_transforms = random_transforms
        self.preprocessed_clip = preprocessed_clip
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.device = device

        # Store image paths instead of loaded images
        self.img_paths = []
        self.labels_list = []

        # Single pass through CSV - much faster
        with open(csv_path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                self.img_paths.append(row[0])

                if preprocessed_clip:
                    # Parse embedding directly as list of floats
                    label = [float(x) for x in row[1:]]
                    self.labels_list.append(label)

        # Convert labels to contiguous tensor on target device for fast indexing
        if preprocessed_clip:
            self.labels = torch.tensor(self.labels_list,
                                       dtype=torch.float32,
                                       device=device)
            del self.labels_list  # Free memory

    def __getitem__(self, idx: int):
        # Load image on-the-fly (lazy loading)
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.img_transforms is not None:
            img = self.img_transforms(img)

        # Apply random augmentations
        if self.random_transforms is not None:
            img = self.random_transforms(img)

        if self.preprocessed_clip:
            label = self.labels[idx]
        else:
            if self.clip_model is None or self.clip_preprocess is None:
                raise ValueError(
                    'clip_model and clip_preprocess must be provided when '
                    'preprocessed_clip=False')
            batch_img = img[None, :, :, :]
            encoded_imgs = self.clip_model.encode_image(
                self.clip_preprocess(batch_img))
            label = encoded_imgs.to(self.device).float()[0]

        return img, label

    def __len__(self) -> int:
        return len(self.img_paths)
