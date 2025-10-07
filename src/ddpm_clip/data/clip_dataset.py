"""
Dataset class for CLIP-embedded image data.
"""

import csv
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional, Callable, Any, Union


class CLIPDataset(Dataset):
    """
    PyTorch Dataset for images with precomputed CLIP embeddings.

    Args:
        csv_path: Path to CSV file containing image paths and CLIP embeddings
        img_transforms: Transforms to apply to images before storing
        random_transforms: Random augmentation transforms to apply on __getitem__
        clip_features: Dimension of CLIP embeddings (default: 512)
        preprocessed_clip: Whether CLIP embeddings are preprocessed in CSV
        clip_model: CLIP model for on-the-fly encoding (required if preprocessed_clip=False)
        clip_preprocess: CLIP preprocessing (required if preprocessed_clip=False)
        device: Device to load tensors onto
    """

    def __init__(self,
                 csv_path: str,
                 img_transforms: Optional[Callable] = None,
                 random_transforms: Optional[Callable] = None,
                 clip_features: int = 512,
                 preprocessed_clip: bool = True,
                 clip_model: Optional[Any] = None,
                 clip_preprocess: Optional[Callable] = None,
                 device: str = 'cuda'):
        self.imgs = []
        self.preprocessed_clip = preprocessed_clip
        self.random_transforms = random_transforms
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.device = device

        # Count rows first to preallocate labels tensor
        with open(csv_path, newline='') as csvfile:
            num_rows = sum(1 for _ in csv.reader(csvfile))

        if preprocessed_clip:
            self.labels = torch.empty(num_rows,
                                      clip_features,
                                      dtype=torch.float,
                                      device=device)

        # Load data
        with open(csv_path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for idx, row in enumerate(reader):
                img_path = row[0]
                img: Union[Image.Image,
                           torch.Tensor] = Image.open(img_path).convert('RGB')

                if img_transforms is not None:
                    img = img_transforms(img)

                # img_transforms should convert PIL Image to Tensor
                if isinstance(img, torch.Tensor):
                    self.imgs.append(img.to(device))
                else:
                    raise TypeError(
                        f"img_transforms must convert PIL Image to Tensor, got {type(img)}"
                    )

                if preprocessed_clip:
                    label = [float(x) for x in row[1:]]
                    self.labels[idx, :] = torch.FloatTensor(label).to(device)

    def __getitem__(self, idx: int):
        img = self.imgs[idx]

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
        return len(self.imgs)
