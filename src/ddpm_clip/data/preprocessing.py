"""
Preprocessing utilities for CLIP embeddings.
"""

import os
import csv
import time
from typing import List, Tuple, Any
import torch
from PIL import Image


def extract_clip_embeddings(image_paths: List[str],
                            clip_model: Any,
                            clip_preprocess: Any,
                            csv_path: str,
                            device: str = 'cuda',
                            batch_size: int = 64,
                            skip_existing: bool = True,
                            verbose: bool = True) -> Tuple[int, float]:
    """
    Extract CLIP embeddings from images and save to CSV file.

    Args:
        image_paths: List of paths to image files
        clip_model: Loaded CLIP model for encoding images
        clip_preprocess: CLIP preprocessing transform
        csv_path: Path where to save/append the CSV file
        device: Device to run CLIP model on (default: 'cuda')
        batch_size: Number of images to process in each batch (default: 64)
        skip_existing: Whether to skip images already in the CSV (default: True)
        verbose: Whether to print progress information (default: True)

    Returns:
        tuple: (num_processed, total_time)
    """
    # Check if CSV exists and load existing paths
    existing_paths = set()
    if skip_existing and os.path.exists(csv_path):
        if verbose:
            print(f"Found existing CSV file: {csv_path}")
        with open(csv_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                if row:
                    existing_paths.add(row[0])
        if verbose:
            print(f"Found {len(existing_paths)} already processed images")
    else:
        if verbose:
            print(f"Creating new CSV file: {csv_path}")

    # Filter out already processed paths
    if skip_existing:
        remaining_paths = [
            path for path in image_paths if path not in existing_paths
        ]
        if verbose:
            print(f"Processing {len(remaining_paths)} new images "
                  f"(skipping {len(existing_paths)} already done)")
    else:
        remaining_paths = image_paths
        if verbose:
            print(f"Processing {len(remaining_paths)} images")

    if len(remaining_paths) == 0:
        if verbose:
            print('All images already processed!')
        return 0, 0.0

    all_rows = []
    start_time = time.time()

    # Process in batches
    for batch_start in range(0, len(remaining_paths), batch_size):
        batch_end = min(batch_start + batch_size, len(remaining_paths))
        batch_paths = remaining_paths[batch_start:batch_end]

        batch_imgs = []
        valid_paths = []

        # Load and preprocess images in batch
        for path in batch_paths:
            try:
                img = Image.open(path).convert('RGB')
                processed_img = clip_preprocess(img)
                batch_imgs.append(processed_img)
                valid_paths.append(path)
            except Exception as e:
                if verbose:
                    print(f"Error processing {path}: {e}")
                continue

        # Encode batch if any valid images
        if batch_imgs:
            clip_imgs_batch = torch.stack(batch_imgs).to(device)

            with torch.no_grad():
                labels_batch = clip_model.encode_image(clip_imgs_batch)

            # Store results
            for path, label in zip(valid_paths, labels_batch):
                all_rows.append([path] + label.cpu().tolist())

            # Periodic cache clearing
            if (batch_start // batch_size + 1) % 20 == 0:
                torch.cuda.empty_cache()

        # Progress reporting
        if verbose and (batch_start // batch_size + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = batch_end / elapsed
            remaining_time = (len(remaining_paths) -
                              batch_end) / rate if rate > 0 else 0
            print(
                f"Processed {batch_end}/{len(remaining_paths)} new images... "
                f"Rate: {rate:.1f} imgs/sec, ETA: {remaining_time:.0f}s")

    # Write results to CSV
    if all_rows:
        if verbose:
            print(f"Appending {len(all_rows)} new entries to CSV...")

        mode = 'a' if os.path.exists(csv_path) else 'w'
        with open(csv_path, mode, newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerows(all_rows)

        total_time = time.time() - start_time
        if verbose:
            print(
                f"Completed! Processed {len(all_rows)} new images in {total_time:.1f}s"
            )
            print(
                f"Average rate: {len(all_rows)/total_time:.1f} images/second")
            print(
                f"Total images in dataset: {len(existing_paths) + len(all_rows)}"
            )

        return len(all_rows), total_time

    return 0, 0.0
