import os
import h5py
import numpy as np
from tqdm import tqdm
from PIL import Image
from multiprocessing import Pool, cpu_count
from functools import partial
import time


def process_image(args, image_size):
    """
    Load and preprocess a single image - optimized for speed.
    """
    idx, img_path, label = args
    try:
        # NEAREST is fastest, no interpolation
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = img.resize(image_size, Image.NEAREST)  # Fastest resize
        return idx, np.array(img, dtype=np.uint8), label
    except Exception:
        # Silently skip errors for speed
        return idx, None, label


def create_hdf5(root_dir,
                output_file,
                image_size=(128, 128),
                split_name='train',
                num_workers=None):
    """
    Ultra-fast HDF5 conversion - no compression, maximum speed.
    """
    if num_workers is None:
        num_workers = cpu_count()

    print(f'Starting with {num_workers} workers...')
    start_time = time.time()

    # Collect all image paths and labels
    classes = sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    image_paths = []
    labels = []

    print(f"[INFO] Scanning '{root_dir}' ...")
    for cls_name in classes:
        cls_dir = os.path.join(root_dir, cls_name)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(cls_dir, fname))
                labels.append(class_to_idx[cls_name])

    n_images = len(image_paths)
    print(f'[INFO] Found {n_images} images in {len(classes)} classes.')

    # Create HDF5 with NO compression for maximum speed
    with h5py.File(output_file, 'w') as h5f:
        h5f.attrs['split'] = split_name
        h5f.attrs['num_classes'] = len(classes)
        h5f.create_dataset('class_names', data=np.string_(classes))

        img_shape = (image_size[1], image_size[0], 3)

        # NO COMPRESSION - pure speed
        h5_imgs = h5f.create_dataset(
            'images',
            shape=(n_images, *img_shape),
            dtype=np.uint8,
            chunks=(1, *img_shape)  # Single image chunks
        )
        h5_labels = h5f.create_dataset('labels',
                                       shape=(n_images, ),
                                       dtype=np.int32)

        # Process with all cores
        print('[INFO] Processing images...')
        processing_start = time.time()

        process_func = partial(process_image, image_size=image_size)

        # Pre-allocate for batch writing
        batch_size = 500
        processed_count = 0

        with Pool(num_workers) as pool:
            for batch_start in tqdm(range(0, n_images, batch_size),
                                    desc='Processing'):
                batch_end = min(batch_start + batch_size, n_images)
                batch_args = [(i, image_paths[i], labels[i])
                              for i in range(batch_start, batch_end)]

                # Process in parallel with small chunksize for responsiveness
                results = pool.map(process_func, batch_args, chunksize=50)

                # Prepare batch arrays
                valid_results = [(idx, img, lbl) for idx, img, lbl in results
                                 if img is not None]

                if valid_results:
                    # Extract data
                    indices, images, lbls = zip(*valid_results)

                    # Write batch
                    for i, (idx, img, lbl) in enumerate(valid_results):
                        h5_imgs[idx] = img
                        h5_labels[idx] = lbl
                        processed_count += 1

        processing_time = time.time() - processing_start
        total_time = time.time() - start_time

        print(f'\n[SUCCESS] Processed {processed_count}/{n_images} images')
        print(
            f'[TIMING] Processing: {processing_time:.1f}s ({processed_count/processing_time:.1f} img/s)'
        )
        print(f'[TIMING] Total: {total_time:.1f}s')
        print(f'[INFO] Saved to {output_file}')


if __name__ == '__main__':
    base_dir = 'ImageNet1k'

    # Process with smaller images for speed
    for split in ['train', 'test']:
        split_dir = os.path.join(base_dir, split)
        if os.path.exists(split_dir):
            output_path = f'{split}_images.h5'
            create_hdf5(
                split_dir,
                output_path,
                image_size=(128, 128),  # Smaller = faster
                split_name=split,
                num_workers=None)
