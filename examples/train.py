#!/usr/bin/env python3
"""
CLIP-Conditioned DDPM Training Script

This script provides a complete training pipeline for CLIP-conditioned diffusion models.

Usage:
    python train.py --config ../config/model_small.yaml
    python train.py --config ../config/model_v2.yaml --no-animation
    python train.py --config ../config/model_large.yaml --skip-clip-extraction

Arguments:
    --config: Path to YAML configuration file (required)
    --no-animation: Skip animation generation to save time
    --skip-clip-extraction: Skip CLIP embedding extraction (assumes CSV exists)

Note: Training automatically resumes from the latest checkpoint if one exists.
"""

import os
import sys
import random
import argparse
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for servers
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import save_image, make_grid
import time
import gc

from ddpm_clip.models import UNet, DDPM, EMA, sample_w
from ddpm_clip.data import CLIPDataset, extract_clip_embeddings
from ddpm_clip.utils import (
    save_animation, 
    generation_image, 
    to_image, 
    load_config,
    setup_device,
    load_clip_model,
    load_checkpoint,
    delete_old_checkpoints,
    plot_loss_curve,
    sample_from_text,
    generate_samples
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train CLIP-conditioned DDPM')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML configuration file')
    parser.add_argument('--no-animation', action='store_true',
                        help='Skip animation generation')
    parser.add_argument('--skip-clip-extraction', action='store_true',
                        help='Skip CLIP embedding extraction')
    return parser.parse_args()


def collect_image_paths(dataset_root, dataset_type='imagenet1k'):
    """
    Collect image paths based on dataset type.
    
    Args:
        dataset_root: Root directory of the dataset
        dataset_type: Type of dataset ('tinyimagenet' or 'imagenet1k')
    
    Returns:
        List of image paths
    """
    if dataset_type == 'tinyimagenet':
        data_paths = glob.glob(f"{dataset_root}/train/*/images/*.JPEG")
    elif dataset_type == 'imagenet1k':
        data_paths = glob.glob(f"{dataset_root}/train/*/*.jpg")
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Use 'tinyimagenet' or 'imagenet1k'")
    
    return data_paths


def prepare_dataset(config, device, skip_extraction=False):
    """
    Prepare dataset and extract CLIP embeddings if needed.
    
    Args:
        config: Configuration object containing dataset parameters
        device: Device to run CLIP extraction on
        skip_extraction: Skip CLIP embedding extraction if True
    
    Returns:
        Path to the CSV file containing CLIP embeddings
    """
    import clip
    
    # Get dataset configuration
    dataset_root = config['data']['dataset_root']
    dataset_type = config['data'].get('dataset_type', 'imagenet1k')
    ndata = config['data']['max_samples']
    random_seed = config['data']['random_seed']
    
    print(f"\nPreparing dataset: {dataset_type}")
    print(f"Dataset root: {dataset_root}")
    
    # Collect image paths based on dataset type
    data_paths = collect_image_paths(dataset_root, dataset_type)
    print(f"Found {len(data_paths)} total images")
    
    # Shuffle and limit dataset size
    random.seed(random_seed)
    random.shuffle(data_paths)
    data_paths = data_paths[:ndata]
    
    print(f"Using {len(data_paths)} images for training")
    
    # Extract CLIP embeddings
    csv_path = config['data']['clip_csv_path']
    
    if not skip_extraction:
        print("\nExtracting CLIP embeddings...")
        clip_model, clip_preprocess, _ = load_clip_model(device)
        
        num_processed, time_taken = extract_clip_embeddings(
            data_paths,
            clip_model,
            clip_preprocess,
            csv_path,
            device=device,
            batch_size=config['training']['batch_size'],
            skip_existing=True,
            verbose=True
        )
        
        print(f"\nCLIP embedding extraction complete!")
        print(f"Processed {num_processed} images in {time_taken:.1f}s")
        if num_processed > 0:
            print(f"Average rate: {num_processed/time_taken:.1f} images/second")
        
        # Clean up CLIP model
        del clip_model, clip_preprocess
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print(f"\nSkipping CLIP extraction - using existing CSV: {csv_path}")
    
    return csv_path


def create_dataloaders(config, csv_path, device):
    """Create dataset and dataloader."""
    IMG_CH = config['model']['img_channels']
    IMG_SIZE = config['model']['img_size']
    BATCH_SIZE = config['training']['batch_size']
    CLIP_FEATURES = config['clip']['features']
    
    # Pre-transforms: applied once when loading
    pre_transforms = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)  # Scale to [-1, 1]
    ])
    
    # Random transforms: applied on-the-fly for data augmentation
    random_transforms = transforms.Compose([
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
    ])
    
    print(f"\nCreating dataset...")
    print(f"  Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Batch size: {BATCH_SIZE}")
    
    # Create dataset
    train_data = CLIPDataset(
        csv_path,
        img_transforms=pre_transforms,
        random_transforms=random_transforms,
        clip_features=CLIP_FEATURES,
        preprocessed_clip=config['data']['preprocessed_clip'],
        device='cpu'  # Keep data on CPU for efficient DataLoader
    )

    print(f" Clip dataset created with {len(train_data)} samples")
    
    dataloader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,  # pins CPU memory for faster GPU transfer
        persistent_workers=True,
        prefetch_factor=2  # Prefetch 2 batches per worker
    )
    
    print(f"  Dataset size: {len(train_data)} images")
    print(f"  Batches per epoch: {len(dataloader)}")
    print(f"  Samples per epoch: {len(dataloader) * BATCH_SIZE}")
    
    INPUT_SIZE = (IMG_CH, IMG_SIZE, IMG_SIZE)
    return dataloader, INPUT_SIZE, BATCH_SIZE


def initialize_model(config, device):
    """Initialize DDPM and U-Net model."""
    T = config['diffusion']['timesteps']
    B_start = config['diffusion']['beta_start']
    B_end = config['diffusion']['beta_end']
    IMG_CH = config['model']['img_channels']
    IMG_SIZE = config['model']['img_size']
    
    # Initialize diffusion process
    B = torch.linspace(B_start, B_end, T).to(device)
    ddpm = DDPM(B, device)
    
    print(f"\nDiffusion process initialized:")
    print(f"  Timesteps: {T}")
    print(f"  Beta schedule: {B_start} to {B_end}")
    
    # Initialize U-Net model
    model = UNet(
        T, IMG_CH, IMG_SIZE,
        down_chs=tuple(config['model']['down_channels']),
        t_embed_dim=config['model']['t_embed_dim'],
        c_embed_dim=config['model']['c_embed_dim']
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel initialized:")
    print(f"  Parameters: {num_params:,}")
    print(f"  Architecture: {config['model']['down_channels']}")
    
    torch.set_float32_matmul_precision('high')
    model = model.to(device)
    model_compiled = torch.compile(model)
    print(f"  Model compiled successfully")
    
    return ddpm, model, model_compiled, T


def setup_training(config, model, device):
    """Setup optimizer, scheduler, and EMA."""
    epochs = config['training']['epochs']
    lrate = config['training']['learning_rate']
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lrate)
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=config['training']['lr_scheduler']['eta_min']
    )
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # EMA
    ema_enabled = config['training']['ema']['enabled']
    if ema_enabled:
        ema = EMA(model, decay=config['training']['ema']['decay'])
        print(f"\nEMA enabled with decay: {config['training']['ema']['decay']}")
    else:
        ema = None
        print("\nEMA disabled")
    
    print(f"\nTraining configuration:")
    print(f"  Model name: {config.model_name}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lrate}")
    print(f"  LR scheduler: CosineAnnealingLR (eta_min={config['training']['lr_scheduler']['eta_min']})")
    print(f"  Mixed precision: Enabled (FP16)")
    print(f"  Conditioning drop probability: {config['training']['c_drop_prob']}")
    print(f"  Save frequency: every {config['training']['save_frequency']} epochs")
    
    delete_old = config['training'].get('delete_old_checkpoints', False)
    print(f"  Delete old checkpoints: {delete_old}")
    
    return optimizer, scheduler, ema, epochs, scaler


def train(config, args):
    """Main training function."""
    # Setup
    device = setup_device()
    
    # Create directories
    config.create_directories()
    save_dir = config.save_dir + "/"
    checkpoint_path = config.checkpoint_dir
    
    # Prepare dataset
    csv_path = prepare_dataset(config, device, skip_extraction=args.skip_clip_extraction)
    
    # Create dataloaders
    dataloader, INPUT_SIZE, BATCH_SIZE = create_dataloaders(config, csv_path, device)
    
    # Initialize model
    ddpm, model, model_compiled, T = initialize_model(config, device)
    
    # Setup training
    optimizer, scheduler, ema, epochs, scaler = setup_training(config, model, device)
    
    # Automatically load checkpoint if one exists
    start_epoch, loss_history = load_checkpoint(config, model, optimizer, scheduler, ema)
    
    if start_epoch >= epochs:
        print(f"\nAlready trained to epoch {epochs}. Increase epochs in config to continue.")
        return
    
    # Load CLIP for generation
    print("\nLoading CLIP model for generation...")
    clip_model, clip_preprocess, _ = load_clip_model(device)
    
    # Training loop
    model.train()
    save_freq = config['training']['save_frequency']
    c_drop_prob = config['training']['c_drop_prob']
    epoch_times = []
    
    print(f"\n{'='*80}")
    print(f"Starting training from epoch {start_epoch} to {epochs-1}")
    print(f"{'='*80}\n")
    
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        
        # Track average loss per epoch
        epoch_loss = 0.0
        
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad(set_to_none=True)
            t = torch.randint(0, T, (BATCH_SIZE,), device=device).float()
            x, c = batch
            x, c = x.to(device, non_blocking=True), c.to(device, non_blocking=True)
            c_mask = ddpm.get_context_mask(c, c_drop_prob)
            
            # Mixed precision training
            with autocast():
                loss = ddpm.get_loss(model_compiled, x, t, c, c_mask)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Update EMA after each step
            if ema is not None:
                ema.update()
            
            epoch_loss += loss.detach()
        
        # Step the learning rate scheduler
        scheduler.step()
        
        # Calculate epoch time and statistics
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        avg_epoch_time = np.mean(epoch_times)
        
        avg_loss = (epoch_loss / len(dataloader)).item()
        loss_history.append(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Estimate remaining time
        remaining_epochs = epochs - epoch - 1
        eta_seconds = remaining_epochs * avg_epoch_time
        eta_mins = eta_seconds / 60
        
        print(f"Epoch {epoch:3d}/{epochs-1} | "
              f"Loss: {avg_loss:.6f} | "
              f"LR: {current_lr:.2e} | "
              f"Time: {epoch_time:.1f}s | "
              f"ETA: {eta_mins:.1f}m")
        
        # Save checkpoint and generate samples
        if epoch % save_freq == 0 or epoch == int(epochs - 1):
            print(f"  → Saving checkpoint and generating samples...")
            
            # Save checkpoint
            checkpoint_data = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'loss_history': loss_history,
                'config': config.to_dict()
            }
            if ema is not None:
                checkpoint_data['ema'] = ema.shadow
            
            torch.save(checkpoint_data, f"{checkpoint_path}/{epoch}.pth")
            
            # Plot loss curve
            if len(loss_history) > 0:
                plot_loss_curve(loss_history, save_dir)
            
            # Delete old checkpoints if enabled
            delete_old = config['training'].get('delete_old_checkpoints', False)
            if delete_old:
                delete_old_checkpoints(checkpoint_path, keep_latest=True, current_epoch=epoch)
            
            # Use EMA weights for generation if enabled
            if ema is not None:
                ema.apply_shadow()
            model.eval()
            
            # Generate samples
            sample_path = save_dir + f"image_ep{epoch:03d}.png"
            x_gen, x_gen_store = generate_samples(
                model=model,
                ddpm=ddpm,
                clip_model=clip_model,
                config=config,
                input_size=INPUT_SIZE,
                timesteps=T,
                device=device,
                save_path=sample_path
            )
            
            # Restore training weights if using EMA
            if ema is not None:
                ema.restore()
            model.train()
    
    print(f"\n{'='*80}")
    print(f"Training completed!")
    print(f"  Final average loss: {loss_history[-1]:.6f}")
    print(f"  Best loss: {min(loss_history):.6f} at epoch {np.argmin(loss_history)}")
    print(f"  Total training time: {sum(epoch_times)/60:.2f} minutes")
    print(f"  Average time per epoch: {avg_epoch_time:.2f}s")
    
    # Plot loss curve
    print(f"\nGenerating loss curve...")
    plot_loss_curve(loss_history, save_dir)
    
    # Generate final samples
    print(f"\nGenerating final samples with EMA weights...")
    if ema is not None:
        ema.apply_shadow()
    model.eval()
    
    final_path = save_dir + "final_generation_ema.png"
    x_gen, x_gen_store = generate_samples(
        model=model,
        ddpm=ddpm,
        clip_model=clip_model,
        config=config,
        input_size=INPUT_SIZE,
        timesteps=T,
        device=device,
        save_path=final_path
    )
    
    # Generate animation if requested
    if not args.no_animation:
        print(f"\nGenerating animation...")
        text_list = config['generation']['text_prompts']
        w_list = config['generation']['guidance_weights']
        grids = [generation_image(x.cpu(), text_list, w=w_list) for x in x_gen_store]
        save_animation(grids, save_dir + "generation_ema.gif")
        print(f"  Animation saved to: {save_dir}generation_ema.gif")
    
    # Clean up old checkpoints after successful training
    delete_old = config['training'].get('delete_old_checkpoints', False)
    if delete_old:
        print(f"\nCleaning up old checkpoints...")
        delete_old_checkpoints(checkpoint_path, keep_latest=True)
        print(f"  Kept only the final checkpoint")
    
    print(f"\n{'='*80}")
    print(f"All files saved to:")
    print(f"  Checkpoints: {checkpoint_path}")
    print(f"  Images: {save_dir}")
    print(f"{'='*80}\n")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    print(f"\n{'='*80}")
    print(f"CLIP-Conditioned DDPM Training")
    print(f"{'='*80}")
    print(f"Model: {config.model_name}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Architecture: {config['model']['down_channels']}")
    print(f"{'='*80}\n")
    
    # Start training
    train(config, args)


if __name__ == "__main__":
    main()
