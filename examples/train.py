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
from ddpm_clip.utils import save_animation, generation_image, to_image, load_config


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


def setup_device():
    """Setup and verify CUDA device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != 'cuda':
        raise RuntimeError("CUDA not available. This script requires GPU support.")
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


def load_clip_model(device):
    """Load CLIP model."""
    import clip
    
    gc.collect()
    torch.cuda.empty_cache()
    
    print("Loading CLIP model...")
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    CLIP_FEATURES = 512
    print(f"CLIP model loaded with {CLIP_FEATURES} features")
    
    return clip_model, clip_preprocess, CLIP_FEATURES


def prepare_dataset(config, device, skip_extraction=False):
    """Prepare dataset and extract CLIP embeddings if needed."""
    import clip
    
    # Collect image paths
    dataset_root = config['data']['dataset_root']
    train_paths = glob.glob(f"{dataset_root}/train/*/images/*.JPEG")
    val_paths = glob.glob(f"{dataset_root}/val/images/*.JPEG")
    data_paths = train_paths
    
    ndata = config['data']['max_samples']
    print(f"Found {len(data_paths)} total images")
    
    # Shuffle and limit dataset size
    random_seed = config['data']['random_seed']
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
        device=device
    )
    
    dataloader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
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
    print(f"  Conditioning drop probability: {config['training']['c_drop_prob']}")
    print(f"  Save frequency: every {config['training']['save_frequency']} epochs")
    
    return optimizer, scheduler, ema, epochs


def load_checkpoint(config, model, optimizer, scheduler, ema):
    """Load checkpoint if one exists (automatic resume behavior)."""
    checkpoint_path = config.checkpoint_dir
    start_epoch = 0
    loss_history = []
    
    if os.path.exists(checkpoint_path):
        checkpoint_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.pth')]
        if checkpoint_files:
            latest_epoch = max([int(f.split('.')[0]) for f in checkpoint_files])
            latest_checkpoint = os.path.join(checkpoint_path, f"{latest_epoch}.pth")
            
            print(f"\nFound checkpoint: {latest_checkpoint}")
            print(f"Automatically resuming training...")
            checkpoint = torch.load(latest_checkpoint, map_location='cuda')
            
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            
            if ema is not None and 'ema' in checkpoint:
                ema.shadow = checkpoint['ema']
            
            start_epoch = checkpoint['epoch'] + 1
            
            if 'loss_history' in checkpoint:
                loss_history = checkpoint['loss_history']
                print(f"Resumed from epoch {start_epoch} with {len(loss_history)} loss values")
            else:
                print(f"Resumed from epoch {start_epoch}")
    else:
        print("\nNo checkpoint found - starting fresh training")
    
    return start_epoch, loss_history


def generate_samples(model, ddpm, clip_model, config, INPUT_SIZE, T, device, save_path):
    """Generate sample images."""
    import clip
    
    text_list = config['generation']['text_prompts']
    w_list = config['generation']['guidance_weights']
    
    text_tokens = clip.tokenize(text_list).to(device)
    c = clip_model.encode_text(text_tokens).float()
    
    with torch.no_grad():
        x_gen, x_gen_store = sample_w(model, ddpm, INPUT_SIZE, T, c, device, w_tests=w_list)
    
    generation_image(x_gen, text_list, w=w_list, save_path=save_path)
    
    return x_gen, x_gen_store


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
    optimizer, scheduler, ema, epochs = setup_training(config, model, device)
    
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
        
        gc.collect()
        torch.cuda.empty_cache()
        
        # Track average loss per epoch
        epoch_losses = []
        
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            t = torch.randint(0, T, (BATCH_SIZE,), device=device).float()
            x, c = batch
            c_mask = ddpm.get_context_mask(c, c_drop_prob)
            loss = ddpm.get_loss(model_compiled, x, t, c, c_mask)
            loss.backward()
            optimizer.step()
            
            # Update EMA after each step
            if ema is not None:
                ema.update()
            
            epoch_losses.append(loss.item())
        
        # Step the learning rate scheduler
        scheduler.step()
        
        # Calculate epoch time and statistics
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        avg_epoch_time = np.mean(epoch_times)
        
        avg_loss = np.mean(epoch_losses)
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
            
            # Use EMA weights for generation if enabled
            if ema is not None:
                ema.apply_shadow()
            model.eval()
            
            # Generate samples
            sample_path = save_dir + f"image_ep{epoch:03d}.png"
            x_gen, x_gen_store = generate_samples(
                model, ddpm, clip_model, config, INPUT_SIZE, T, device, sample_path
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
        model, ddpm, clip_model, config, INPUT_SIZE, T, device, final_path
    )
    
    # Generate animation if requested
    if not args.no_animation:
        print(f"\nGenerating animation...")
        text_list = config['generation']['text_prompts']
        w_list = config['generation']['guidance_weights']
        grids = [generation_image(x.cpu(), text_list, w=w_list) for x in x_gen_store]
        save_animation(grids, save_dir + "generation_ema.gif")
        print(f"  Animation saved to: {save_dir}generation_ema.gif")
    
    print(f"\n{'='*80}")
    print(f"All files saved to:")
    print(f"  Checkpoints: {checkpoint_path}")
    print(f"  Images: {save_dir}")
    print(f"{'='*80}\n")


def plot_loss_curve(loss_history, save_dir):
    """Plot and save loss curve."""
    if len(loss_history) == 0:
        print("No loss history to plot")
        return
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(len(loss_history)), loss_history, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss Over Time')
    plt.grid(True, alpha=0.3)
    
    # Plot last 50% of training to see recent trends
    plt.subplot(1, 2, 2)
    start_idx = len(loss_history) // 2
    plt.plot(range(start_idx, len(loss_history)), loss_history[start_idx:], 
             linewidth=2, color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss (Last 50%)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/loss_curve.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Loss curve saved to: {save_dir}/loss_curve.png")


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
