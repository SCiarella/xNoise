"""
Training utilities for CLIP-conditioned DDPM.

This module provides helper functions for:
- Device setup and verification
- CLIP model loading
- Checkpoint management (loading, saving, deletion)
- Loss curve visualization
"""

import os
import torch
import matplotlib.pyplot as plt
import gc


def setup_device():
    """
    Setup and verify CUDA device.

    Returns:
        torch.device: CUDA device

    Raises:
        RuntimeError: If CUDA is not available
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError(
            'CUDA not available. This script requires GPU support.')
    print(f'Using device: {device}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    return device


def load_clip_model(device):
    """
    Load CLIP model.

    Args:
        device: torch device to load model on

    Returns:
        tuple: (clip_model, clip_preprocess, CLIP_FEATURES)
    """
    import clip

    gc.collect()
    torch.cuda.empty_cache()

    print('Loading CLIP model...')
    clip_model, clip_preprocess = clip.load('ViT-B/32', device=device)
    clip_model.eval()
    CLIP_FEATURES = 512
    print(f'CLIP model loaded with {CLIP_FEATURES} features')

    return clip_model, clip_preprocess, CLIP_FEATURES


def load_checkpoint(config, model, optimizer=None, scheduler=None, ema=None):
    """
    Load checkpoint if one exists (automatic resume behavior).

    Args:
        config: Configuration object with checkpoint_dir attribute
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        ema: Optional EMA object to load shadow weights into

    Returns:
        tuple: (start_epoch, loss_history)
    """
    checkpoint_path = config.checkpoint_dir
    start_epoch = 0
    loss_history = []

    if os.path.exists(checkpoint_path):
        checkpoint_files = [
            f for f in os.listdir(checkpoint_path) if f.endswith('.pth')
        ]
        if checkpoint_files:
            latest_epoch = max(
                [int(f.split('.')[0]) for f in checkpoint_files])
            latest_checkpoint = os.path.join(checkpoint_path,
                                             f'{latest_epoch}.pth')

            print(f'\nFound checkpoint: {latest_checkpoint}')
            print('Automatically resuming training...')
            checkpoint = torch.load(latest_checkpoint,
                                    map_location='cuda',
                                    weights_only=False)

            model.load_state_dict(checkpoint['model'])

            if optimizer is not None and 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])

            if scheduler is not None and 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])

            if ema is not None and 'ema' in checkpoint:
                ema.shadow = checkpoint['ema']

            start_epoch = checkpoint['epoch'] + 1

            if 'loss_history' in checkpoint:
                loss_history = checkpoint['loss_history']
                print(
                    f'Resumed from epoch {start_epoch} with {len(loss_history)} loss values'
                )
            else:
                print(f'Resumed from epoch {start_epoch}')
    else:
        print('\nNo checkpoint found - starting fresh training')

    return start_epoch, loss_history


def delete_old_checkpoints(checkpoint_path,
                           keep_latest=True,
                           current_epoch=None):
    """
    Delete old checkpoint files to save disk space.

    Args:
        checkpoint_path: Directory containing checkpoint files
        keep_latest: If True, keep only the latest checkpoint (or current_epoch if provided)
        current_epoch: If provided, keep only this checkpoint when keep_latest=True
    """
    if not os.path.exists(checkpoint_path):
        return

    checkpoint_files = [
        f for f in os.listdir(checkpoint_path) if f.endswith('.pth')
    ]

    if not checkpoint_files:
        return

    if keep_latest:
        # Keep only the checkpoint from current_epoch (if specified) or the latest one
        if current_epoch is not None:
            # Delete all checkpoints except the current one
            for f in checkpoint_files:
                epoch = int(f.split('.')[0])
                if epoch != current_epoch:
                    checkpoint_file = os.path.join(checkpoint_path, f)
                    try:
                        os.remove(checkpoint_file)
                        print(f'  → Deleted old checkpoint: {f}')
                    except Exception as e:
                        print(f'  → Warning: Could not delete {f}: {e}')
        else:
            # Keep only the latest checkpoint
            latest_epoch = max(
                [int(f.split('.')[0]) for f in checkpoint_files])
            for f in checkpoint_files:
                epoch = int(f.split('.')[0])
                if epoch != latest_epoch:
                    checkpoint_file = os.path.join(checkpoint_path, f)
                    try:
                        os.remove(checkpoint_file)
                        print(f'  → Deleted old checkpoint: {f}')
                    except Exception as e:
                        print(f'  → Warning: Could not delete {f}: {e}')
    else:
        # Delete all checkpoints
        for f in checkpoint_files:
            checkpoint_file = os.path.join(checkpoint_path, f)
            try:
                os.remove(checkpoint_file)
                print(f'  → Deleted checkpoint: {f}')
            except Exception as e:
                print(f'  → Warning: Could not delete {f}: {e}')


def plot_loss_curve(loss_history, save_dir):
    """
    Plot and save loss curve.

    Args:
        loss_history: List of loss values per epoch
        save_dir: Directory to save the plot
    """
    if len(loss_history) == 0:
        print('No loss history to plot')
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
    plt.plot(range(start_idx, len(loss_history)),
             loss_history[start_idx:],
             linewidth=2,
             color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss (Last 50%)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/loss_curve.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f'  Loss curve saved to: {save_dir}/loss_curve.png')
