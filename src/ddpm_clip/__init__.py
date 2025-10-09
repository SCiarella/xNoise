"""
DDPM-CLIP: Text-to-Image Generation with Diffusion Models
"""

__version__ = '0.1.0'

from .models.ddpm import DDPM
from .models.unet import UNet
from .models.ema import EMA
from .data.clip_dataset import CLIPDataset
from .data.preprocessing import extract_clip_embeddings
from .utils.visualization import save_animation, generation_image, to_image
from .utils.training import (setup_device, load_clip_model, load_checkpoint,
                             delete_old_checkpoints, plot_loss_curve)
from .utils.generation import sample_from_text, generate_samples

__all__ = [
    'DDPM', 'UNet', 'EMA', 'CLIPDataset', 'extract_clip_embeddings',
    'save_animation', 'generation_image', 'to_image', 'setup_device',
    'load_clip_model', 'load_checkpoint', 'delete_old_checkpoints',
    'plot_loss_curve', 'sample_from_text', 'generate_samples'
]
