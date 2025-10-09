"""
Generation utilities for CLIP-conditioned DDPM.

This module provides helper functions for:
- Text-to-image generation
- Sample generation with different guidance weights
"""

import torch
from .visualization import generation_image


def sample_from_text(text_list,
                     clip_model,
                     model,
                     ddpm,
                     input_size,
                     timesteps,
                     device,
                     w_values=None):
    """
    Generate images from text prompts using CLIP-conditioned diffusion.

    Args:
        text_list: List of text prompts
        clip_model: CLIP model for text encoding
        model: Diffusion U-Net model
        ddpm: DDPM object for sampling
        input_size: Tuple of (channels, height, width)
        timesteps: Number of diffusion timesteps
        device: torch device
        w_values: Optional list of guidance weights to test

    Returns:
        tuple: (x_gen, x_gen_store) - final images and intermediate steps
    """
    import clip
    # Import here to avoid circular dependency
    from ..models import sample_w

    text_tokens = clip.tokenize(text_list).to(device)
    c = clip_model.encode_text(text_tokens).float()

    with torch.no_grad():
        x_gen, x_gen_store = sample_w(model,
                                      ddpm,
                                      input_size,
                                      timesteps,
                                      c,
                                      device,
                                      w_tests=w_values)

    return x_gen, x_gen_store


def generate_samples(model,
                     ddpm,
                     clip_model,
                     config,
                     input_size,
                     timesteps,
                     device,
                     save_path=None):
    """
    Generate sample images using configuration settings.

    Args:
        model: Diffusion U-Net model
        ddpm: DDPM object for sampling
        clip_model: CLIP model for text encoding
        config: Configuration object with generation settings
        input_size: Tuple of (channels, height, width)
        timesteps: Number of diffusion timesteps
        device: torch device
        save_path: Optional path to save generated image grid

    Returns:
        tuple: (x_gen, x_gen_store) - final images and intermediate steps
    """
    text_list = config['generation']['text_prompts']
    w_list = config['generation']['guidance_weights']

    x_gen, x_gen_store = sample_from_text(text_list=text_list,
                                          clip_model=clip_model,
                                          model=model,
                                          ddpm=ddpm,
                                          input_size=input_size,
                                          timesteps=timesteps,
                                          device=device,
                                          w_values=w_list)

    if save_path is not None:
        generation_image(x_gen, text_list, w=w_list, save_path=save_path)

    return x_gen, x_gen_store
