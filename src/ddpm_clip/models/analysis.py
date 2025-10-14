"""
Analysis and visualization tools for DDPM models.

This module contains functions for analyzing the diffusion process,
computing metrics, and creating visualizations.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import clip
from scipy.stats import entropy

from ..utils.visualization import to_image
from ..utils.vocabularies import (_get_tiny_imagenet_labels,
                                  _get_cifar100_labels, _get_imagenet1k_labels,
                                  _get_openai_imagenet_labels)


@torch.no_grad()
def visualize_diffusion_process(ddpm,
                                model,
                                clip_model,
                                clip_preprocess,
                                embed_size,
                                device,
                                img_ch,
                                img_size,
                                timesteps_to_show=None,
                                prompt='A cute cat',
                                w=0.5,
                                top_k=5,
                                vocabulary='imagenet1k'):
    """
    Visualize the diffusion process and compute CLIP similarities.

    Parameters
    ----------
    ddpm : DDPM
        DDPM instance containing diffusion parameters.
    model : nn.Module
        The trained denoising model.
    clip_model : nn.Module
        CLIP model for computing embeddings.
    clip_preprocess : transforms.Compose
        CLIP preprocessing transformations.
    embed_size : int
        Size of CLIP embeddings.
    device : str or torch.device
        Device to run computations on.
    img_ch : int
        Number of image channels.
    img_size : int
        Size of square images (height and width).
    timesteps_to_show : list of int or None, optional
        Specific timesteps to visualize. If None, shows evenly spaced steps, by default None.
    prompt : str, optional
        Text prompt for generation, by default "A cute cat".
    w : float, optional
        Guidance weight for classifier-free guidance, by default 0.5.
    top_k : int, optional
        Number of top similar images to display, by default 5.
    vocabulary : str, optional
        Which vocabulary to use: 'cifar100', 'tinyimagenet', 'imagenet1k', or 'openai_imagenet'
        by default 'imagenet1k'.

    Returns
    -------
    tuple
        (stored_images, stored_noises) - Lists of images and noises at analyzed timesteps.
    """
    # Select vocabulary based on parameter
    if vocabulary == 'cifar100':
        labels = _get_cifar100_labels()
    elif vocabulary == 'tinyimagenet':
        labels = _get_tiny_imagenet_labels()
    elif vocabulary == 'imagenet1k':
        labels = _get_imagenet1k_labels()
    elif vocabulary == 'openai_imagenet':
        labels = _get_openai_imagenet_labels()
    else:
        raise ValueError(f'Unknown vocabulary: {vocabulary}')

    if timesteps_to_show is None:
        # Show 8 evenly spaced timesteps from 0 to T-1 (clean to noisy)
        timesteps_to_show = torch.linspace(0, ddpm.T - 1, 8).long().tolist()
        print('Showing timesteps:', timesteps_to_show)

    n_steps = len(timesteps_to_show)

    # Prepare text embeddings for comparison
    text_tokens = clip.tokenize(labels).to(device)
    text_embeddings = clip_model.encode_text(text_tokens).float()
    text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)

    # Create figure with subplots
    fig, axes = plt.subplots(n_steps, 8, figsize=(22, n_steps * 2))
    if n_steps == 1:
        axes = axes.reshape(1, -1)

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.025, hspace=0.05)

    # Column headers
    col_titles = [
        'Diffusion Step', 'Top CLIP Match (Image)', 'Conditioned Noise',
        'Top CLIP Match (Cond. Noise)', 'Unconditioned Noise',
        'Top CLIP Match (Uncond. Noise)', 'Noise Difference',
        'Top CLIP Match (Diff.)'
    ]
    for i, title in enumerate(col_titles):
        axes[0, i].set_title(title, fontsize=12, fontweight='bold')

    # Font size for CLIP interpretation text (easy to tune)
    clip_text_fontsize = 10

    # Embed conditioning text
    text_tokens = clip.tokenize(prompt).to(device)
    text_embedding = clip_model.encode_text(text_tokens).float()

    # Generate images using the shared method
    n_samples = 1
    _, stored_data = ddpm.generate_images_with_guidance(
        model=model,
        img_ch=img_ch,
        img_size=img_size,
        text_embedding=text_embedding,
        n_samples=n_samples,
        w=w,
        timesteps_to_store=timesteps_to_show,
        return_intermediate=True)

    stored_images = [img[0] for img in stored_data['images']]
    stored_noises = [noise[0] for noise in stored_data['noises_conditioned']]
    stored_uncond_noises = [
        noise[0] for noise in stored_data['noises_unconditioned']
    ]

    for idx, t_val in enumerate(timesteps_to_show):

        # Column 1: Show the noisy image
        img_pil = to_image(stored_images[idx].detach().cpu())
        axes[idx, 0].imshow(img_pil)
        axes[idx, 0].set_ylabel(f't={t_val}', fontsize=12)
        axes[idx, 0].axis('off')

        # Column 2: CLIP interpretation of the noisy image
        try:
            image_pil = to_image(stored_images[idx].detach().cpu())
            image_clip_input = torch.tensor(
                np.stack([clip_preprocess(image_pil)])).to(device)
            image_embedding = clip_model.encode_image(image_clip_input).float()
            image_embedding /= image_embedding.norm(dim=-1, keepdim=True)

            # Find top_k matches for noisy image
            image_similarities = (text_embeddings * image_embedding).sum(-1)
            top_image_values, top_image_indices = image_similarities.topk(
                top_k)

            # Create text showing top_k matches
            image_matches = []
            for i in range(top_k):
                label = labels[top_image_indices[i].item()]
                score = top_image_values[i].item()
                image_matches.append(f'{label}: {score:.2f}')
            image_text = '\n'.join(image_matches)

            axes[idx, 1].text(0.5,
                              0.5,
                              image_text,
                              ha='center',
                              va='center',
                              transform=axes[idx, 1].transAxes,
                              fontsize=clip_text_fontsize,
                              bbox=dict(boxstyle='round,pad=0.3',
                                        facecolor='lightgreen',
                                        alpha=0.8))
        except Exception:
            axes[idx, 1].text(0.5,
                              0.5,
                              'Image CLIP\nanalysis failed',
                              ha='center',
                              va='center',
                              transform=axes[idx, 1].transAxes,
                              fontsize=clip_text_fontsize)
        axes[idx, 1].axis('off')

        # Column 3: Show the conditioned noise
        noise_vis = stored_noises[idx]
        if noise_vis.shape[0] == 3:  # RGB
            axes[idx, 2].imshow(noise_vis.permute(1, 2, 0).cpu().numpy())
        else:  # Grayscale
            axes[idx, 2].imshow(noise_vis[0].cpu().numpy(), cmap='gray')
        axes[idx, 2].axis('off')

        # Column 4: CLIP interpretation of the conditioned noise
        try:
            noise_pil = to_image(noise_vis)
            noise_clip_input = torch.tensor(
                np.stack([clip_preprocess(noise_pil)])).to(device)
            noise_embedding = clip_model.encode_image(noise_clip_input).float()
            noise_embedding /= noise_embedding.norm(dim=-1, keepdim=True)

            # Find top_k matches for noise
            noise_similarities = (text_embeddings * noise_embedding).sum(-1)
            top_noise_values, top_noise_indices = noise_similarities.topk(
                top_k)

            # Create text showing top_k matches
            noise_matches = []
            for i in range(top_k):
                label = labels[top_noise_indices[i].item()]
                score = top_noise_values[i].item()
                noise_matches.append(f'{label}: {score:.2f}')
            noise_text = '\n'.join(noise_matches)

            axes[idx, 3].text(0.5,
                              0.5,
                              noise_text,
                              ha='center',
                              va='center',
                              transform=axes[idx, 3].transAxes,
                              fontsize=clip_text_fontsize,
                              bbox=dict(boxstyle='round,pad=0.3',
                                        facecolor='lightblue',
                                        alpha=0.8))
        except Exception:
            axes[idx, 3].text(0.5,
                              0.5,
                              'Noise CLIP\nanalysis failed',
                              ha='center',
                              va='center',
                              transform=axes[idx, 3].transAxes,
                              fontsize=clip_text_fontsize)
        axes[idx, 3].axis('off')

        # Column 5: Show the unconditioned noise
        uncond_noise_vis = stored_uncond_noises[idx]
        if uncond_noise_vis.shape[0] == 3:  # RGB
            axes[idx,
                 4].imshow(uncond_noise_vis.permute(1, 2, 0).cpu().numpy())
        else:  # Grayscale
            axes[idx, 4].imshow(uncond_noise_vis[0].cpu().numpy(), cmap='gray')
        axes[idx, 4].axis('off')

        # Column 6: CLIP interpretation of the unconditioned noise
        try:
            uncond_noise_pil = to_image(uncond_noise_vis)
            uncond_noise_clip_input = torch.tensor(
                np.stack([clip_preprocess(uncond_noise_pil)])).to(device)
            uncond_noise_embedding = clip_model.encode_image(
                uncond_noise_clip_input).float()
            uncond_noise_embedding /= uncond_noise_embedding.norm(dim=-1,
                                                                  keepdim=True)

            # Find top_k matches for unconditioned noise
            uncond_noise_similarities = (text_embeddings *
                                         uncond_noise_embedding).sum(-1)
            top_uncond_noise_values, top_uncond_noise_indices = uncond_noise_similarities.topk(
                top_k)

            # Create text showing top_k matches
            uncond_noise_matches = []
            for i in range(top_k):
                label = labels[top_uncond_noise_indices[i].item()]
                score = top_uncond_noise_values[i].item()
                uncond_noise_matches.append(f'{label}: {score:.2f}')
            uncond_noise_text = '\n'.join(uncond_noise_matches)

            axes[idx, 5].text(0.5,
                              0.5,
                              uncond_noise_text,
                              ha='center',
                              va='center',
                              transform=axes[idx, 5].transAxes,
                              fontsize=clip_text_fontsize,
                              bbox=dict(boxstyle='round,pad=0.3',
                                        facecolor='lightyellow',
                                        alpha=0.8))
        except Exception:
            axes[idx, 5].text(0.5,
                              0.5,
                              'Uncond. Noise CLIP\nanalysis failed',
                              ha='center',
                              va='center',
                              transform=axes[idx, 5].transAxes,
                              fontsize=clip_text_fontsize)
        axes[idx, 5].axis('off')

        # Column 7: Show the difference between conditioned and unconditioned noise
        noise_diff = stored_noises[idx] - stored_uncond_noises[idx]
        if noise_diff.shape[0] == 3:  # RGB
            # Enhance visibility by normalizing to full range
            diff_vis = (noise_diff - noise_diff.min()) / (
                noise_diff.max() - noise_diff.min() + 1e-8)
            axes[idx, 6].imshow(diff_vis.permute(1, 2, 0).cpu().numpy())
        else:  # Grayscale
            axes[idx, 6].imshow(noise_diff[0].cpu().numpy(),
                                cmap='seismic',
                                vmin=noise_diff.min(),
                                vmax=noise_diff.max())
        axes[idx, 6].axis('off')

        # Column 8: CLIP interpretation of the noise difference
        try:
            diff_pil = to_image(noise_diff)
            diff_clip_input = torch.tensor(
                np.stack([clip_preprocess(diff_pil)])).to(device)
            diff_embedding = clip_model.encode_image(diff_clip_input).float()
            diff_embedding /= diff_embedding.norm(dim=-1, keepdim=True)

            # Find top_k matches for noise difference
            diff_similarities = (text_embeddings * diff_embedding).sum(-1)
            top_diff_values, top_diff_indices = diff_similarities.topk(top_k)

            # Create text showing top_k matches
            diff_matches = []
            for i in range(top_k):
                label = labels[top_diff_indices[i].item()]
                score = top_diff_values[i].item()
                diff_matches.append(f'{label}: {score:.2f}')
            diff_text = '\n'.join(diff_matches)

            axes[idx, 7].text(0.5,
                              0.5,
                              diff_text,
                              ha='center',
                              va='center',
                              transform=axes[idx, 7].transAxes,
                              fontsize=clip_text_fontsize,
                              bbox=dict(boxstyle='round,pad=0.3',
                                        facecolor='lightcoral',
                                        alpha=0.8))
        except Exception:
            axes[idx, 7].text(0.5,
                              0.5,
                              'Diff CLIP\nanalysis failed',
                              ha='center',
                              va='center',
                              transform=axes[idx, 7].transAxes,
                              fontsize=clip_text_fontsize)
        axes[idx, 7].axis('off')

    plt.suptitle(
        f"DDPM process + CLIP conditioning:'{prompt}' with weight {w}",
        fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.98], pad=0.5)
    plt.show()

    return stored_images, stored_noises


@torch.no_grad()
def analyze_diffusion_metrics(ddpm,
                              model,
                              clip_model,
                              clip_preprocess,
                              device,
                              img_ch,
                              img_size,
                              timesteps_to_analyze=None,
                              prompt='A cute cat',
                              w=0.5,
                              n_samples=1):
    """
    Analyze the diffusion process by computing various quality metrics at each step.

    This function tracks the evolution of:
    1. Image quality (PSNR, sharpness)
    2. Feature strength (edges, shapes, texture, color variance)
    3. Text-image alignment (CLIP similarity with prompt)
    4. Noise strength (variance, entropy)

    Parameters
    ----------
    ddpm : DDPM
        DDPM instance containing diffusion parameters.
    model : nn.Module
        The trained denoising model.
    clip_model : nn.Module
        CLIP model for computing text-image alignment.
    clip_preprocess : transforms.Compose
        CLIP preprocessing transformations.
    device : str or torch.device
        Device to run computations on.
    img_ch : int
        Number of image channels.
    img_size : int
        Size of square images (height and width).
    timesteps_to_analyze : list of int or None, optional
        Specific timesteps to analyze. If None, analyzes all timesteps.
    prompt : str, optional
        Text prompt for CLIP alignment measurement, by default "A cute cat".
    w : float, optional
        Guidance weight for classifier-free guidance, by default 0.5.
    n_samples : int, optional
        Number of samples to generate in parallel. Metrics will be averaged
        across all samples, except for images where only the first is stored, by default 1.

    Returns
    -------
    dict
        Dictionary containing metric histories (averaged over n_samples):
        - 'timesteps': List of analyzed timesteps
        - 'generation_steps': List of generation steps (t' = T - t)
        - 'image_quality': PSNR values (higher = better quality)
        - 'sharpness': Laplacian variance (higher = sharper)
        - 'edge_strength': Sobel gradient magnitude
        - 'color_variance': Color channel variance
        - 'texture_strength': High-frequency content measure
        - 'clip_similarity': Text-image alignment score
        - 'noise_variance': Noise variance estimate
        - 'noise_entropy': Noise entropy measure
        - 'images': List of first sample images at each analyzed timestep
    """
    # Embed conditioning text for CLIP alignment
    text_tokens = clip.tokenize([prompt]).to(device)
    text_embedding = clip_model.encode_text(text_tokens).float()
    text_embedding /= text_embedding.norm(dim=-1, keepdim=True)

    # Initialize metric storage
    metrics = {
        'timesteps': [],  # Original timesteps (t)
        'generation_steps': [],  # Generation steps (t' = T - t)
        'image_quality': [],  # PSNR
        'sharpness': [],  # Laplacian variance
        'edge_strength': [],  # Sobel gradient magnitude
        'color_variance': [],  # Color channel variance
        'texture_strength': [],  # High-frequency content
        'clip_similarity': [],  # Text-image alignment
        'noise_variance': [],  # Noise variance
        'noise_entropy': [],  # Noise entropy
        'images': []  # Store intermediate images
    }

    # Determine which timesteps to analyze
    if timesteps_to_analyze is None:
        timesteps_to_analyze = list(range(ddpm.T))

    # Prepare context for generation
    c = text_embedding.repeat(n_samples, 1)
    c = c.repeat(2, 1)  # Double it for conditioned + unconditioned
    c_mask = torch.ones_like(c).to(device)
    c_mask[n_samples:] = 0.0

    # Reverse diffusion process to generate the image (n_samples in parallel)
    x_t = torch.randn((n_samples, img_ch, img_size, img_size),
                      device=ddpm.device)

    for i in range(0, ddpm.T)[::-1]:  # T-1, T-2, ..., 1, 0
        t = torch.tensor([i], device=ddpm.device)
        t = t.repeat(n_samples, 1, 1, 1)

        # Double the batch for conditional and unconditional
        x_t_double = x_t.repeat(2, 1, 1, 1)
        t_double = t.repeat(2, 1, 1, 1)

        # Predict noise
        e_t = model(x_t_double, t_double, c, c_mask)
        e_t_keep_c = e_t[:n_samples]
        e_t_drop_c = e_t[n_samples:]

        # Apply classifier-free guidance
        e_t_guided = (1 + w) * e_t_keep_c - w * e_t_drop_c

        # Perform reverse diffusion step (deduplicate batch first)
        x_t = ddpm.reverse_q(x_t, t, e_t_guided)

        # Analyze metrics at specified timesteps
        if i in timesteps_to_analyze:
            metrics['timesteps'].append(i)
            metrics['generation_steps'].append(ddpm.T - i)  # t' = T - t

            # Store only the first image for visualization
            metrics['images'].append(x_t[0].detach().clone())

            # Compute metrics for all samples and average them
            # Temporary storage for this timestep's metrics
            psnr_list = []
            sharpness_list = []
            edge_strength_list = []
            color_variance_list = []
            texture_strength_list = []
            clip_similarity_list = []
            noise_variance_list = []
            noise_entropy_list = []

            for sample_idx in range(n_samples):
                img = x_t[sample_idx].detach()

                # 1. IMAGE QUALITY METRICS
                # PSNR (Peak Signal-to-Noise Ratio) - comparing to a reference range
                # Higher values indicate better quality
                img_normalized = (img + 1) / 2  # Normalize to [0, 1]
                mse = torch.mean((img_normalized - 0.5)**2)
                psnr = 20 * torch.log10(
                    torch.tensor(1.0) / (torch.sqrt(mse) + 1e-8))
                psnr_list.append(psnr.item())

                # Sharpness (Laplacian variance)
                # Higher values indicate sharper images
                laplacian_kernel = torch.tensor(
                    [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                    dtype=torch.float32,
                    device=device).unsqueeze(0).unsqueeze(0)
                laplacian_kernel = laplacian_kernel.repeat(img_ch, 1, 1, 1)

                laplacian = F.conv2d(img.unsqueeze(0),
                                     laplacian_kernel,
                                     padding=1,
                                     groups=img_ch)
                sharpness = torch.var(laplacian).item()
                sharpness_list.append(sharpness)

                # 2. FEATURE STRENGTH METRICS
                # Edge strength (Sobel gradient magnitude)
                sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                       dtype=torch.float32,
                                       device=device).unsqueeze(0).unsqueeze(0)
                sobel_y = sobel_x.transpose(-2, -1)

                sobel_x = sobel_x.repeat(img_ch, 1, 1, 1)
                sobel_y = sobel_y.repeat(img_ch, 1, 1, 1)

                grad_x = F.conv2d(img.unsqueeze(0),
                                  sobel_x,
                                  padding=1,
                                  groups=img_ch)
                grad_y = F.conv2d(img.unsqueeze(0),
                                  sobel_y,
                                  padding=1,
                                  groups=img_ch)
                edge_strength = torch.sqrt(grad_x**2 + grad_y**2).mean().item()
                edge_strength_list.append(edge_strength)

                # Color variance (how varied the colors are)
                color_var = torch.var(img, dim=[1, 2]).mean().item()
                color_variance_list.append(color_var)

                # Texture strength (high-frequency content via FFT)
                if img_ch >= 1:
                    # Use first channel for texture analysis
                    fft = torch.fft.fft2(img[0])
                    fft_shift = torch.fft.fftshift(fft)
                    magnitude = torch.abs(fft_shift)

                    # Measure high-frequency content (outer regions of FFT)
                    h, w = magnitude.shape
                    center_h, center_w = h // 2, w // 2
                    mask_size = min(h, w) // 4

                    # Create high-frequency mask (everything except center)
                    mask = torch.ones_like(magnitude)
                    mask[center_h - mask_size:center_h + mask_size,
                         center_w - mask_size:center_w + mask_size] = 0

                    high_freq_energy = (magnitude *
                                        mask).sum() / (magnitude.sum() + 1e-8)
                    texture_strength_list.append(high_freq_energy.item())
                else:
                    texture_strength_list.append(0.0)

                # 3. TEXT ALIGNMENT METRIC
                # CLIP similarity with prompt
                try:
                    img_pil = to_image(img.cpu())
                    img_clip_input = torch.tensor(
                        np.stack([clip_preprocess(img_pil)])).to(device)
                    img_embedding = clip_model.encode_image(
                        img_clip_input).float()
                    img_embedding /= img_embedding.norm(dim=-1, keepdim=True)

                    clip_sim = (text_embedding * img_embedding).sum().item()
                    clip_similarity_list.append(clip_sim)
                except Exception:
                    clip_similarity_list.append(0.0)

                # 4. NOISE STRENGTH METRICS
                # Noise variance
                noise_var = torch.var(e_t_guided[sample_idx]).item()
                noise_variance_list.append(noise_var)

                # Noise entropy (information content)
                noise_flat = e_t_guided[sample_idx].flatten().cpu().numpy()
                # Discretize for entropy calculation
                hist, _ = np.histogram(noise_flat, bins=50, density=True)
                hist = hist + 1e-10  # Avoid log(0)
                noise_ent = entropy(hist)
                noise_entropy_list.append(noise_ent)

            # Average all metrics across samples
            metrics['image_quality'].append(np.mean(psnr_list))
            metrics['sharpness'].append(np.mean(sharpness_list))
            metrics['edge_strength'].append(np.mean(edge_strength_list))
            metrics['color_variance'].append(np.mean(color_variance_list))
            metrics['texture_strength'].append(np.mean(texture_strength_list))
            metrics['clip_similarity'].append(np.mean(clip_similarity_list))
            metrics['noise_variance'].append(np.mean(noise_variance_list))
            metrics['noise_entropy'].append(np.mean(noise_entropy_list))

    return metrics


@torch.no_grad()
def plot_diffusion_metrics(metrics,
                           save_path=None,
                           diffusion_images=None,
                           prompt=None,
                           w=None):
    """
    Visualize the metrics computed during the diffusion process.

    Parameters
    ----------
    metrics : dict
        Dictionary of metrics from analyze_diffusion_metrics().
    save_path : str or None, optional
        Path to save the figure. If None, displays the plot.
    diffusion_images : list of tuples or None, optional
        List of (image_tensor, generation_step) tuples from the diffusion process.
        Each tensor should be of shape (C, H, W).
    prompt : str or None, optional
        Text prompt used for generation (displayed in figure title).
    w : float or None, optional
        Guidance weight used for generation (displayed in figure title).

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib figure object.
    """
    generation_steps = metrics['generation_steps']

    # Create figure with 2 rows for images + 2 rows for metrics
    fig = plt.figure(figsize=(18, 16))

    # Add overall title with prompt and guidance weight
    title_parts = ['Diffusion Process Analysis']
    if prompt is not None:
        title_parts.append(f'Prompt: "{prompt}"')
    if w is not None:
        title_parts.append(f'Guidance weight (w): {w}')
    fig.suptitle(' | '.join(title_parts),
                 fontsize=16,
                 fontweight='bold',
                 y=0.995)

    # Create grid: 2 rows for images (6 images total), 2 rows for metrics
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

    # Display images in top 2 rows (up to 6 images)
    image_axes = []
    for row in range(2):
        for col in range(3):
            image_axes.append(fig.add_subplot(gs[row, col]))

    for idx, ax in enumerate(image_axes):
        ax.axis('off')
        if idx < len(diffusion_images):
            img_data = diffusion_images[idx]
            # Handle both tuple (img, generation_step) and tensor formats
            if isinstance(img_data, tuple):
                img, gen_step = img_data
            else:
                img = img_data
                gen_step = generation_steps[idx] if idx < len(
                    generation_steps) else idx

            # Convert tensor to displayable format
            if isinstance(img, torch.Tensor):
                img_display = to_image(img.cpu())
                ax.imshow(img_display)
                ax.set_title(f"t' = {gen_step}",
                             fontsize=12,
                             fontweight='bold')

    # Plot metrics in bottom 2 rows
    metric_axes = []
    for row in range(2, 4):
        for col in range(3):
            metric_axes.append(fig.add_subplot(gs[row, col]))

    plot_configs = [
        ('image_quality', 'Image Quality (PSNR)', metric_axes[0], 'blue'),
        ('sharpness', 'Sharpness (Laplacian Variance)', metric_axes[1],
         'green'),
        ('edge_strength', 'Edge Strength (Sobel)', metric_axes[2], 'red'),
        ('color_variance', 'Color Variance', metric_axes[3], 'purple'),
        ('texture_strength', 'Texture Strength (High-Freq)', metric_axes[4],
         'orange'),
        ('clip_similarity', 'Text-Image Alignment (CLIP)', metric_axes[5],
         'cyan'),
    ]

    for metric_name, title, ax, color in plot_configs:
        ax.plot(generation_steps,
                metrics[metric_name],
                color=color,
                linewidth=2,
                marker='o',
                markersize=3)
        ax.set_xlabel("Generation Step t' (0 = noise → T = clean)",
                      fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Metrics plot saved to: {save_path}')
    else:
        plt.show()

    return fig


@torch.no_grad()
def analyze_prompt_alignment(ddpm,
                             model,
                             clip_model,
                             clip_preprocess,
                             device,
                             img_ch,
                             img_size,
                             timesteps_to_analyze=None,
                             generation_prompt='A fast car',
                             alignment_prompts=None,
                             w=0.5,
                             n_samples=1):
    """
    Analyze how the generated image aligns with different prompts throughout the diffusion process.

    This function generates an image using one prompt, then tracks how well the intermediate
    images align with various other prompts via CLIP similarity.

    Parameters
    ----------
    ddpm : DDPM
        DDPM instance containing diffusion parameters.
    model : nn.Module
        The trained denoising model.
    clip_model : nn.Module
        CLIP model for computing text-image alignment.
    clip_preprocess : transforms.Compose
        CLIP preprocessing transformations.
    device : str or torch.device
        Device to run computations on.
    img_ch : int
        Number of image channels.
    img_size : int
        Size of square images (height and width).
    timesteps_to_analyze : list of int or None, optional
        Specific timesteps to analyze. If None, analyzes every 10th timestep.
    generation_prompt : str, optional
        Text prompt used to generate the image, by default "A fast car".
    alignment_prompts : list of str or None, optional
        List of prompts to check alignment against. If None, uses the generation prompt.
    w : float, optional
        Guidance weight for classifier-free guidance, by default 0.5.
    n_samples : int, optional
        Number of samples to generate in parallel (similarities averaged), by default 1.

    Returns
    -------
    dict
        Dictionary containing:
        - 'generation_steps': List of generation steps (t' = T - t)
        - 'alignment_prompts': List of prompts used for alignment
        - 'similarities': 2D array of shape (n_timesteps, n_prompts) with CLIP similarities
        - 'images': List of first sample images at each analyzed timestep
    """
    if alignment_prompts is None:
        alignment_prompts = [generation_prompt]

    if timesteps_to_analyze is None:
        timesteps_to_analyze = list(range(0, ddpm.T, 10))

    # Embed the generation prompt for conditional generation
    gen_text_tokens = clip.tokenize([generation_prompt]).to(device)
    gen_text_embedding = clip_model.encode_text(gen_text_tokens).float()
    gen_text_embedding /= gen_text_embedding.norm(dim=-1, keepdim=True)

    # Prepare context for generation (n_samples conditioned + n_samples unconditioned)
    c = gen_text_embedding.repeat(n_samples, 1)
    c = c.repeat(2, 1)  # Double it for conditioned + unconditioned
    c_mask = torch.ones_like(c).to(device)
    c_mask[n_samples:] = 0.0

    # Embed all alignment prompts
    alignment_text_tokens = clip.tokenize(alignment_prompts).to(device)
    alignment_embeddings = clip_model.encode_text(
        alignment_text_tokens).float()
    alignment_embeddings /= alignment_embeddings.norm(dim=-1, keepdim=True)

    # Initialize storage
    results = {
        'generation_steps': [],
        'alignment_prompts': alignment_prompts,
        'similarities': [],  # Will be a list of lists
        'images': []
    }

    # Reverse diffusion process to generate the image (n_samples in parallel)
    x_t = torch.randn((n_samples, img_ch, img_size, img_size),
                      device=ddpm.device)

    for i in range(0, ddpm.T)[::-1]:  # T-1, T-2, ..., 1, 0
        t = torch.tensor([i], device=ddpm.device)
        t = t.repeat(n_samples, 1, 1, 1)

        # Double the batch for conditional and unconditional
        x_t_double = x_t.repeat(2, 1, 1, 1)
        t_double = t.repeat(2, 1, 1, 1)

        # Predict noise
        e_t = model(x_t_double, t_double, c, c_mask)
        e_t_keep_c = e_t[:n_samples]
        e_t_drop_c = e_t[n_samples:]

        # Apply classifier-free guidance
        e_t_guided = (1 + w) * e_t_keep_c - w * e_t_drop_c

        # Perform reverse diffusion step (deduplicate batch first)
        x_t = ddpm.reverse_q(x_t, t, e_t_guided)

        # Analyze alignment at specified timesteps
        if i in timesteps_to_analyze:
            results['generation_steps'].append(ddpm.T - i)  # t' = T - t

            # Store only the first image for visualization
            results['images'].append(x_t[0].detach().clone())

            # Compute CLIP similarities with all alignment prompts for all samples
            similarities_per_prompt = []

            for prompt_idx in range(len(alignment_prompts)):
                prompt_embedding = alignment_embeddings[prompt_idx:prompt_idx +
                                                        1]

                # Compute similarity for each sample
                sample_similarities = []
                for sample_idx in range(n_samples):
                    img = x_t[sample_idx].detach()

                    try:
                        img_pil = to_image(img.cpu())
                        img_clip_input = torch.tensor(
                            np.stack([clip_preprocess(img_pil)])).to(device)
                        img_embedding = clip_model.encode_image(
                            img_clip_input).float()
                        img_embedding /= img_embedding.norm(dim=-1,
                                                            keepdim=True)

                        clip_sim = (prompt_embedding *
                                    img_embedding).sum().item()
                        sample_similarities.append(clip_sim)
                    except Exception:
                        sample_similarities.append(0.0)

                # Average across samples
                similarities_per_prompt.append(np.mean(sample_similarities))

            results['similarities'].append(similarities_per_prompt)

    # Convert similarities to numpy array for easier plotting
    results['similarities'] = np.array(results['similarities'])

    return results


@torch.no_grad()
def plot_prompt_alignment(results,
                          save_path=None,
                          show_images=True,
                          num_images=6):
    """
    Visualize how image alignment with different prompts evolves during diffusion.

    Parameters
    ----------
    results : dict
        Dictionary from analyze_prompt_alignment().
    save_path : str or None, optional
        Path to save the figure. If None, displays the plot.
    show_images : bool, optional
        Whether to show intermediate images, by default True.
    num_images : int, optional
        Number of evenly-spaced images to display, by default 6.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib figure object.
    """
    generation_steps = results['generation_steps']
    alignment_prompts = results['alignment_prompts']
    similarities = results['similarities']

    if show_images and len(results['images']) > 0:
        # Create figure with images on top and plot below
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2,
                              num_images,
                              height_ratios=[1, 2],
                              hspace=0.3,
                              wspace=0.2)

        # Select evenly spaced images
        n_total_images = len(results['images'])
        image_indices = [
            int(i * (n_total_images - 1) / (num_images - 1))
            for i in range(num_images)
        ]

        # Display images in top row
        for col, idx in enumerate(image_indices):
            ax = fig.add_subplot(gs[0, col])
            img_display = to_image(results['images'][idx].cpu())
            ax.imshow(img_display)
            ax.set_title(f"t' = {generation_steps[idx]}",
                         fontsize=10,
                         fontweight='bold')
            ax.axis('off')

        # Plot similarities in bottom row (spanning all columns)
        ax_plot = fig.add_subplot(gs[1, :])
    else:
        # Just create a single plot
        fig, ax_plot = plt.subplots(figsize=(14, 6))

    # Plot each prompt's alignment over time
    colors = plt.cm.tab10(np.linspace(0, 1, len(alignment_prompts)))

    for i, prompt in enumerate(alignment_prompts):
        ax_plot.plot(generation_steps,
                     similarities[:, i],
                     label=prompt,
                     color=colors[i],
                     linewidth=2.5,
                     marker='o',
                     markersize=5)

    ax_plot.set_xlabel("Generation Step t' (0 = noise → T = clean)",
                       fontsize=12,
                       fontweight='bold')
    ax_plot.set_ylabel('CLIP Similarity', fontsize=12, fontweight='bold')
    ax_plot.set_title('Prompt Alignment Throughout Diffusion Process',
                      fontsize=14,
                      fontweight='bold')
    ax_plot.legend(loc='best', fontsize=10)
    ax_plot.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Prompt alignment plot saved to: {save_path}')
    else:
        plt.show()

    return fig
