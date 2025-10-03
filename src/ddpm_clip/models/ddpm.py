import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import clip

from ..utils.visualization import generation_image, to_image


class DDPM:
    """
    Denoising Diffusion Probabilistic Model (DDPM) implementation.

    Parameters
    ----------
    B : torch.Tensor
        Beta schedule tensor of shape (T,) defining noise levels.
    device : str or torch.device
        Device to run computations on ('cuda' or 'cpu').
    """

    def __init__(self, B, device):
        self.B = B  # Beta schedule (noise variance)
        self.T = len(B)  # Number of diffusion timesteps
        self.device = device

        # Forward diffusion variables
        # These precomputed values optimize the forward diffusion process
        self.a = 1.0 - self.B  # Alpha values (1 - beta)
        self.a_bar = torch.cumprod(self.a,
                                   dim=0)  # Cumulative product of alphas
        self.sqrt_a_bar = torch.sqrt(
            self.a_bar)  # Mean Coefficient for forward process
        self.sqrt_one_minus_a_bar = torch.sqrt(
            1 - self.a_bar)  # St. Dev. Coefficient for noise

        # Reverse diffusion variables
        # These precomputed values optimize the reverse diffusion process
        self.sqrt_a_inv = torch.sqrt(
            1 / self.a)  # Inverse sqrt alpha for denoising
        self.pred_noise_coeff = (1 - self.a) / torch.sqrt(
            1 - self.a_bar)  # Coefficient for predicted noise

    def q(self, x_0, t):
        """
        Forward diffusion process q(x_t | x_0).

        Applies noise to the original image according to the diffusion schedule.
        This implements the closed-form solution for adding noise at any timestep t:
        x_t = sqrt(α_bar_t) * x_0 + sqrt(1 - α_bar_t) * ε

        Args:
            x_0 (torch.Tensor): Original clean image [batch_size, channels, height, width]
            t (torch.Tensor): Timestep(s) for noise application [batch_size]

        Returns:
            tuple: (x_t, noise) where:
                - x_t: Noisy image at timestep t
                - noise: The Gaussian noise that was added
        """
        t = t.int()
        noise = torch.randn_like(x_0)  # Sample random noise from N(0,1)
        sqrt_a_bar_t = self.sqrt_a_bar[t, None, None,
                                       None]  # Broadcast to image dimensions
        sqrt_one_minus_a_bar_t = self.sqrt_one_minus_a_bar[t, None, None, None]

        x_t = sqrt_a_bar_t * x_0 + sqrt_one_minus_a_bar_t * noise
        return x_t, noise

    def get_loss(self, model, x_0, t, c, c_mask):
        """
        Calculate the loss for training the denoising model with context.

        Parameters
        ----------
        model : nn.Module
            The denoising neural network model.
        x_0 : torch.Tensor
            Clean input images of shape (batch_size, channels, height, width).
        t : torch.Tensor
            Timesteps for each image in the batch of shape (batch_size,).
        c : torch.Tensor
            Context embeddings of shape (batch_size, context_dim).
        c_mask : torch.Tensor
            Context mask of shape (batch_size, 1) for dropping context.

        Returns
        -------
        torch.Tensor
            The computed loss value.
        """
        x_noisy, noise = self.q(x_0, t)  # Add noise to clean images
        noise_pred = model(x_noisy, t, c, c_mask)  # Model predicts the noise
        return F.mse_loss(noise,
                          noise_pred)  # Compare actual vs predicted noise

    def get_context_mask(self, c, c_drop_prob):
        """
        Generate a mask to randomly drop context information during training.

        Parameters
        ----------
        c : torch.Tensor
            Context tensor of shape (batch_size, context_dim).
        c_drop_prob : float
            Probability of dropping context for each sample.

        Returns
        -------
        torch.Tensor
            Binary mask tensor of shape (batch_size, 1) where 0 indicates dropped context.
        """
        c_mask = torch.bernoulli(torch.ones_like(c).float() - c_drop_prob).to(
            self.device)
        return c_mask

    @torch.no_grad()
    def reverse_q(self, x_t, t, e_t):
        """
        Reverse diffusion process p(x_{t-1} | x_t).

        Takes a noisy image at timestep t and removes some noise to get x_{t-1}.
        This implements the reverse diffusion equation:
        μ_t = (1/√α_t) * (x_t - ((1-α_t)/√(1-ᾱ_t)) * ε_t)
        x_{t-1} = μ_t + σ_t * z (where z ~ N(0,1) if t > 0)

        Args:
            x_t (torch.Tensor): Noisy image at current timestep t
            t (torch.Tensor): Current timestep(s)
            e_t (torch.Tensor): Predicted noise from the model

        Returns:
            torch.Tensor: Less noisy image at timestep t-1
        """
        t = t.int()
        pred_noise_coeff_t = self.pred_noise_coeff[
            t]  # Coefficient for noise removal
        sqrt_a_inv_t = self.sqrt_a_inv[t]  # Scaling factor

        # Calculate the mean of the reverse distribution
        u_t = sqrt_a_inv_t * (x_t - pred_noise_coeff_t * e_t)

        if t[0] == 0:  # All t values should be the same
            return u_t  # At t=0, no additional noise is added (final clean image)
        else:
            # For t > 0, add noise from previous timestep according to the variance schedule
            B_t = self.B[t - 1]  # Beta at previous timestep
            new_noise = torch.randn_like(x_t)  # Fresh random noise
            return u_t + torch.sqrt(
                B_t) * new_noise  # Add stochastic component

    @torch.no_grad()
    def sample_images(self,
                      model,
                      img_ch,
                      img_size,
                      ncols,
                      *model_args,
                      axis_on=False):
        """
        Generate and visualize images using the trained DDPM model.

        Starts from pure noise and iteratively denoises it over T timesteps,
        showing intermediate results at regular intervals.

        Args:
            model: Trained neural network that predicts noise
            img_ch (int): Number of image channels (e.g., 3 for RGB)
            img_size (int): Height/width of square images
            ncols (int): Number of columns in the visualization grid
            *model_args: Additional arguments for the model (context, masks, etc.)
            axis_on (bool): Whether to show axes on the plots
        """
        # Start from pure Gaussian noise
        x_t = torch.randn((1, img_ch, img_size, img_size), device=self.device)
        plt.figure(figsize=(8, 8))
        hidden_rows = self.T / ncols  # Determines how often to show intermediate results
        plot_number = 1

        # Reverse diffusion: Go from T to 0, removing noise step by step
        for i in range(0, self.T)[::-1]:  # T-1, T-2, ..., 1, 0
            t = torch.full((1, ), i, device=self.device).float()
            e_t = model(x_t, t, *model_args)  # Model predicts noise to remove
            x_t = self.reverse_q(x_t, t, e_t)  # Remove predicted noise

            # Show intermediate results at regular intervals
            if i % hidden_rows == 0:
                ax = plt.subplot(1, ncols + 1, plot_number)
                if not axis_on:
                    ax.axis('off')
                generation_image(x_t.detach().cpu())
                plot_number += 1
        plt.show()


# Visualization function for diffusion process analysis
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
                                top_k=5):
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

    Returns
    -------
    None
        Displays matplotlib figures.
    """
    # CIFAR-100 class names as our vocabulary
    cifar100_labels = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee',
        'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus',
        'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
        'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab',
        'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish',
        'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
        'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man',
        'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom',
        'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
        'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper',
        'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
        'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale',
        'willow_tree', 'wolf', 'woman', 'worm'
    ]

    if timesteps_to_show is None:
        # Show 8 evenly spaced timesteps from 0 to T-1 (clean to noisy)
        timesteps_to_show = torch.linspace(0, ddpm.T - 1, 8).long().tolist()
        print('Showing timesteps:', timesteps_to_show)

    n_steps = len(timesteps_to_show)

    # Prepare text embeddings for comparison - use clip.tokenize, not clip_model.tokenize
    text_tokens = clip.tokenize(cifar100_labels).to(device)
    text_embeddings = clip_model.encode_text(text_tokens).float()
    text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)

    # Create figure with subplots
    fig, axes = plt.subplots(n_steps, 4, figsize=(12, n_steps * 2))
    if n_steps == 1:
        axes = axes.reshape(1, -1)

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.05, hspace=0.1)

    # Column headers
    col_titles = [
        'Diffusion Step', 'Top CLIP Match (Image)', 'Added Noise',
        'Top CLIP Match (Noise)'
    ]
    for i, title in enumerate(col_titles):
        axes[0, i].set_title(title, fontsize=14, fontweight='bold')

    stored_images = []
    stored_noises = []

    n_samples = 1

    # Embed conditioning text
    text_tokens = clip.tokenize(prompt).to(device)
    c = clip_model.encode_text(text_tokens).float()

    # Don't drop context for generation
    c_mask = torch.ones(embed_size).to(device)
    c_mask[n_samples:] = 0.0

    # **** Reverse diffusion process to generate the image
    # Start from pure Gaussian noise
    x_t = torch.randn((1, img_ch, img_size, img_size), device=ddpm.device)
    # Go from T to 0, removing noise step by step
    for i in range(0, ddpm.T)[::-1]:  # T-1, T-2, ..., 1, 0
        t = torch.full((1, ), i, device=ddpm.device).float()
        # Double the batch (to have condition+uncondition)
        x_t = x_t.repeat(2, 1, 1, 1)
        t = t.repeat(2, 1, 1, 1)

        # Find the noise
        e_t = model(x_t, t, c, c_mask)
        e_t_keep_c = e_t[:n_samples]
        e_t_drop_c = e_t[n_samples:]
        # This is the weighted conditioning
        e_t = (1 + w) * e_t_keep_c - w * e_t_drop_c

        # Deduplicate batch for reverse diffusion
        x_t = x_t[:n_samples]
        t = t[:n_samples]
        x_t = ddpm.reverse_q(x_t, t, e_t)

        if i in timesteps_to_show:
            # Store the (first) image at this timestep
            stored_images.append(x_t[0].clone())
            # Store the (first) predicted noise at this timestep
            stored_noises.append(e_t[0].clone())

    for idx, t_val in enumerate(timesteps_to_show):
        t = torch.tensor([t_val], device=device).float()

        # Column 1: Show the noisy image
        # Convert tensor to displayable format using your existing to_image function
        img_pil = to_image(stored_images[idx].detach().cpu())
        axes[idx, 0].imshow(img_pil)
        axes[idx, 0].set_ylabel(f't={t_val}', fontsize=12)
        axes[idx, 0].axis('off')

        # Column 2: CLIP interpretation of the noisy image
        try:
            # Process noisy image for CLIP using your existing to_image function
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
                label = cifar100_labels[top_image_indices[i].item()]
                score = top_image_values[i].item()
                image_matches.append(f"{label}: {score:.2f}")
            image_text = '\n'.join(image_matches)

            axes[idx, 1].text(0.5,
                              0.5,
                              image_text,
                              ha='center',
                              va='center',
                              transform=axes[idx, 1].transAxes,
                              fontsize=8,
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
                              fontsize=8)
        axes[idx, 1].axis('off')

        # Column 3: Show the noise that was added
        # Normalize noise for visualization (convert from [-1,1] to [0,1])
        noise_vis = stored_noises[idx]
        if noise_vis.shape[0] == 3:  # RGB
            axes[idx, 2].imshow(noise_vis.permute(1, 2, 0).cpu().numpy())
        else:  # Grayscale
            axes[idx, 2].imshow(noise_vis[0].cpu().numpy(), cmap='gray')
        axes[idx, 2].axis('off')

        # Column 4: CLIP interpretation of the noise
        try:
            # Process noise for CLIP using your existing to_image function
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
                label = cifar100_labels[top_noise_indices[i].item()]
                score = top_noise_values[i].item()
                noise_matches.append(f"{label}: {score:.2f}")
            noise_text = '\n'.join(noise_matches)

            axes[idx, 3].text(0.5,
                              0.5,
                              noise_text,
                              ha='center',
                              va='center',
                              transform=axes[idx, 3].transAxes,
                              fontsize=8,
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
                              fontsize=8)
        axes[idx, 3].axis('off')

    plt.suptitle(
        f"DDPM process + CLIP conditioning:'{prompt}' with weight {w}",
        fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.98], pad=0.5)
    plt.show()

    return stored_images, stored_noises


@torch.no_grad()
def sample_w(model,
             ddpm,
             input_size,
             T,
             c,
             device,
             w_tests=None,
             store_freq=10):
    """
    Generate samples using classifier-free guidance.

    Parameters
    ----------
    model : nn.Module
        The trained denoising model.
    ddpm : DDPM
        DDPM instance containing diffusion parameters.
    input_size : tuple of int
        Shape of input images (channels, height, width).
    T : int
        Number of diffusion timesteps.
    c : torch.Tensor
        Context embeddings of shape (batch_size, context_dim).
    device : str or torch.device
        Device to run computations on.
    w_tests : list of float, optional
        Guidance weights to test. Negative values reduce conditioning,
                       positive values strengthen it. Default: [-2, -1, -0.5, 0, 0.5, 1, 2]
    store_freq : int, optional
        How often to store intermediate results for animation, by default 10

    Returns
    -------
    x_i : torch.Tensor
        Final generated images of shape (batch_size, channels, height, width).
    x_i_store : list of torch.Tensor
        List of intermediate images during the denoising process.
    """
    if w_tests is None:
        w_tests = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    # Preprase "grid of samples" with w for rows and c for columns
    n_samples = len(w_tests) * len(c)

    # One w for each c
    w = torch.tensor(w_tests).float().repeat_interleave(len(c))
    w = w[:, None, None, None].to(device)  # Make w broadcastable
    x_t = torch.randn(n_samples, *input_size).to(device)

    # One c for each w
    c = c.repeat(len(w_tests), 1)

    # Double the batch
    c = c.repeat(2, 1)

    # Don't drop context at test time
    c_mask = torch.ones_like(c).to(device)
    c_mask[n_samples:] = 0.0

    x_t_store = []
    for i in range(0, T)[::-1]:
        # Duplicate t for each sample
        t = torch.tensor([i]).to(device)
        t = t.repeat(n_samples, 1, 1, 1)

        # Double the batch
        x_t = x_t.repeat(2, 1, 1, 1)
        t = t.repeat(2, 1, 1, 1)

        # Find weighted noise
        e_t = model(x_t, t, c, c_mask)
        e_t_keep_c = e_t[:n_samples]
        e_t_drop_c = e_t[n_samples:]
        e_t = (1 + w) * e_t_keep_c - w * e_t_drop_c

        # Deduplicate batch for reverse diffusion
        x_t = x_t[:n_samples]
        t = t[:n_samples]
        x_t = ddpm.reverse_q(x_t, t, e_t)

        # Store values for animation
        if i % store_freq == 0 or i == T or i < 10:
            x_t_store.append(x_t)

    x_t_store = torch.stack(x_t_store)
    return x_t, x_t_store
