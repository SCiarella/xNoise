import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from ..utils.visualization import generation_image


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
    def generate_images_with_guidance(self,
                                      model,
                                      img_ch,
                                      img_size,
                                      text_embedding,
                                      n_samples=1,
                                      w=0.5,
                                      timesteps_to_store=None,
                                      return_intermediate=False):
        """
        Core image generation function with classifier-free guidance.

        This is the shared implementation used by all generation functions to avoid code duplication.

        Parameters
        ----------
        model : nn.Module
            The trained denoising model.
        img_ch : int
            Number of image channels.
        img_size : int
            Size of square images (height and width).
        text_embedding : torch.Tensor
            CLIP text embedding of shape (1, embed_dim).
        n_samples : int, optional
            Number of samples to generate in parallel, by default 1.
        w : float, optional
            Guidance weight for classifier-free guidance, by default 0.5.
        timesteps_to_store : list of int or None, optional
            Specific timesteps at which to store intermediate results.
            If None, no intermediate results are stored.
        return_intermediate : bool, optional
            Whether to return intermediate results, by default False.

        Returns
        -------
        x_t : torch.Tensor
            Final generated images of shape (n_samples, channels, height, width).
        stored_data : dict or None
            If return_intermediate is True, returns dict with:
            - 'images': List of image tensors at specified timesteps
            - 'noises_conditioned': List of conditioned noise predictions
            - 'noises_unconditioned': List of unconditioned noise predictions
            - 'timesteps': List of timesteps where data was stored
        """
        # Prepare context embeddings (conditioned + unconditioned)
        c = text_embedding.repeat(n_samples, 1)
        c = c.repeat(2, 1)  # Double for conditioned + unconditioned
        c_mask = torch.ones_like(c).to(self.device)
        c_mask[n_samples:] = 0.0

        # Initialize storage
        stored_data = None
        if return_intermediate:
            stored_data = {
                'images': [],
                'noises_conditioned': [],
                'noises_unconditioned': [],
                'timesteps': []
            }

        # Start from pure Gaussian noise
        x_t = torch.randn((n_samples, img_ch, img_size, img_size),
                          device=self.device)

        # Reverse diffusion: Go from T to 0, removing noise step by step
        for i in range(0, self.T)[::-1]:  # T-1, T-2, ..., 1, 0
            t = torch.tensor([i], device=self.device)
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

            # Perform reverse diffusion step
            x_t = self.reverse_q(x_t, t, e_t_guided)

            # Store intermediate results if requested
            if return_intermediate and timesteps_to_store is not None and i in timesteps_to_store:
                stored_data['images'].append(x_t.clone())
                stored_data['noises_conditioned'].append(e_t_keep_c.clone())
                stored_data['noises_unconditioned'].append(e_t_drop_c.clone())
                stored_data['timesteps'].append(i)

        if return_intermediate:
            return x_t, stored_data
        return x_t

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
