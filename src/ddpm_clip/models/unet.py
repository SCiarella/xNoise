import math
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class GELUConvBlock(nn.Module):

    def __init__(self, in_ch, out_ch, group_size):
        super().__init__()
        # Use inplace operations where possible
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.norm = nn.GroupNorm(group_size, out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class RearrangePoolBlock(nn.Module):

    def __init__(self, in_chs, group_size):
        super().__init__()
        self.rearrange = Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w',
                                   p1=2,
                                   p2=2)
        self.conv = GELUConvBlock(4 * in_chs, in_chs, group_size)

    def forward(self, x):
        return self.conv(self.rearrange(x))


class DownBlock(nn.Module):

    def __init__(self, in_chs, out_chs, group_size):
        super(DownBlock, self).__init__()
        layers = [
            GELUConvBlock(in_chs, out_chs, group_size),
            GELUConvBlock(out_chs, out_chs, group_size),
            RearrangePoolBlock(out_chs, group_size),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UpBlock(nn.Module):

    def __init__(self, in_chs, out_chs, group_size):
        super(UpBlock, self).__init__()
        layers = [
            nn.ConvTranspose2d(2 * in_chs, out_chs, 2, 2),
            GELUConvBlock(out_chs, out_chs, group_size),
            GELUConvBlock(out_chs, out_chs, group_size),
            GELUConvBlock(out_chs, out_chs, group_size),
            GELUConvBlock(out_chs, out_chs, group_size),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class SinusoidalPositionEmbedBlock(nn.Module):
    """
    Sinusoidal position embeddings for encoding timestep information.

    Parameters
    ----------
    dim : int
        Dimension of the embedding.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        Generate sinusoidal embeddings for given timesteps.

        Parameters
        ----------
        time : torch.Tensor
            Timesteps of shape (batch_size,).

        Returns
        -------
        torch.Tensor
            Position embeddings of shape (batch_size, dim).
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(
            torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class EmbedBlock(nn.Module):
    """
    Embedding block for processing time or context embeddings.

    Parameters
    ----------
    input_dim : int
        Dimension of input embeddings.
    emb_dim : int
        Dimension of output embeddings.
    """

    def __init__(self, input_dim, emb_dim):
        super(EmbedBlock, self).__init__()
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
            nn.Unflatten(1, (emb_dim, 1, 1)),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the embedding block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, emb_dim, 1, 1).
        """
        x = x.view(-1, self.input_dim)
        return self.model(x)


class ResidualConvBlock(nn.Module):

    def __init__(self, in_chs, out_chs, group_size):
        super().__init__()
        self.conv1 = GELUConvBlock(in_chs, out_chs, group_size)
        self.conv2 = GELUConvBlock(out_chs, out_chs, group_size)
        # Add projection if channels don't match
        self.shortcut = nn.Conv2d(in_chs, out_chs,
                                  1) if in_chs != out_chs else nn.Identity()

    def forward(self, x):
        return self.shortcut(x) + self.conv2(self.conv1(x))


class UNet(nn.Module):
    """
    U-Net architecture for conditional image denoising in diffusion models.

    Parameters
    ----------
    T : int
        Number of diffusion timesteps.
    ch : int
        Number of input/output channels.
    img_size : int
        Size of input images (height and width).
    down_chs : tuple of int, optional
        Number of channels for each downsampling level, by default (256, 256, 512).
    t_embed_dim : int, optional
        Dimension of time embeddings, by default 8.
    c_embed_dim : int, optional
        Dimension of context embeddings, by default 512.
    """

    def __init__(self,
                 T,
                 img_ch,
                 img_size,
                 down_chs=(64, 64, 128),
                 t_embed_dim=8,
                 c_embed_dim=10):
        super().__init__()
        self.T = T
        up_chs = down_chs[::-1]
        latent_image_size = img_size // 4
        small_group_size = 8
        big_group_size = 32

        # Initial convolution
        self.down0 = ResidualConvBlock(img_ch, down_chs[0], small_group_size)

        # Downsample
        self.down1 = DownBlock(down_chs[0], down_chs[1], big_group_size)
        self.down2 = DownBlock(down_chs[1], down_chs[2], big_group_size)

        # Store dimensions for reshape
        self.latent_dim = down_chs[2] * latent_image_size**2
        self.latent_shape = (down_chs[2], latent_image_size, latent_image_size)

        # Simplified bottleneck - consider replacing with attention layers
        self.dense_emb = nn.Sequential(
            nn.Linear(self.latent_dim, down_chs[1]),
            nn.GELU(),
            nn.Linear(down_chs[1], self.latent_dim),
            nn.GELU(),
        )

        # Time embeddings
        self.sinusoidaltime = SinusoidalPositionEmbedBlock(t_embed_dim)
        self.t_emb1 = EmbedBlock(t_embed_dim, up_chs[0])
        self.t_emb2 = EmbedBlock(t_embed_dim, up_chs[1])

        # Context embeddings
        self.c_embed1 = EmbedBlock(c_embed_dim, up_chs[0])
        self.c_embed2 = EmbedBlock(c_embed_dim, up_chs[1])

        # Upsample
        self.up0 = GELUConvBlock(up_chs[0], up_chs[0], big_group_size)
        self.up1 = UpBlock(up_chs[0], up_chs[1], big_group_size)
        self.up2 = UpBlock(up_chs[1], up_chs[2], big_group_size)

        # Output
        self.out = nn.Sequential(
            nn.Conv2d(2 * up_chs[-1], up_chs[-1], 3, 1, 1),
            nn.GroupNorm(small_group_size, up_chs[-1]),
            nn.GELU(),
            nn.Conv2d(up_chs[-1], img_ch, 3, 1, 1),
        )

    def forward(self, x, t, c, c_mask):
        """
        Forward pass through the U-Net with context conditioning.

        Parameters
        ----------
        x : torch.Tensor
            Input noisy images of shape (batch_size, ch, img_size, img_size).
        t : torch.Tensor
            Timesteps of shape (batch_size,).
        c : torch.Tensor
            Context embeddings of shape (batch_size, c_embed_dim).
        c_mask : torch.Tensor
            Context mask of shape (batch_size, 1) for dropping context.

        Returns
        -------
        torch.Tensor
            Predicted noise of shape (batch_size, ch, img_size, img_size).
        """
        down0 = self.down0(x)
        down1 = self.down1(down0)
        down2 = self.down2(down1)

        # Flatten and process through bottleneck
        latent_vec = down2.flatten(1)
        latent_vec = self.dense_emb(latent_vec)
        latent_vec = latent_vec.view(-1, *self.latent_shape)

        # Time embeddings
        t = self.sinusoidaltime(t.float() / self.T)
        t_emb1 = self.t_emb1(t)
        t_emb2 = self.t_emb2(t)

        # Context embeddings with masking
        c_masked = c * c_mask
        c_emb1 = self.c_embed1(c_masked)
        c_emb2 = self.c_embed2(c_masked)

        # Upsample with conditioning
        up0 = self.up0(latent_vec)
        up1 = self.up1(c_emb1 * up0 + t_emb1, down2)
        up2 = self.up2(c_emb2 * up1 + t_emb2, down1)

        return self.out(torch.cat((up2, down0), 1))
