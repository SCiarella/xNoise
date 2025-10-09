#!/usr/bin/env python3
"""
Generate a thumbnail for the xNoise project.
Shows the progression from noise to image with CLIP guidance.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def create_thumbnail(output_path='thumbnail.png', dpi=300):
    """Create a clean, image-only thumbnail showing noise-to-image progression."""

    # Create figure with no margins
    fig = plt.figure(figsize=(16, 9), facecolor='black')
    gs = GridSpec(3,
                  6,
                  figure=fig,
                  wspace=0.02,
                  hspace=0.02,
                  left=0.01,
                  right=0.99,
                  top=0.99,
                  bottom=0.01)

    # Generate sample noise patterns that progress from random to structured
    np.random.seed(42)

    # Three different prompts for variety
    prompts_config = [{
        'name': 'goldfish',
        'colors': [(0.9, 0.5, 0.1), (0.1, 0.4, 0.7)],
        'pattern': 'radial'
    }, {
        'name': 'tree',
        'colors': [(0.3, 0.6, 0.2), (0.4, 0.3, 0.2)],
        'pattern': 'vertical'
    }, {
        'name': 'sunset',
        'colors': [(1.0, 0.6, 0.2), (0.3, 0.2, 0.5)],
        'pattern': 'gradient'
    }]

    for row, config in enumerate(prompts_config):
        for col in range(6):
            ax = fig.add_subplot(gs[row, col])

            # Create progressively less noisy images (0 = pure noise, 5 = clean)
            noise_level = 1.0 - (col / 5.0)

            # Generate base pattern
            size = 128
            base = np.zeros((size, size, 3))
            x, y = np.meshgrid(np.linspace(-1, 1, size),
                               np.linspace(-1, 1, size))

            if config['pattern'] == 'radial':
                # Radial pattern (goldfish)
                r = np.sqrt(x**2 + y**2)
                theta = np.arctan2(y, x)
                pattern = np.cos(theta * 3) * np.exp(-r)

                for c in range(3):
                    base[:, :, c] = config['colors'][0][c] * (
                        1 - r) + config['colors'][1][c] * r
                    base[:, :, c] += pattern * 0.2

            elif config['pattern'] == 'vertical':
                # Vertical pattern (tree)
                for c in range(3):
                    base[:, :, c] = config['colors'][0][c] * (
                        1 - y * 0.5) + config['colors'][1][c] * (y * 0.5 + 0.5)
                # Add some texture
                texture = np.sin(x * 10) * np.sin(y * 8) * 0.1
                base += texture[:, :, np.newaxis]

            else:  # gradient (sunset)
                gradient = (y + 1) / 2
                for c in range(3):
                    base[:, :, c] = config['colors'][0][c] * (
                        1 - gradient) + config['colors'][1][c] * gradient
                # Add sun-like circle
                sun_r = np.sqrt(x**2 + (y + 0.3)**2)
                sun = np.exp(-sun_r * 8)
                base[:, :, 0] += sun * 0.3
                base[:, :, 1] += sun * 0.3

            # Add structured noise that decreases over denoising steps
            noise = np.random.randn(size, size, 3) * 0.5

            # Blend based on denoising progress
            image = base * (1 - noise_level) + (base * 0.5 +
                                                noise * 0.5) * noise_level
            image = np.clip(image, 0, 1)

            # Apply progressive smoothing for later stages
            if col > 3:
                try:
                    from scipy.ndimage import gaussian_filter
                    sigma = (col - 3) * 0.7
                    for c in range(3):
                        image[:, :, c] = gaussian_filter(image[:, :, c],
                                                         sigma=sigma)
                except ImportError:
                    pass

            # Add slight vignette effect for final image
            if col == 5:
                r = np.sqrt(x**2 + y**2)
                vignette = 1 - (r * 0.3)
                image *= vignette[:, :, np.newaxis]

            ax.imshow(image, aspect='auto', interpolation='bilinear')
            ax.axis('off')

    # Save with high quality, transparent or black background
    plt.savefig(output_path,
                dpi=dpi,
                bbox_inches='tight',
                facecolor='black',
                edgecolor='none',
                pad_inches=0)
    print(f"✓ Thumbnail saved to: {output_path}")
    plt.close()


if __name__ == '__main__':
    create_thumbnail('thumbnail.png', dpi=80)
    print('Done! Clean image-only thumbnail generated.')
