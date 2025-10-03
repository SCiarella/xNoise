import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image as PILImage


def fig_to_image(fig):
    """
    Convert a matplotlib figure to a PIL Image.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The matplotlib figure to convert.

    Returns
    -------
    PIL.Image.Image
        Converted PIL Image.
    """
    import io
    # Save figure to a bytes buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img = PILImage.open(buf)
    img_copy = img.copy()
    buf.close()
    plt.close(fig)
    return img_copy


def save_animation(xs, gif_name, interval=300, repeat_delay=5000):
    """
    Save a list of images or figures as an animated GIF.

    Parameters
    ----------
    xs : list
        List of PIL Images or matplotlib figures to be saved in the gif.
    gif_name : str
        The name of the gif file.
    interval : int, optional
        Interval between frames in milliseconds (default is 300).
    repeat_delay : int, optional
        Delay before repeating the gif in milliseconds (default is 5000).

    Returns
    -------
    None
    """
    # Convert figures to images if needed
    images = []
    for x in xs:
        if hasattr(x, 'canvas'):  # It's a matplotlib figure
            images.append(fig_to_image(x))
        else:  # It's already a PIL Image
            images.append(x)

    # Save as GIF using PIL
    if images:
        images[0].save(gif_name,
                       save_all=True,
                       append_images=images[1:],
                       duration=interval,
                       loop=0)
        print(f"Animation saved to: {gif_name}")


def generation_image(image_list, text_list=None, w=None, save_path=None):
    """
    Display a list of tensor images with optional text labels.

    Parameters
    ----------
    image_list : list of torch.Tensor
        List of image tensors to display.
    text_list : list of str or None, optional
        List of text labels for each image. If None, no labels are shown, by default None.
    w : list of float or None, optional
        List of w values to display as row labels. If None, defaults to [-2, -1, 0, 1, 2].
    save_path : str or None, optional
        Path to save the figure. If None, figure is not saved, by default None.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib figure object containing the images.
    """
    if w is None:
        w = [-2, -1, 0, 1, 2]

    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: torch.minimum(torch.tensor([1]), t)),
        transforms.Lambda(lambda t: torch.maximum(torch.tensor([0]), t)),
        transforms.ToPILImage(),
    ])

    if text_list is not None:
        # Check if we have multiple individual images or a single grid
        if len(image_list) > 1:
            # Multiple individual images - much simpler!
            nrows = len(w) if w is not None else 1
            ncols = len(text_list)

            # Create subplot layout
            fig, axes = plt.subplots(nrows,
                                     ncols,
                                     figsize=(3 * ncols, 3 * nrows))
            if nrows == 1:
                axes = axes.reshape(1, -1)
            elif ncols == 1:
                axes = axes.reshape(-1, 1)

            # Add column labels (text_list)
            for j, text in enumerate(text_list):
                axes[0, j].set_title(text, fontsize=10, pad=10)

            # Display individual images
            for i in range(nrows):
                for j in range(ncols):
                    img_idx = i * ncols + j
                    if img_idx < len(image_list):
                        axes[i, j].imshow(
                            reverse_transforms(
                                image_list[img_idx].detach().cpu()))
                        axes[i, j].axis('off')

                # Add row label (w value) to the left of the first column
                if w is not None and i < len(w):
                    fig.text(0.02,
                             0.5 - (i - (nrows - 1) / 2) * (0.8 / nrows),
                             f'w = {w[i]}',
                             rotation=0,
                             fontsize=12,
                             ha='left',
                             va='center',
                             weight='bold')

            # Adjust layout to make room for row labels
            plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.1)

            # Save if path provided
            if save_path is not None:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Figure saved to: {save_path}")

            plt.close(fig)
            return fig

        else:
            # Single grid image that needs to be split - fallback for compatibility
            grid_tensor = image_list[0].detach().cpu()

            # Assume the grid is arranged as: rows=w_values, cols=text_prompts
            nrows = len(w) if w is not None else 1
            ncols = len(text_list)

            # make_grid creates padding, so we need to account for it
            padding = 2  # Default padding used by make_grid
            total_height, total_width = grid_tensor.shape[-2:]

            # Calculate individual image dimensions accounting for padding
            img_height = (total_height - padding * (nrows + 1)) // nrows
            img_width = (total_width - padding * (ncols + 1)) // ncols

            # Create subplot layout
            fig, axes = plt.subplots(nrows,
                                     ncols,
                                     figsize=(3 * ncols, 3 * nrows))
            if nrows == 1:
                axes = axes.reshape(1, -1)
            elif ncols == 1:
                axes = axes.reshape(-1, 1)

            # Add column labels (text_list)
            for j, text in enumerate(text_list):
                axes[0, j].set_title(text, fontsize=10, pad=10)

            # Split the grid and display individual images
            for i in range(nrows):
                for j in range(ncols):
                    # Calculate position accounting for padding
                    y_start = padding + i * (img_height + padding)
                    y_end = y_start + img_height
                    x_start = padding + j * (img_width + padding)
                    x_end = x_start + img_width

                    individual_img = grid_tensor[:, y_start:y_end,
                                                 x_start:x_end]
                    axes[i, j].imshow(reverse_transforms(individual_img))
                    axes[i, j].axis('off')

                # Add row label (w value) to the left of the first column
                if w is not None and i < len(w):
                    fig.text(0.02,
                             0.5 - (i - (nrows - 1) / 2) * (0.8 / nrows),
                             f'w = {w[i]}',
                             rotation=0,
                             fontsize=12,
                             ha='left',
                             va='center',
                             weight='bold')

            # Adjust layout to make room for row labels
            plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.1)

            # Save if path provided
            if save_path is not None:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Figure saved to: {save_path}")

            plt.close(fig)
            return fig
    else:
        # No text labels, just display the first image
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(reverse_transforms(image_list[0].detach().cpu()))
        plt.axis('off')

        # Save if path provided
        if save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        plt.close(fig)
        return fig


def to_image(tensor):
    """
    Convert a tensor to a PIL Image.

    Parameters
    ----------
    tensor : torch.Tensor
        Image tensor of shape (channels, height, width).

    Returns
    -------
    PIL.Image.Image
        Converted PIL Image.
    """
    tensor = (tensor + 1) / 2
    ones = torch.ones_like(tensor)
    tensor = torch.min(torch.stack([tensor, ones]), 0)[0]
    zeros = torch.zeros_like(tensor)
    tensor = torch.max(torch.stack([tensor, zeros]), 0)[0]
    return transforms.functional.to_pil_image(tensor)
