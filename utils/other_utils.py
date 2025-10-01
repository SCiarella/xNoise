import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import Image
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

def save_animation(xs, gif_name, interval=300, repeat_delay=5000):
    """
    Save a list of images as an animated GIF.
    
    Parameters
    ----------
    xs : list
        List of images to be saved in the gif.
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
    fig = plt.figure()
    plt.axis('off')
    imgs = []

    for x_t in xs:
        im = plt.imshow(x_t, animated=True)
        imgs.append([im])

    animate = animation.ArtistAnimation(fig, imgs, interval=interval, repeat_delay=repeat_delay)
    animate.save(gif_name)


def show_tensor_image(image_list, text_list=None):
    """
    Display a list of tensor images with optional text labels.
    
    Parameters
    ----------
    image_list : list of torch.Tensor
        List of image tensors to display.
    text_list : list of str or None, optional
        List of text labels for each image. If None, no labels are shown, by default None.
    
    Returns
    -------
    None
        Displays matplotlib figure.
    """
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: torch.minimum(torch.tensor([1]), t)),
        transforms.Lambda(lambda t: torch.maximum(torch.tensor([0]), t)),
        transforms.ToPILImage(),
    ])
    
    # Handle grid images (single tensor containing multiple images)
    if len(image_list) == 1 and text_list is not None and len(text_list) > 1:
        # This is a grid image, display it with a figure title
        plt.figure(figsize=(12, 4))
        plt.imshow(reverse_transforms(image_list[0].detach().cpu()))
        plt.axis('off')
        plt.title(' | '.join(text_list), fontsize=12)
    elif text_list is not None:
        # Display multiple images in subplots
        plt.figure(figsize=(12, 4))
        nrows = 1
        ncols = len(text_list)
        for i in range(len(text_list)):
            ax = plt.subplot(nrows, ncols, i+1)
            ax.set_title(text_list[i])
            ax.axis('off')
            if i < len(image_list):
                plt.imshow(reverse_transforms(image_list[i].detach().cpu()))
    else:
        # No text labels, just display the first image
        plt.figure(figsize=(8, 8))
        plt.imshow(reverse_transforms(image_list[0].detach().cpu()))
        plt.axis('off')
    

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


def plot_generated_images(noise, result):
    plt.figure(figsize=(8,8))
    nrows = 1
    ncols = 2
    samples = {
        "Noise" : noise,
        "Generated Image" : result
    }
    for i, (title, img) in enumerate(samples.items()):
        ax = plt.subplot(nrows, ncols, i+1)
        ax.set_title(title)
        show_tensor_image(img)
    plt.show()