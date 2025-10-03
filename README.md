# CLIP-Guided DDPM for Text-to-Image Generation

A PyTorch implementation of Denoising Diffusion Probabilistic Models (DDPM) guided by CLIP embeddings for text-to-image generation. This project demonstrates how to use CLIP encodings to condition a diffusion model without requiring explicit text descriptions for each training image.

## Overview

This project implements a text-to-image generation system using:
- **DDPM (Denoising Diffusion Probabilistic Models)**: A generative model that learns to denoise images through a reverse diffusion process
- **CLIP (Contrastive Language-Image Pre-Training)**: Used to align text and image embeddings, enabling text-guided generation
- **UNet Architecture**: A neural network with skip connections for learning the denoising process

The key insight is that we can train a diffusion model using only image CLIP encodings (without text descriptions) and still generate images from text prompts at inference time.

## Features

- ✨ Text-to-image generation using natural language prompts
- 🎨 CLIP-guided diffusion for semantic control
- 🔧 Modular architecture with reusable components
- 📊 Visualization tools for the diffusion process
- 💾 Training checkpoint management
- 🎬 Animation generation for the denoising process
- 📈 Classifier-free guidance with configurable strength

## Project Structure

```
.
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore rules
├── LICENSE                     # Project license
├── setup.py                    # Package installation configuration
│
├── src/                        # Source code
│   └── ddpm_clip/  
│       ├── __init__.py
│       ├── models/            # Model architectures
│       │   ├── __init__.py
│       │   ├── unet.py       # UNet implementation
│       │   └── ddpm.py       # DDPM core logic
│       ├── data/             # Dataset utilities
│       │   ├── __init__.py
│       │   └── dataset.py    # Custom dataset classes
│       └── utils/            # Utility functions
│           ├── __init__.py
│           ├── visualization.py
│           └── training.py
│
├── scripts/                   # Training and inference scripts
│   ├── train.py              # Training script
│   ├── generate.py           # Image generation script
│   └── preprocess_clip.py    # CLIP encoding preprocessing
│
├── config/                   # Configuration files
│   ├── default.yaml         # Default hyperparameters
│   └── example.yaml         # Example configuration
│
├── examples/                 # Example notebooks and demos
│   └── test.ipynb           # Main demonstration notebook
│
├── data/                    # Dataset directory (not tracked)
│   └── tiny-imagenet-200/
│
├── model_checkpoint/        # Model checkpoints (not tracked)
├── images/                  # Generated images (not tracked)
└── TRASH/                   # Temporary files (not tracked)
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ GPU memory

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ddpm-clip.git
cd ddpm-clip
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package (this will install all dependencies automatically):
```bash
pip install -e .
```

Alternatively, you can install dependencies separately:
```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

### Quick Start with Jupyter Notebook

The easiest way to get started is with the provided Jupyter notebook:

```bash
jupyter notebook examples/test.ipynb
```

### Training from Scratch

1. **Prepare your dataset**: Place images in `data/tiny-imagenet-200/` or modify the path in the config

2. **Preprocess CLIP embeddings** (optional but recommended for faster training):
```bash
python scripts/preprocess_clip.py --data_dir data/tiny-imagenet-200/train --output clip.csv
```

3. **Train the model**:
```bash
python scripts/train.py --config config/default.yaml
```

### Generate Images

Generate images from text prompts:

```bash
python scripts/generate.py \
    --checkpoint model_checkpoint/9.pth \
    --prompts "A fish" "A tree" "A baby" \
    --output images/generated.png
```

### Python API

```python
import torch
from ddpm_clip.models import UNet, DDPM
import clip

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model, clip_preprocess = clip.load("ViT-B/32")

# Initialize DDPM
T = 400
B = torch.linspace(0.0001, 0.02, T).to(device)
ddpm = DDPM(B, device)

# Initialize UNet
model = UNet(T, img_ch=3, img_size=32, down_chs=(256, 256, 512),
             t_embed_dim=8, c_embed_dim=512)
model.load_state_dict(torch.load("model_checkpoint/9.pth")["model"])
model.to(device).eval()

# Generate images
text_prompts = ["A fish swimming", "A beautiful tree"]
text_tokens = clip.tokenize(text_prompts).to(device)
text_embeddings = clip_model.encode_text(text_tokens).float()

images = ddpm.sample(model, text_embeddings)
```

## Key Concepts

### CLIP Encoding
CLIP creates aligned embeddings for text and images in the same vector space. This allows the model to:
- Learn from image embeddings during training
- Generate images from text embeddings at inference time
- No need for explicit text captions in the training data

### Classifier-Free Guidance
The model supports classifier-free guidance, where we:
- Train with random context dropout (`c_drop_prob`)
- At inference, interpolate between conditional and unconditional predictions
- Control guidance strength with parameter `w`

### Diffusion Process
- **Forward process**: Gradually adds Gaussian noise to images over T timesteps
- **Reverse process**: Model learns to denoise, generating images from pure noise
- Uses a variance schedule (beta schedule) to control noise levels

## Configuration

Key hyperparameters in `config/default.yaml`:

```yaml
# Model parameters
img_size: 32
img_channels: 3
timesteps: 400
beta_start: 0.0001
beta_end: 0.02

# Training parameters
epochs: 10
batch_size: 16
learning_rate: 0.0001
c_drop_prob: 0.1

# UNet architecture
down_channels: [256, 256, 512]
t_embed_dim: 8
c_embed_dim: 512

# CLIP model
clip_model: "ViT-B/32"
```

## Dataset

This project uses [Tiny ImageNet](http://cs231n.stanford.edu/tiny-imagenet-200.zip), a subset of ImageNet with 200 classes and 500 training images per class (64x64 pixels).

To use your own dataset:
1. Organize images in a directory structure
2. Update `data_paths` in the config or script
3. Ensure images are in a format readable by PIL

## Training Details

- **Loss Function**: MSE between predicted and actual noise
- **Optimizer**: Adam with learning rate 1e-4
- **Data Augmentation**: Random crops and horizontal flips
- **Context Dropout**: 10% during training for classifier-free guidance
- **Checkpointing**: Model saved every 5 epochs

## Results

The model learns to generate images conditioned on text prompts. Example generations:

- "A fish" → Fish-like shapes and textures
- "A tree" → Tree structures and foliage patterns
- "A baby" → Face-like features

Note: Results depend on training data diversity and training duration.

## Visualization

The project includes tools to visualize:
- Generated images from text prompts
- The denoising process as an animated GIF
- CLIP embedding similarities between text and images
- Training loss over time

## Performance Tips

- **GPU Memory**: Reduce `batch_size` if you encounter OOM errors
- **Training Speed**: Preprocess CLIP embeddings to CSV for faster training
- **Quality**: Train for more epochs and with more diverse data
- **Generation**: Adjust classifier-free guidance strength `w` for different effects

## Troubleshooting

**Issue**: CUDA out of memory
- Solution: Reduce `batch_size` or `img_size`

**Issue**: Poor generation quality
- Solution: Train longer, use more training data, adjust learning rate

**Issue**: Slow training
- Solution: Use preprocessed CLIP embeddings, enable mixed precision training

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ho2020denoising,
  title={Denoising Diffusion Probabilistic Models},
  author={Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}

@article{radford2021learning,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and others},
  journal={International Conference on Machine Learning},
  year={2021}
}
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original DDPM paper by Ho et al.
- CLIP by OpenAI
- Inspired by Stable Diffusion and DALL-E architectures
- Built with PyTorch and various open-source libraries

## Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: This is a research/educational project. Generated images may not always match text descriptions perfectly, especially with limited training data.
