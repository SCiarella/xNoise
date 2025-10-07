# Understanding Noise in Text-to-Image Generation

A PyTorch implementation of CLIP-guided Denoising Diffusion Probabilistic Models (DDPM) focused on **explainability**. This project analyzes how diffusion models encode semantic information in noise patterns.

## Key Features

- **Explainability-Focused**: Visualize and analyze noise patterns during generation
- **CLIP-Guided**: Text-conditioned image generation using CLIP embeddings
- **Classifier-Free Guidance**: Adjustable guidance strength for semantic control
- **Educational**: Complete tutorial notebook with theory and implementation

![Diffusion Process Animation](images/analysis_generation.gif)

## Quick Start

```bash
# Install
pip install -e .

# Train on Tiny ImageNet
python examples/train.py --config config/model_default.yaml

# Explore the analysis notebook
jupyter notebook examples/explain_and_visualize.ipynb
```

## Project Structure

```
├── examples/
│   ├── explain_and_visualize.ipynb  # Main tutorial (START HERE)
│   ├── train_model.ipynb           # Training walkthrough
│   └── train.py                    # Training script
├── src/ddpm_clip/
│   ├── models/     # UNet, DDPM, EMA
│   ├── data/       # CLIP dataset and preprocessing
│   └── utils/      # Visualization and config utilities
└── config/         # Model configurations (small/default/large)
```

## Training

The training script supports multiple configurations and automatic checkpointing:

```bash
# Basic training
python examples/train.py --config config/model_default.yaml

# Skip CLIP extraction (if already done)
python examples/train.py --config config/model_small.yaml --skip-clip-extraction

# Disable animation generation
python examples/train.py --config config/model_large.yaml --no-animation
```

Training automatically resumes from the latest checkpoint if one exists.

## Analysis

The `explain_and_visualize.ipynb` notebook provides:
- Theoretical foundations of diffusion models
- Noise pattern analysis and interpretation
- Timestep attribution to identify critical denoising steps
- Guidance visualization to understand text conditioning effects
- Generation animations and trajectory analysis

## Dataset

Tested on [Tiny ImageNet](http://cs231n.stanford.edu/tiny-imagenet-200.zip) (200 classes, 64×64 images). CLIP embeddings are automatically extracted during training or can be precomputed.

## References

- [DDPM (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)
- [CLIP (Radford et al., 2021)](https://arxiv.org/abs/2103.00020)
- [Classifier-Free Guidance (Ho & Salimans, 2022)](https://arxiv.org/abs/2207.12598)

## License

MIT License - see [LICENSE](LICENSE)
