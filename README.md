# Understanding Noise in Text-to-Image Generation

A PyTorch implementation of CLIP-guided Denoising Diffusion Probabilistic Models (DDPM) with emphasis on **interpretability**. This project explores how diffusion models learn to encode semantic information in noise patterns.

## 🎯 Key Features

- **Explainability First**: Analyze and visualize how noise encodes semantic content
- **Timestep Attribution**: Identify which denoising steps are most critical
- **Guidance Visualization**: Quantify the effect of text conditioning on generation
- **Process Transparency**: Animations and analysis tools to understand the "black box"

## 🔬 Core Concepts

**Diffusion Models** learn to predict noise rather than generate pixels directly. By understanding noise patterns, they can reverse the corruption process—transforming pure noise into images.

**CLIP** aligns text and images in a shared embedding space, enabling text-guided generation without requiring text captions in training data.

**Classifier-Free Guidance** amplifies semantic control via:
$$\tilde{\epsilon}_\theta(x_t, c) = (1 + w) \cdot \epsilon_\theta(x_t, c) - w \cdot \epsilon_\theta(x_t, \emptyset)$$

This allows us to visualize how text changes noise prediction and tune guidance strength.

## 📚 Complete Tutorial

👉 **Start here: [examples/test.ipynb](examples/test.ipynb)**

The notebook provides comprehensive coverage of:
- Theoretical foundations and mathematical background
- Step-by-step implementation details
- Training on Tiny ImageNet with CLIP conditioning
- xAI analysis: noise patterns, denoising trajectories, and guidance effects
- Visualizations and animations of the generation process

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/yourusername/ddpm-clip.git
cd ddpm-clip
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .

# Launch interactive tutorial
jupyter notebook examples/test.ipynb
```

**Generate images** (after training):
```bash
python scripts/generate.py \
    --checkpoint model_checkpoint/9.pth \
    --prompts "A goldfish" "An oak tree" \
    --output images/generated.png
```

## 🏗️ Project Structure

```
├── examples/test.ipynb       # 📓 Complete tutorial (START HERE)
├── src/ddpm_clip/           # Core implementation
│   ├── models/              # UNet and DDPM
│   ├── data/                # Dataset utilities
│   └── utils/               # Visualization tools
├── scripts/                 # Training & inference scripts
└── config/                  # Hyperparameter configs
```

## 🎨 Usage Examples

**Train your model:**
```bash
python scripts/preprocess_clip.py --data_dir data/tiny-imagenet-200/train
python scripts/train.py --config config/default.yaml
```

**Analyze noise patterns:**
```python
from ddpm_clip.models import visualize_diffusion_process

visualize_diffusion_process(
    ddpm=ddpm, model=model, prompt="A tabby cat",
    w=1, top_k=7  # Show 7 most critical timesteps
)
```

**Compare guidance strengths:**
```python
x_gen, x_gen_store = sample_tinyimg(
    text_list=["A goldfish", "An oak tree"],
    w_values=[-2, -1, 0, 1, 2]
)
```

## 🔍 xAI Insights

Through analysis in the notebook, you'll discover:
- Early timesteps establish global structure (composition, layout)
- Middle timesteps define semantic content (object identity)
- Late timesteps refine details (texture, edges)
- Different prompts may require different guidance weights for optimal results

## ⚙️ Key Configuration

```yaml
timesteps: 400              # Diffusion steps
batch_size: 16
learning_rate: 0.0001
c_drop_prob: 0.1           # Context dropout for classifier-free guidance
down_channels: [256, 256, 512]
```

## 📊 Dataset

Uses [Tiny ImageNet](http://cs231n.stanford.edu/tiny-imagenet-200.zip): 200 classes, 500 images/class, 64×64 resolution.

**Custom data**: Run `scripts/preprocess_clip.py` to extract CLIP embeddings from your images.

## 🐛 Common Issues

| Issue | Solution |
|-------|----------|
| CUDA OOM | Reduce `batch_size` or `img_size` |
| Poor quality | Train longer, increase data, adjust `w` |
| Slow training | Use preprocessed CLIP embeddings |

## 🎓 Key References

- [DDPM (Ho et al., 2020)](https://arxiv.org/abs/2006.11239) - Foundational diffusion paper
- [CLIP (Radford et al., 2021)](https://arxiv.org/abs/2103.00020) - Vision-language alignment
- [Classifier-Free Guidance (Ho & Salimans, 2022)](https://arxiv.org/abs/2207.12598) - Guidance technique

## 📄 Citation

```bibtex
@article{ho2020denoising,
  title={Denoising Diffusion Probabilistic Models},
  author={Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  journal={NeurIPS},
  year={2020}
}
```

## 📜 License

MIT License - see [LICENSE](LICENSE)

---

**📓 For theory, implementation details, and xAI analysis, see [examples/test.ipynb](examples/test.ipynb)**
