# 🧬 Noise Genome Estimator

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

A deep learning framework for estimating noise parameters in images using contrastive learning and Vision Transformers.

---

## 📖 Overview

This project implements a noise estimation system that can characterize various types of image noise (Gaussian, salt-and-pepper, Poisson, quantization, and anisotropic noise) in a unified framework. The model uses a Vision Transformer (ViT) backbone with contrastive learning to learn robust noise representations.

## ✨ Key Features

| Feature | Description |
|---|---|
|  **Multiple Noise Types** | Gaussian, salt-and-pepper, Poisson, quantization, and anisotropic noise |
|  **Contrastive Learning** | Noise-independent contrastive loss for discriminative feature learning |
|  **Scalable Architecture** | Vision Transformer (ViT) and Swin Transformer backbone support |
|  **Distributed Training** | Multi-GPU training via PyTorch DistributedDataParallel |
|  **Mixed Precision** | Automatic mixed precision (AMP) for faster training |
|  **Comprehensive Evaluation** | MAE, MSE, RMSE, and R² metrics |

---

## 🛠️ Installation

### Requirements

- Python 3.8+
- CUDA 11.0+ (for GPU training)
- PyTorch 2.0+

### Setup

**1. Clone the repository:**
```bash
git clone https://github.com/FDU-donglab/CoP.git
cd CoP
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

> 💡 For PyTorch, it is recommended to install a version matching your CUDA environment from the [official website](https://pytorch.org/get-started/locally/) before running the above command.

**3. Create necessary directories:**
```bash
mkdir -p datasets/train datasets/val datasets/test checkpoints noise_params
```

---

## 📂 Dataset Structure

Organize your dataset as follows:

```
datasets/
├── train/
│   ├── image1.png
│   ├── image2.jpg
│   └── ...
├── val/
│   ├── image1.png
│   └── ...
└── test/
    ├── image1.png
    └── ...
```

Supported image formats: `.png`, `.jpg`, `.tif`, `.tiff`

---

## 🚀 Usage

### Training

```bash
# Single GPU
python train.py --mode train \
    --train-dataset-path ./datasets/train \
    --validation-dataset-path ./datasets/val \
    --num-epochs 300 \
    --batch-size 16

# Multi-GPU (DistributedDataParallel)
torchrun --nproc_per_node=4 train.py --mode train \
    --train-dataset-path ./datasets/train \
    --validation-dataset-path ./datasets/val
```

### Testing

```bash
python train.py --mode test \
    --checkpoint-load-path ./checkpoints/20250109_1230/model_epoch_300.pth \
    --test-image-path ./datasets/test
```

---

## ⚙️ Configuration

All parameters can be configured via command line arguments. Run `python train.py --help` for the full list.

Key options:

| Argument | Default | Description |
|---|---|---|
| `--model-type` | `vit` | Backbone: `vit` or `swin` |
| `--batch-size` | `16` | Training batch size |
| `--learning-rate` | `1.5e-4` | Initial learning rate |
| `--num-epochs` | `300` | Number of training epochs |
| `--crop-size-whole-xy` | `192` | Input image patch size |
| `--patch-size-in-tr` | `16` | Transformer patch embedding size |

---

## 🏛️ Model Architecture

### Backbone Options

- **Vision Transformer (ViT)** — Standard ViT with patch embeddings
- **Swin Transformer** — Hierarchical transformer with shifting windows

### Loss Function

The model optimizes a combined loss:

$$\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{contrast}} + \beta \cdot \mathcal{L}_{\text{noise}}$$

| Pre-training | Description | Default weight |
|---|---|---|
| $\mathcal{L}_{\text{contrast}}$ | InfoNCE-style contrastive loss | $\alpha = 1$ |
| $\mathcal{L}_{\text{noise}}$ | MSE loss for noise parameter regression | $\beta = 0$ |

| Fine-tuning | Description | Default weight |
|---|---|---|
| $\mathcal{L}_{\text{contrast}}$ | InfoNCE-style contrastive loss | $\alpha = 0$ |
| $\mathcal{L}_{\text{noise}}$ | MSE loss for noise parameter regression | $\beta = 1$ |

| Joint-learning | Description | Default weight |
|---|---|---|
| $\mathcal{L}_{\text{contrast}}$ | InfoNCE-style contrastive loss | $\alpha = 0.5$ |
| $\mathcal{L}_{\text{noise}}$ | MSE loss for noise parameter regression | $\beta = 0.5$ |

---
### Model Zoo
Our paper is under reviewed at Nature, and we will release the model weights and pre-trained models after the review.
## 🗂️ Project Structure

```
CoP/
├── src/
│   ├── __init__.py
│   ├── dataset.py          # Dataset classes
│   ├── engines.py          # Training and testing engines
│   ├── utils.py            # Utility functions
│   └── models/
│       ├── __init__.py
│       ├── masked_autoencoder.py
│       ├── vision_transformer.py
│       ├── swin_transformer.py
│       └── ...
├── configs/
│   └── default.py          # Default configuration
├── scripts/
│   ├── preprocess.py       # Data preprocessing
│   └── evaluate.py         # Evaluation utilities
├── train.py                # Main entry point
├── requirements.txt        # Python dependencies
├── setup.py                # Package setup
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting a pull request or opening an issue.

---

## 📜 License

This project is licensed under the **GNU General Public License v3.0** — see the [LICENSE](LICENSE) file for details.

---

## 📝 Citation

If you use this work in your research, please cite:

	arXiv:2601.17047

---

## 📬 Contact

For questions or suggestions, please contact: [guyj23@m.fudan.edu.cn](mailto:guyj23@m.fudan.edu.cn)
