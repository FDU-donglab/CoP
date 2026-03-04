# Noise Genome Estimator

A deep learning model for estimating noise parameters in images using contrastive learning and Vision Transformers.

## Overview

This project implements a noise estimation system that can characterize various types of image noise (Gaussian, salt-and-pepper, Poisson, quantization, and anisotropic noise) in a unified framework. The model uses a Vision Transformer (ViT) backbone with contrastive learning to learn robust noise representations.

## Key Features

- **Multiple Noise Types**: Supports Gaussian, salt-and-pepper, Poisson, quantization, and anisotropic noise
- **Contrastive Learning**: Uses noise-independent contrastive loss to learn discriminative features
- **Scalable Architecture**: Built with Vision Transformer (ViT) and Swin Transformer support
- **Distributed Training**: Full support for multi-GPU training with PyTorch DistributedDataParallel
- **Mixed Precision**: Automatic mixed precision training for improved performance
- **Comprehensive Evaluation**: Includes metrics like MAE, MSE, RMSE, and R²

## Installation

### Requirements

- Python 3.8+
- CUDA 11.0+ (for GPU training)
- PyTorch 2.0+

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Noise_Genome_Estimator.git
cd Noise_Genome_Estimator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create necessary directories:
```bash
mkdir -p datasets/train datasets/val datasets/test checkpoints noise_params
```

## Dataset Structure

Organize your datasets in the following structure:

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

## Usage

### Training

```bash
# Single GPU training
python train.py --mode train \
    --train-dataset-path ./datasets/train \
    --validation-dataset-path ./datasets/val \
    --num-epochs 300 \
    --batch-size 16

# Multi-GPU training with DistributedDataParallel
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

## Configuration

All parameters can be configured via command line arguments. See `train.py --help` for a complete list.

Key configuration options:

- `--model-type`: Choose between 'vit' (Vision Transformer) or 'swin' (Swin Transformer)
- `--batch-size`: Training batch size
- `--learning-rate`: Initial learning rate
- `--crop-size-whole-xy`: Input image patch size (default: 192)
- `--patch-size-in-tr`: Transformer patch embedding size (default: 16)

## Model Architecture

### Backbone Options

1. **Vision Transformer (ViT)**: Standard ViT with patch embeddings
2. **Swin Transformer**: Hierarchical transformer with shifting windows

### Loss Function

The model uses a combined loss function:

$$L = \alpha \cdot L_{contrast} + \beta \cdot L_{noise}$$

Where:
- $L_{contrast}$: Contrastive loss (InfoNCE-style)
- $L_{noise}$: MSE loss for noise parameter regression
- $\alpha, \beta$: Loss weights (default: 0.5 each)

## Project Structure

```
Noise_Genome_Estimator/
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
├── docs/
│   └── ...
├── train.py                # Main training script
├── requirements.txt        # Python dependencies
├── setup.py               # Package setup
├── .gitignore
├── LICENSE
└── README.md
```


## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@article{gu2025noisegenome,
  title={Noise Genome Estimator: Learning Unified Noise Representations},
  author={Gu, Yuanjie and others},
  journal={...},
  year={2025}
}
```

## Acknowledgments

- Thanks to the PyTorch team for the excellent deep learning framework
- Inspired by recent work in image quality assessment and noise modeling
- Built upon timm (PyTorch Image Models) for transformer implementations

## Contact

For questions or suggestions, please contact: yuanjie.gu@fudan.edu.cn
