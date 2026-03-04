"""
Noise Genome Estimator - Main Training and Testing Script

Usage:
    python train.py [options]
    python train.py --mode test --checkpoint-load-path <path>
"""

import os
import argparse
from src.engines import ContrastiveTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and Test Noise Genome Estimator"
    )
    
    # Mode selection
    parser.add_argument(
        '--mode', type=str, default='train', choices=['train', 'test'],
        help='Run mode: train or test'
    )
    
    # Distributed training
    parser.add_argument(
        '--local-rank', type=int,
        default=int(os.environ.get('LOCAL_RANK', 0)),
        help='Local rank for DistributedDataParallel'
    )
    parser.add_argument(
        '--rank', type=int,
        default=int(os.environ.get('RANK', 0)),
        help='Global rank for DistributedDataParallel'
    )
    parser.add_argument(
        '--world-size', type=int,
        default=int(os.environ.get('WORLD_SIZE', 1)),
        help='World size for DistributedDataParallel'
    )
    
    # Training configuration
    parser.add_argument(
        '--num-epochs', type=int, default=300,
        help="Number of epochs for training"
    )
    parser.add_argument(
        '--batch-size', type=int, default=16,
        help="Batch size for training"
    )
    parser.add_argument(
        '--max-device-batch-size', type=int, default=16,
        help="Maximum batch size per device"
    )
    parser.add_argument(
        '--number-works', type=int, default=16,
        help="Number of workers for data loading"
    )
    parser.add_argument(
        '--learning-rate', type=float, default=1.5e-4,
        help="Learning rate for training"
    )
    parser.add_argument(
        '--warmup-epoch', type=int, default=30,
        help="Number of warmup epochs for learning rate scheduling"
    )
    parser.add_argument(
        '--lr-gamma', type=float, default=0.5,
        help="Learning rate decay factor"
    )
    
    # Checkpoint configuration
    parser.add_argument(
        '--checkpoint-save-path', type=str, default='./checkpoints',
        help="Path to save checkpoints"
    )
    parser.add_argument(
        '--checkpoint-flash-frequency', type=int, default=5,
        help="Interval (in epochs) for saving checkpoints"
    )
    parser.add_argument(
        '--checkpoint-load-path', type=str,
        default="./checkpoint/checkpoints/imnt_woCL/model_epoch_300.pth",
        help="Path to load checkpoint for training"
    )
    parser.add_argument(
        '--save-optimizer', type=bool, default=True,
        help="Whether to save optimizer state in checkpoints"
    )
    
    # Model configuration
    parser.add_argument(
        '--seed', type=int, default=3407,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        '--model-type', type=str, default='vit', choices=['vit', 'swin'],
        help="Model type ('vit' or 'swin')"
    )
    parser.add_argument(
        '--crop-size-whole-xy', type=int, default=192,
        help="Crop size for training images"
    )
    parser.add_argument(
        '--if-crop-and-resize', type=bool, default=False,
        help="Whether to crop and resize training images"
    )
    parser.add_argument(
        '--patch-size-in-tr', type=int, default=16,
        help="Transformer embedding patch size"
    )
    parser.add_argument(
        '--in-out-channels', type=int, default=3,
        help="Number of image channels (1 for gray, 3 for RGB)"
    )
    
    # Dataset paths
    parser.add_argument(
        '--train-dataset-path', type=str, default="./datasets/train",
        help="Path to the training dataset"
    )
    parser.add_argument(
        '--validation-dataset-path', type=str,
        default="/data/root/yuanjie/20250804",
        help="Path to the validation dataset"
    )
    parser.add_argument(
        '--test-image-path', type=str,
        default="/data/root/yuanjie/20250804/",
        help="Path to the testing dataset"
    )
    parser.add_argument(
        '--test-param-path', type=str, default="./noise_params",
        help="Path to the testing noise parameters"
    )
    
    # Visualization
    parser.add_argument(
        '--if-train-visdom-visialize', type=bool, default=True,
        help="Whether to use TensorBoard for training visualization"
    )
    parser.add_argument(
        '--visdom-flash-frequency', type=int, default=100,
        help="Interval (in iterations) for visualization updates"
    )
    
    # Device
    parser.add_argument(
        '--gpu', type=int, default=1,
        help="GPU id to use"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create trainer
    trainer = ContrastiveTrainer(args=args)
    
    # Run appropriate mode
    if args.mode == 'train':
        trainer.train(args)
    elif args.mode == 'test':
        trainer.test(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == '__main__':
    main()
