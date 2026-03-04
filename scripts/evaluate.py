#!/usr/bin/env python3
"""
Simple evaluation script for testing a trained model.

Usage:
    python scripts/evaluate.py \
        --checkpoint ./checkpoints/model.pth \
        --data-dir ./datasets/test \
        --output-dir ./results
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from src.engines import ContrastiveTrainer
from src.dataset import TestDataset
from torch.utils.data import DataLoader


def evaluate_model(checkpoint_path, data_dir, output_dir, batch_size=16, gpu=0):
    """
    Evaluate a trained model on test data.
    
    Args:
        checkpoint_path (str): Path to model checkpoint
        data_dir (str): Directory containing test images
        output_dir (str): Directory to save evaluation results
        batch_size (int): Batch size for evaluation
        gpu (int): GPU device ID
    """
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create trainer with dummy args for model initialization
    class Args:
        model_type = 'vit'
        crop_size_whole_xy = 192
        patch_size_in_tr = 16
        in_out_channels = 3
    
    args = Args()
    
    from src.models.masked_autoencoder import build_noiser
    from src.utils import load_checkpoint_only
    
    # Build and load model
    model = build_noiser(
        model_type=args.model_type,
        img_size=args.crop_size_whole_xy,
        patch_size=args.patch_size_in_tr,
        in_chans=args.in_out_channels
    ).to(device)
    
    load_checkpoint_only(checkpoint_path, model)
    model.eval()
    
    # Load test dataset
    test_dataset = TestDataset(data_dir, param_dir=None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    print(f"Evaluating on {len(test_dataset)} images...")
    
    all_outputs = []
    all_filenames = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            filenames = batch['file_name']
            
            features, outputs = model(images)
            all_outputs.append(outputs.cpu().numpy())
            all_filenames.extend(filenames)
    
    # Concatenate results
    outputs_np = np.concatenate(all_outputs, axis=0)
    
    # Save results
    results_file = output_dir / "noise_estimates.npz"
    np.savez(results_file, outputs=outputs_np, filenames=all_filenames)
    print(f"Results saved to {results_file}")
    
    # Print summary
    print(f"\nEstimation Summary:")
    print(f"  Mean noise strength: {outputs_np.mean(axis=0)}")
    print(f"  Std noise strength:  {outputs_np.std(axis=0)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate noise estimator model")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory containing test images')
    parser.add_argument('--output-dir', type=str, default='./results', help='Output directory for results')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    
    args = parser.parse_args()
    
    evaluate_model(
        args.checkpoint,
        args.data_dir,
        args.output_dir,
        args.batch_size,
        args.gpu
    )
