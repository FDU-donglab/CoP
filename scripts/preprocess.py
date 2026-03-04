#!/usr/bin/env python3
"""
Data preprocessing utilities for Noise Genome Estimator.

Usage:
    python scripts/preprocess.py --input-dir ./raw_data --output-dir ./datasets/train
"""

import argparse
import os
from pathlib import Path
import shutil
import logging
from PIL import Image
import numpy as np
from skimage import io, img_as_ubyte

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_image(image_path, min_size=64):
    """
    Validate if an image is suitable for training.
    
    Args:
        image_path (str): Path to image file
        min_size (int): Minimum image dimension
        
    Returns:
        bool: True if image is valid
    """
    try:
        img = Image.open(image_path)
        if img.size[0] < min_size or img.size[1] < min_size:
            logger.warning(f"Image {image_path} is too small: {img.size}")
            return False
        return True
    except Exception as e:
        logger.error(f"Error reading {image_path}: {e}")
        return False


def convert_image_format(image_path, output_path, target_format='PNG'):
    """
    Convert image to standard format and size.
    
    Args:
        image_path (str): Input image path
        output_path (str): Output image path
        target_format (str): Target image format (PNG, JPEG, etc.)
    """
    try:
        img = Image.open(image_path)
        
        # Convert to RGB if necessary
        if img.mode in ('RGBA', 'LA', 'P'):
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = rgb_img
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Save in target format
        img.save(output_path, format=target_format, quality=95)
        logger.info(f"Converted {image_path} -> {output_path}")
        
    except Exception as e:
        logger.error(f"Error converting {image_path}: {e}")


def organize_dataset(input_dir, output_dir, train_ratio=0.8, val_ratio=0.1):
    """
    Organize images into train/val/test splits.
    
    Args:
        input_dir (str): Input directory with images
        output_dir (str): Output directory for organized splits
        train_ratio (float): Fraction for training
        val_ratio (float): Fraction for validation
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create split directories
    splits = ['train', 'val', 'test']
    for split in splits:
        (output_path / split).mkdir(parents=True, exist_ok=True)
    
    # Get all valid images
    valid_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    all_images = []
    
    for ext in valid_formats:
        all_images.extend(list(input_path.glob(f'**/*{ext}')))
        all_images.extend(list(input_path.glob(f'**/*{ext.upper()}')))
    
    # Shuffle and split
    all_images = list(set(all_images))
    np.random.shuffle(all_images)
    
    n_total = len(all_images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_images = all_images[:n_train]
    val_images = all_images[n_train:n_train+n_val]
    test_images = all_images[n_train+n_val:]
    
    logger.info(f"Total images: {n_total}")
    logger.info(f"Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")
    
    # Copy/convert images
    for split, images in [('train', train_images), ('val', val_images), ('test', test_images)]:
        for img_path in images:
            if validate_image(img_path):
                output_file = output_path / split / img_path.name
                if img_path.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
                    shutil.copy(img_path, output_file)
                else:
                    convert_image_format(str(img_path), str(output_file))


def main():
    parser = argparse.ArgumentParser(description="Preprocess data for Noise Genome Estimator")
    parser.add_argument('--input-dir', type=str, required=True, help='Input directory with raw images')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for processed data')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Training data ratio')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='Validation data ratio')
    parser.add_argument('--min-size', type=int, default=64, help='Minimum image size')
    
    args = parser.parse_args()
    
    logger.info(f"Starting data preprocessing...")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    organize_dataset(
        args.input_dir,
        args.output_dir,
        args.train_ratio,
        args.val_ratio
    )
    
    logger.info(f"Data preprocessing completed!")


if __name__ == '__main__':
    main()
