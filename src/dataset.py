"""
Noise Genome Estimator - Dataset Module

Author: Yuanjie Gu @ Fudan
Date: 2024-11-05
Description: PyTorch Dataset classes for loading and augmenting image data.
    Supports training, validation, and testing datasets with noise addition.
"""

import os
import random
import numpy as np
import torch
import kornia as K
import kornia.augmentation as Ka
from skimage import io
from torch.utils.data import Dataset, DataLoader
import tifffile


class TensorNoiseAdder:
    """
    Utility class for adding various types of noise to image tensors.
    
    Supported noise types:
    - Gaussian noise
    - Salt and pepper noise
    - Poisson noise
    - Quantization noise
    - Anisotropic noise
    - Random combination of noise types
    """
    
    def __init__(self, image_tensor):
        """
        Initialize with an image tensor.
        
        Args:
            image_tensor (torch.Tensor): Image tensor, typically shape [B, C, H, W] or [C, H, W]
        """
        self.image_tensor = image_tensor
        self._tensor_checker(self.image_tensor)
        self.image_tensor = self._value_clamped(self.image_tensor)

    def _value_clamped(self, tensor):
        """Clamp tensor values to [0, 1]."""
        return torch.clamp(tensor, 0, 1)
    
    def _tensor_checker(self, tensor):
        """Verify that input is a PyTorch tensor."""
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("Input image must be a PyTorch tensor")

    @torch.no_grad()
    def add_gaussian_noise(self, strength=0.1, color_noise=True):
        """Add Gaussian noise to the image."""
        std = strength
        noise = torch.randn_like(
            self.image_tensor if color_noise else self.image_tensor[:, 0:1, :, :]
        ) * std
        if not color_noise:
            noise = noise.expand_as(self.image_tensor)
        noisy_image = self.image_tensor + noise
        return self._value_clamped(noisy_image)
    
    @torch.no_grad()
    def add_salt_and_pepper_noise(self, strength=0.1, color_noise=True):
        """Add salt and pepper noise to the image."""
        amount = strength * 0.25
        noise = torch.rand_like(
            self.image_tensor if color_noise else self.image_tensor[:, 0:1, :, :]
        )
        black_mask = noise < amount
        white_mask = noise > (1 - amount)
        noisy_image = self.image_tensor.clone()
        if not color_noise:
            black_mask = black_mask.expand_as(self.image_tensor)
            white_mask = white_mask.expand_as(self.image_tensor)
        noisy_image[black_mask] = 0
        noisy_image[white_mask] = 1
        return self._value_clamped(noisy_image)

    @torch.no_grad()
    def add_poisson_noise(self, strength=0.1, color_noise=True):
        """Add Poisson noise to the image."""
        strength = strength * 0.25
        if color_noise:
            noisy_image = torch.poisson(self.image_tensor * strength) + self.image_tensor
        else:
            noisy_image = torch.poisson(self.image_tensor[:, 0:1, :, :] * strength)
            noisy_image = noisy_image.expand_as(self.image_tensor) + self.image_tensor
        return self._value_clamped(noisy_image)

    @torch.no_grad()
    def add_quantization_noise(self, strength=0.1, color_noise=True):
        """Add quantization noise to the image."""
        levels = strength * 0.25
        noise = torch.rand_like(
            self.image_tensor if color_noise else self.image_tensor[:, 0:1, :, :]
        )
        if not color_noise:
            noise = noise.expand_as(self.image_tensor)
        noisy_image = torch.floor(self.image_tensor / levels + noise) * levels
        return self._value_clamped(noisy_image)

    @torch.no_grad()
    def add_anisotropic_noise(self, strength=0.1, color_noise=True):
        """Add anisotropic (directional) noise to the image."""
        std = strength * 0.25
        noise = torch.randn_like(
            self.image_tensor if color_noise else self.image_tensor[:, 0:1, :, :]
        ) * std
        if not color_noise:
            noise = noise.expand_as(self.image_tensor)
        
        # Apply Gaussian blur kernel
        kernel = torch.tensor(
            [[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], 
             [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]],
            dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)
        kernel = kernel / kernel.sum()
        kernel = kernel.expand(self.image_tensor.size(1), 1, 5, 5).to(self.image_tensor.device)
        
        noise = torch.nn.functional.conv2d(noise, kernel, padding=2, groups=self.image_tensor.size(1))
        noisy_image = self.image_tensor + noise
        return self._value_clamped(noisy_image)

    @torch.no_grad()
    def add_normal(self, **kwargs):
        """No noise (clean image)."""
        return self._value_clamped(self.image_tensor)
    
    @staticmethod
    def softmax(x):
        """Compute softmax for strength weighting."""
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)

    @torch.no_grad()
    def add_random_noise(self):
        """
        Add random combination of noise types.
        
        Returns:
            tuple: (noisy_image, noise_strength_params)
        """
        noise_functions = [
            self.add_gaussian_noise,
            self.add_salt_and_pepper_noise,
            self.add_poisson_noise,
            self.add_quantization_noise,
            self.add_anisotropic_noise,
            self.add_normal
        ]
        strength = self.softmax(np.random.normal(0, 1, len(noise_functions)))
        
        for idx, noise_fn in enumerate(noise_functions):
            self.image_tensor = noise_fn(strength=strength[idx], color_noise=True)
            self.image_tensor = self._value_clamped(self.image_tensor)
        
        return self.image_tensor, strength
    
    @torch.no_grad()
    def add_fix_noise(self, strength):
        """
        Add noise with fixed strength parameters.
        
        Args:
            strength (array-like): Noise strength for each noise type
            
        Returns:
            torch.Tensor: Noisy image
        """
        noise_functions = [
            self.add_gaussian_noise,
            self.add_salt_and_pepper_noise,
            self.add_poisson_noise,
            self.add_quantization_noise,
            self.add_anisotropic_noise,
            self.add_normal
        ]
        
        for idx, noise_fn in enumerate(noise_functions):
            self.image_tensor = noise_fn(strength=strength[idx], color_noise=True)
            self.image_tensor = self._value_clamped(self.image_tensor)
        
        return self.image_tensor


class TrainDataset(Dataset):
    """
    Training dataset with data augmentation and noise injection.
    
    Features:
    - Random crops
    - Geometric augmentations (flip, rotation)
    - Multiple noise types with contrastive learning
    """
    
    SUPPORTED_FORMATS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff')

    def __init__(self, image_dir, patch_size, if_crop_and_resize):
        """
        Initialize training dataset.
        
        Args:
            image_dir (str): Root directory containing images
            patch_size (int): Size of image patches
            if_crop_and_resize (bool): Whether to resize instead of crop
        """
        self.image_dir = image_dir
        self.file_list = []
        
        for root, _, files in os.walk(self.image_dir):
            for file in files:
                if file.lower().endswith(self.SUPPORTED_FORMATS):
                    self.file_list.append(os.path.join(root, file))
        
        # Define Kornia augmentations
        self.transformsK = Ka.AugmentationSequential(
            Ka.RandomHorizontalFlip(p=0.5),
            Ka.RandomVerticalFlip(p=0.5),
            Ka.RandomRotation(degrees=90, p=0.5),
            data_keys=["input"]
        )
        self.to_gray = Ka.AugmentationSequential(Ka.RandomGrayscale(p=1.0))
        
        if if_crop_and_resize:
            self.cropperK = Ka.Resize((patch_size, patch_size), p=1.)
        else:
            self.cropperK = Ka.RandomCrop((patch_size, patch_size), p=1., cropping_mode="resample")

    def _image2tensor2norm2aug(self, image):
        """Convert image to normalized tensor."""
        img_tensor = K.image_to_tensor(image, keepdim=False).float()
        img_tensor = K.enhance.normalize_min_max(img_tensor, 0., 1.)
        return img_tensor

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Load two different images
        idx_contrastive = random.randint(0, len(self.file_list) - 1)
        while idx_contrastive == idx:
            idx_contrastive = random.randint(0, len(self.file_list) - 1)
        
        image = io.imread(self.file_list[idx])
        image_contrastive = io.imread(self.file_list[idx_contrastive])
        
        # Handle RGBA images
        if image.shape[-1] == 4:
            image = image[..., :3]
        if image_contrastive.shape[-1] == 4:
            image_contrastive = image_contrastive[..., :3]
        
        # Handle grayscale images
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1).repeat(3, axis=-1)
        if len(image_contrastive.shape) == 2:
            image_contrastive = np.expand_dims(image_contrastive, axis=-1).repeat(3, axis=-1)
        
        # Convert to tensor and augment
        target = self._image2tensor2norm2aug(image)
        target = self.transformsK(self.cropperK(target))
        target_contrastive = self._image2tensor2norm2aug(image_contrastive)
        target_contrastive = self.transformsK(self.cropperK(target_contrastive))
        
        # Add noise with contrastive learning
        input_org, strength_org = TensorNoiseAdder(target).add_random_noise()
        input_contrastive_org = TensorNoiseAdder(target_contrastive).add_fix_noise(strength_org)
        input_con, strength_con = TensorNoiseAdder(target).add_random_noise()
        
        # Random grayscale conversion
        if np.random.rand(1) < 0.5:
            input_org = self.to_gray(input_org)
            input_con = self.to_gray(input_con)
            input_contrastive_org = self.to_gray(input_contrastive_org)
        
        return {
            'input_org': input_org.squeeze(0),
            'input_con': input_con.squeeze(0),
            'input_contrastive_org': input_contrastive_org.squeeze(0),
            'noise_params_org': torch.from_numpy(strength_org).type_as(input_org),
            'noise_params_con': torch.from_numpy(strength_con).type_as(input_con)
        }


class ValidationDataset(Dataset):
    """Validation dataset with single image loading."""
    
    SUPPORTED_FORMATS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff')
    
    def __init__(self, image_dir, patch_size, if_crop_and_resize):
        """
        Initialize validation dataset.
        
        Args:
            image_dir (str): Root directory containing images
            patch_size (int): Size of image patches
            if_crop_and_resize (bool): Whether to resize instead of crop
        """
        self.image_dir = image_dir
        self.file_list = []
        
        for root, _, files in os.walk(self.image_dir):
            for file in files:
                if file.lower().endswith(self.SUPPORTED_FORMATS):
                    self.file_list.append(os.path.join(root, file))
        
        if if_crop_and_resize:
            self.cropperK = Ka.Resize((patch_size, patch_size), p=1.)
        else:
            self.cropperK = Ka.RandomCrop((patch_size, patch_size), p=1., cropping_mode="resample")

    def _image2tensor2norm2aug(self, image):
        """Convert image to normalized tensor."""
        img_tensor = K.image_to_tensor(image, keepdim=False).float()
        img_tensor = K.enhance.normalize_min_max(img_tensor, 0., 1.)
        return img_tensor

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        image_path = self.file_list[idx]
        image = io.imread(image_path)
        
        if image.shape[-1] == 4:
            image = image[..., :3]
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1).repeat(3, axis=-1)
        
        image_tensor = self.cropperK(self._image2tensor2norm2aug(image))
        input_org, strength_org = TensorNoiseAdder(image_tensor).add_random_noise()
        
        return {
            'input_org': input_org.squeeze(0),
            'noise_params_org': torch.from_numpy(strength_org).type_as(input_org)
        }


class TestDataset(Dataset):
    """Test dataset for model evaluation."""
    
    SUPPORTED_FORMATS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff')
    
    def __init__(self, image_dir, param_dir=None):
        """
        Initialize test dataset.
        
        Args:
            image_dir (str): Directory containing test images
            param_dir (str, optional): Directory containing noise parameters.
                If provided and matching .npz files exist, ground-truth noise
                parameters will be loaded for metric computation.
        """
        self.image_dir = image_dir
        self.param_dir = param_dir
        self.file_list = []
        
        for root, _, files in os.walk(self.image_dir):
            for file in files:
                if file.lower().endswith(self.SUPPORTED_FORMATS):
                    self.file_list.append(os.path.join(root, file))

    def __len__(self):
        return len(self.file_list)
    
    def _image2tensor2norm2aug(self, image):
        """Convert image to normalized tensor."""
        img_tensor = K.image_to_tensor(image, keepdim=False).float()
        img_tensor = K.enhance.normalize_min_max(img_tensor, 0., 1.)
        return img_tensor
    
    def __getitem__(self, idx):
        image_path = self.file_list[idx]
        image = io.imread(image_path)
        
        # Ensure 3-channel RGB
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        
        # Normalize to [0,1] and convert to tensor
        image = self._image2tensor2norm2aug(image)
        image_tensor = image.squeeze(0)
        
        # Load noise params if provided
        noise_params = torch.zeros(6, dtype=torch.float32)  # placeholder
        if self.param_dir is not None:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            param_path = os.path.join(self.param_dir, base_name + '.npz')
            if os.path.exists(param_path):
                npz = np.load(param_path)
                noise_params = torch.from_numpy(npz['noise_strength']).float()
        
        return {
            'image': image_tensor,
            'file_name': os.path.basename(image_path),
            'noise_params': noise_params
        }


class TiffDataset(torch.utils.data.Dataset):
    """
    Dataset for reading 3D TIFF data (x, y, t).
    
    Features:
    - Supports time-series imaging data
    - Per-frame normalization
    - Automatic padding for small images
    """
    
    def __init__(self, tiff_path, frame_axis=0, norm=True):
        """
        Initialize TIFF dataset.
        
        Args:
            tiff_path (str): Path to TIFF file
            frame_axis (int): Axis along which frames are arranged
            norm (bool): Whether to normalize data
        """
        self.tiff_path = tiff_path
        self.data = io.imread(tiff_path)
        
        if frame_axis != 0:
            self.data = self.data.swapaxes(0, frame_axis)
        
        self.length = self.data.shape[0]
        self.norm = norm
        
        # Store original spatial shapes
        self.original_spatial_shapes = []
        for t in range(self.length):
            if self.data[t].ndim == 2:
                h, w = self.data[t].shape
            elif self.data[t].ndim == 3:
                h, w, _ = self.data[t].shape
            else:
                raise ValueError(f"Unsupported shape at frame {t}")
            self.original_spatial_shapes.append((h, w))
        
        # Compute per-frame min/max for normalization
        if self.norm:
            self._min = []
            self._max = []
            for t in range(self.length):
                if self.data[t].ndim == 3:
                    min_vals = [self.data[t, ..., c].min() for c in range(self.data[t].shape[-1])]
                    max_vals = [self.data[t, ..., c].max() for c in range(self.data[t].shape[-1])]
                elif self.data[t].ndim == 2:
                    min_vals = [self.data[t].min()]
                    max_vals = [self.data[t].max()]
                else:
                    raise ValueError(f"Unsupported shape at frame {t}")
                self._min.append(min_vals)
                self._max.append(max_vals)
        else:
            self._min = None
            self._max = None

    def __len__(self):
        return self.length
    
    def _image2tensor2norm2aug(self, image):
        """Convert image to normalized tensor."""
        img_tensor = K.image_to_tensor(image, keepdim=False).float()
        img_tensor = K.enhance.normalize_min_max(img_tensor, 0., 1.)
        return img_tensor

    def __getitem__(self, idx):
        img = self.data[idx]
        img_tensor = self._image2tensor2norm2aug(img.astype(np.float32))
        return img_tensor.squeeze(0)


def random_square_crop(tensor, crop_size):
    """
    Randomly crop square patches from batch of images.
    
    Args:
        tensor (torch.Tensor): Input tensor [B, C, H, W]
        crop_size (int): Size of square crop
        
    Returns:
        torch.Tensor: Cropped tensor [B, C, crop_size, crop_size]
    """
    B, C, H, W = tensor.shape
    if crop_size > H or crop_size > W:
        raise ValueError(f"crop_size {crop_size} exceeds input size {H}x{W}")
    
    top = torch.randint(0, H - crop_size + 1, (B,))
    left = torch.randint(0, W - crop_size + 1, (B,))
    
    crops = []
    for i in range(B):
        crops.append(tensor[i:i+1, :, top[i]:top[i]+crop_size, left[i]:left[i]+crop_size])
    
    return torch.cat(crops, dim=0)


def remove_padding(img_tensor, original_spatial_shape):
    """
    Remove padding from image tensor.
    
    Args:
        img_tensor (torch.Tensor): Tensor with padding [C, H, W] or [H, W]
        original_spatial_shape (tuple): Original shape (h, w)
        
    Returns:
        torch.Tensor: Unpadded tensor
    """
    h, w = original_spatial_shape
    if img_tensor.ndim == 3:
        return img_tensor[:, :h, :w]
    elif img_tensor.ndim == 2:
        return img_tensor[:h, :w]
    else:
        raise ValueError("Unsupported tensor dimension. Expected 2D or 3D tensor.")


if __name__ == "__main__":
    # Example usage
    image_dir = './datasets/test'
    dataset = TrainDataset(image_dir=image_dir, patch_size=192, if_crop_and_resize=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    for batch in dataloader:
        print(f"Batch keys: {batch.keys()}")
        print(f"Input shape: {batch['input_org'].shape}")
        break
