"""
Noise Genome Estimator - Training and Testing Engine

Author: Yuanjie Gu @ Fudan
Date: 2024-09-06
Description: Training and testing pipelines for the noise estimation model with contrastive learning.
"""

import torch
import gc
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import TrainDataset, ValidationDataset, TestDataset
from src.models.masked_autoencoder import build_noiser
from tqdm import tqdm
from src.utils import (
    init_distributed_mode, setup_seed, config_to_json, 
    model_summary, load_checkpoint, load_checkpoint_only
)
import os
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import math
import warnings

warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"


class NoiseIndependentContrastiveLoss(nn.Module):
    """
    Combined contrastive loss and noise prediction loss.
    
    Balances contrastive learning (comparing similar vs. different noise)
    with direct noise parameter regression.
    """
    
    def __init__(self, temperature=0.1, alpha=0.5, beta=0.5):
        """
        Initialize loss function.
        
        Args:
            temperature (float): Temperature scaling for contrastive loss
            alpha (float): Weight for contrastive loss
            beta (float): Weight for noise prediction loss
        """
        super(NoiseIndependentContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.mse_loss = nn.MSELoss()

    def forward(self, features, outputs, labels, batch_size):
        """
        Compute combined loss.
        
        Args:
            features (torch.Tensor): Extracted features [3B, D]
            outputs (torch.Tensor): Predicted noise params [3B, 6]
            labels (torch.Tensor): Ground truth noise params [3B, 6]
            batch_size (int): Original batch size
            
        Returns:
            dict: Loss components and total loss
        """
        # Noise prediction loss
        noise_loss = self.mse_loss(outputs, labels)
        
        # Normalize features for contrastive loss
        features = nn.functional.normalize(features, dim=1)
        f_org_norm = features[:batch_size]
        f_con_norm = features[batch_size:2*batch_size]
        f_contrast_norm = features[2*batch_size:]

        # Contrastive loss: same noise should have high similarity
        pos_sim = torch.sum(f_org_norm * f_contrast_norm, dim=-1)
        pos_sim = torch.exp(pos_sim / self.temperature)
        
        # Different noises should have low similarity
        neg_sim = torch.mm(f_org_norm, f_con_norm.T)
        neg_sim = torch.exp(neg_sim / self.temperature)

        # Mask out self-similarity
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=features.device)
        neg_sim = neg_sim * mask.float()

        # Contrastive loss
        contrast_loss = -torch.log(
            pos_sim / (pos_sim + torch.sum(neg_sim, dim=1) + 1e-6)
        ).mean()

        # Combined loss
        total_loss = self.alpha * contrast_loss + self.beta * noise_loss

        return {
            "total_loss": total_loss,
            "contrast_loss": contrast_loss,
            "noise_loss": noise_loss
        }


class ContrastiveTrainer:
    """
    Trainer for noise genome estimator with contrastive learning.
    
    Handles distributed training, validation, checkpointing, and testing.
    """
    
    def __init__(self, args):
        """
        Initialize trainer.
        
        Args:
            args: Configuration arguments
        """
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        self.checkpoint_folder = os.path.join(
            args.checkpoint_save_path, 
            f"{self.timestamp}"
        )
        
        if args.checkpoint_load_path == "None":
            args.checkpoint_load_path = None
        
        os.makedirs(self.checkpoint_folder, exist_ok=True)
        config_to_json(
            vars(args), 
            output_dir=self.checkpoint_folder, 
            stage_name=f"{self.timestamp}"
        )

    def train(self, args):
        """
        Train the noise estimator model.
        
        Args:
            args: Training configuration
        """
        print(f">>>>>>>>>> Training started at {self.timestamp} ..........")
        
        # Initialize distributed training
        init_distributed_mode(args)
        setup_seed(args.seed)
        print(f"{'START TRAINING':^50}")
        
        # Initialize TensorBoard
        writer = None
        if args.if_train_visdom_visialize:
            writer = SummaryWriter(log_dir=self.checkpoint_folder)
        
        # Build model
        model = build_noiser(
            model_type=args.model_type,
            img_size=args.crop_size_whole_xy,
            patch_size=args.patch_size_in_tr,
            in_chans=args.in_out_channels
        )
        loss_fn = NoiseIndependentContrastiveLoss(
            temperature=0.1, alpha=0.5, beta=0.5
        )

        # Initialize datasets
        train_dataset = TrainDataset(
            image_dir=args.train_dataset_path,
            patch_size=args.crop_size_whole_xy,
            if_crop_and_resize=args.if_crop_and_resize
        )
        val_dataset = ValidationDataset(
            image_dir=args.validation_dataset_path,
            patch_size=args.crop_size_whole_xy,
            if_crop_and_resize=args.if_crop_and_resize
        )

        # Create distributed samplers
        train_sampler = torch.utils.data.DistributedSampler(
            train_dataset, shuffle=True
        )
        val_sampler = torch.utils.data.DistributedSampler(
            val_dataset, shuffle=False
        )

        # Create data loaders
        train_dataloader = DataLoader(
            dataset=train_dataset,
            sampler=train_sampler,
            batch_size=args.batch_size,
            num_workers=args.number_works,
            drop_last=True
        )
        val_dataloader = DataLoader(
            dataset=val_dataset,
            sampler=val_sampler,
            batch_size=args.batch_size,
            num_workers=args.number_works,
            drop_last=True
        )
        
        # Setup optimizer and scheduler
        optimizer = optim.AdamW(
            params=model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.05
        )
        
        lr_func = lambda epoch: min(
            (epoch + 1) / (args.warmup_epoch + 1e-8),
            0.5 * (math.cos(epoch / args.num_epochs * math.pi) + 1)
        )
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_func, verbose=True
        )

        # Load checkpoint if provided
        if args.checkpoint_load_path is not None:
            load_checkpoint(args.checkpoint_load_path, model, optimizer, lr_scheduler)

        # Setup device and DDP
        model = model.to(args.rank)
        model_summary(model)
        print("Using Distributed Data Parallel (DDP)...")
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], output_device=args.gpu
        )
        
        # Setup mixed precision training
        scaler = torch.cuda.amp.GradScaler()
        assert args.batch_size % min(args.max_device_batch_size, args.batch_size) == 0
        steps_per_update = args.batch_size // min(args.max_device_batch_size, args.batch_size)
        
        step_count = 0
        loss_contrast_log = []
        loss_noise_log = []
        best_val = float('inf')

        # Training loop
        for epoch in range(args.num_epochs):
            train_dataloader.sampler.set_epoch(epoch)
            val_dataloader.sampler.set_epoch(epoch)
            
            progress_bar = tqdm(
                train_dataloader,
                desc=f'Epoch {epoch + 1}/{args.num_epochs}',
                unit='batch',
                ncols=150
            )
            
            loss_total_val = [0]
            for idx, terms in enumerate(progress_bar):
                step_count += 1
                
                # Prepare batch data
                input_org = terms['input_org'].to(args.rank)
                input_con = terms['input_con'].to(args.rank)
                input_contrastive_org = terms['input_contrastive_org'].to(args.rank)
                noise_params_org = terms['noise_params_org'].to(args.rank)
                noise_params_con = terms['noise_params_con'].to(args.rank)
                
                # Concatenate for contrastive learning
                inputs = torch.cat((input_org, input_con, input_contrastive_org), dim=0)
                labels = torch.cat((noise_params_org, noise_params_con, noise_params_org), dim=0)
                
                # Forward pass with mixed precision
                with torch.cuda.amp.autocast():
                    features, outputs = model(inputs)
                    losses = loss_fn(features, outputs, labels, args.batch_size)
                
                loss_contrast_log.append(losses["contrast_loss"].item())
                loss_noise_log.append(losses["noise_loss"].item())
                loss_total = losses["total_loss"]
                
                # Backward pass
                if step_count % steps_per_update == 0:
                    optimizer.zero_grad()
                    scaler.scale(loss_total).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20.0)
                    scaler.step(optimizer)
                    scaler.update()
                
                loss_total_val.append(loss_total.item())
                progress_bar.set_postfix(
                    loss_total=np.mean(loss_total_val),
                    refresh=True
                )
                
                # Log to TensorBoard
                if writer and idx % args.visdom_flash_frequency == 0:
                    writer.add_scalar(
                        'Training/Contrastive Loss',
                        losses["contrast_loss"].item(),
                        epoch * len(train_dataloader) + idx
                    )
                    writer.add_scalar(
                        'Training/Noise Loss',
                        losses["noise_loss"].item(),
                        epoch * len(train_dataloader) + idx
                    )
            
            lr_scheduler.step()
            gc.collect()
            torch.cuda.empty_cache()

            # Validation and checkpoint saving
            if (epoch + 1) % args.checkpoint_flash_frequency == 0 or epoch == 0:
                model.eval()
                validation_loss = []
                
                with torch.no_grad():
                    for terms in val_dataloader:
                        inputs = terms['input_org'].to(args.rank)
                        labels = terms['noise_params_org'].to(args.rank)
                        features, outputs = model(inputs)
                        val_losses = torch.abs(outputs - labels)
                        validation_loss.append(val_losses.mean().item())
                
                avg_validation_loss = np.mean(validation_loss)
                
                if writer:
                    writer.add_scalar(
                        'Validation/Noise Loss',
                        avg_validation_loss,
                        epoch
                    )
                
                print(f"\nValidation Loss at epoch {epoch + 1}: {avg_validation_loss:.6f}")
                
                model.train()
                
                # Save best model
                if avg_validation_loss < best_val:
                    print("Saving best model checkpoint...")
                    state_dict = (
                        model.module.state_dict()
                        if isinstance(model, nn.DataParallel)
                        else model.state_dict()
                    )
                    
                    if args.save_optimizer:
                        checkpoint = {
                            'config': "vit_noiser",
                            'epoch': epoch + 1,
                            'model': state_dict,
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                        }
                    else:
                        checkpoint = state_dict
                    
                    checkpoint_path = (
                        f"{self.checkpoint_folder}/model_epoch_{epoch + 1}.pth"
                    )
                    torch.save(checkpoint, checkpoint_path)
                    
                    loss_log_path = (
                        f"{self.checkpoint_folder}/loss_log_epoch_{epoch + 1}.npz"
                    )
                    np.savez(
                        loss_log_path,
                        contrast_loss=loss_contrast_log,
                        noise_loss=loss_noise_log
                    )
                    
                    print(f"Checkpoint saved to {checkpoint_path}")
                    best_val = avg_validation_loss

    def test(self, args):
        """
        Test the noise estimator model.
        
        Args:
            args: Testing configuration
        """
        print(f">>>>>>>>>> Testing started at {self.timestamp} ..........")
        setup_seed(args.seed)
        
        device = torch.device(
            f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
        )
        torch.cuda.set_device(device)

        # Build model
        model = build_noiser(
            model_type=args.model_type,
            img_size=args.crop_size_whole_xy,
            patch_size=args.patch_size_in_tr,
            in_chans=args.in_out_channels
        ).to(device)

        # Load checkpoint
        if args.checkpoint_load_path is not None:
            load_checkpoint_only(args.checkpoint_load_path, model)
        else:
            print("Warning: No checkpoint provided. Using random initialization.")
        
        model.eval()
        model_summary(model)
        print(f"Using device: {device}")

        # Create test dataset
        test_dataset = TestDataset(args.test_image_path, args.test_param_path)
        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.number_works,
            drop_last=False
        )

        # Testing loop
        all_outputs = []
        all_gt_noises = []
        all_features = []
        all_loss = []
        
        with torch.no_grad():
            for idx, terms in enumerate(
                tqdm(test_dataloader, desc="Testing", ncols=120)
            ):
                inputs = terms['image'].to(device)
                gt_noise = terms['noise_params'].to(device)
                
                features, outputs = model(inputs)
                all_outputs.append(outputs.detach().cpu().numpy())
                all_gt_noises.append(gt_noise.detach().cpu().numpy())
                all_features.append(features.detach().cpu().numpy())
                
                loss_val = torch.abs(outputs - gt_noise).mean().item()
                all_loss.append(loss_val)

        # Compute metrics
        outputs_np = np.concatenate(all_outputs, axis=0)
        gt_np = np.concatenate(all_gt_noises, axis=0)
        
        mae = np.mean(np.abs(outputs_np - gt_np))
        mse = np.mean((outputs_np - gt_np) ** 2)
        rmse = np.sqrt(mse)
        
        # R² Score
        ss_res = np.sum((outputs_np - gt_np) ** 2)
        ss_tot = np.sum((gt_np - np.mean(gt_np)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        
        print(f"\n{'Testing Results':^50}")
        print(f"MAE:  {mae:.6f}")
        print(f"MSE:  {mse:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"R²:   {r2:.6f}")

        # Save results
        results_path = os.path.join(
            self.checkpoint_folder,
            "test_outputs_and_gt_noise.npz"
        )
        np.savez(
            results_path,
            outputs=outputs_np,
            gt_noise=gt_np,
            features=np.concatenate(all_features, axis=0)
        )
        print(f"\nResults saved to {results_path}")
