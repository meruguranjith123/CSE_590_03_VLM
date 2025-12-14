"""
Training script for Frequency Prior Codebook and Pixel Codebook model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
from pathlib import Path
from tqdm import tqdm

from models.vq_codebook_model import create_vq_codebook_model
from datasets.dataset_loaders import get_dataloader


def compute_reconstruction_loss(reconstructed: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute reconstruction loss (L1 + L2)"""
    l1_loss = nn.L1Loss()(reconstructed, target)
    l2_loss = nn.MSELoss()(reconstructed, target)
    return l1_loss + 0.5 * l2_loss


def train_epoch(model, dataloader, optimizer, device, epoch, writer, args):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_vq_freq_loss = 0.0
    total_vq_pixel_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.num_epochs}')
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)  # [B, C, H, W], values in [-1, 1] or [0, 1]
        
        # Normalize to [-1, 1] if needed
        if images.max() <= 1.0:
            images = images * 2.0 - 1.0
        
        # Forward pass
        reconstructed, vq_loss_freq, vq_loss_pixel = model(images)
        
        # Compute losses
        recon_loss = compute_reconstruction_loss(reconstructed, images)
        total_loss_batch = recon_loss + args.vq_weight * (vq_loss_freq + vq_loss_pixel)
        
        # Backward pass
        optimizer.zero_grad()
        total_loss_batch.backward()
        
        # Gradient clipping
        if args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        
        optimizer.step()
        
        # Accumulate losses
        total_loss += total_loss_batch.item()
        total_recon_loss += recon_loss.item()
        total_vq_freq_loss += vq_loss_freq.item()
        total_vq_pixel_loss += vq_loss_pixel.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{total_loss/num_batches:.4f}',
            'Recon': f'{total_recon_loss/num_batches:.4f}',
            'VQ': f'{(total_vq_freq_loss+total_vq_pixel_loss)/num_batches:.4f}'
        })
        
        # Log to tensorboard
        if batch_idx % args.log_freq == 0 and writer is not None:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/TotalLoss', total_loss_batch.item(), global_step)
            writer.add_scalar('Train/ReconstructionLoss', recon_loss.item(), global_step)
            writer.add_scalar('Train/VQ_Loss_Freq', vq_loss_freq.item(), global_step)
            writer.add_scalar('Train/VQ_Loss_Pixel', vq_loss_pixel.item(), global_step)
            
            if batch_idx == 0:
                # Log sample images
                with torch.no_grad():
                    # Denormalize for visualization
                    img_orig = (images[:4] + 1.0) / 2.0
                    img_recon = (reconstructed[:4] + 1.0) / 2.0
                    img_orig = torch.clamp(img_orig, 0, 1)
                    img_recon = torch.clamp(img_recon, 0, 1)
                    
                    # Concatenate original and reconstructed
                    comparison = torch.cat([img_orig, img_recon], dim=0)
                    writer.add_images('Train/Reconstruction', comparison, global_step)
    
    avg_loss = total_loss / num_batches
    avg_recon = total_recon_loss / num_batches
    avg_vq_freq = total_vq_freq_loss / num_batches
    avg_vq_pixel = total_vq_pixel_loss / num_batches
    
    return avg_loss, avg_recon, avg_vq_freq, avg_vq_pixel


def validate(model, dataloader, device, epoch, writer, args):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_vq_freq_loss = 0.0
    total_vq_pixel_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validation'):
            images = images.to(device)
            
            # Normalize to [-1, 1] if needed
            if images.max() <= 1.0:
                images = images * 2.0 - 1.0
            
            # Forward pass
            reconstructed, vq_loss_freq, vq_loss_pixel = model(images)
            
            # Compute losses
            recon_loss = compute_reconstruction_loss(reconstructed, images)
            total_loss_batch = recon_loss + args.vq_weight * (vq_loss_freq + vq_loss_pixel)
            
            total_loss += total_loss_batch.item()
            total_recon_loss += recon_loss.item()
            total_vq_freq_loss += vq_loss_freq.item()
            total_vq_pixel_loss += vq_loss_pixel.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_recon = total_recon_loss / num_batches
    avg_vq_freq = total_vq_freq_loss / num_batches
    avg_vq_pixel = total_vq_pixel_loss / num_batches
    
    # Log to tensorboard
    if writer is not None:
        writer.add_scalar('Val/TotalLoss', avg_loss, epoch)
        writer.add_scalar('Val/ReconstructionLoss', avg_recon, epoch)
        writer.add_scalar('Val/VQ_Loss_Freq', avg_vq_freq, epoch)
        writer.add_scalar('Val/VQ_Loss_Pixel', avg_vq_pixel, epoch)
        
        # Log sample images
        images = images[:4]
        reconstructed = reconstructed[:4]
        img_orig = (images + 1.0) / 2.0
        img_recon = (reconstructed + 1.0) / 2.0
        img_orig = torch.clamp(img_orig, 0, 1)
        img_recon = torch.clamp(img_recon, 0, 1)
        comparison = torch.cat([img_orig, img_recon], dim=0)
        writer.add_images('Val/Reconstruction', comparison, epoch)
    
    return avg_loss, avg_recon, avg_vq_freq, avg_vq_pixel


def main():
    parser = argparse.ArgumentParser(description='Train VQ Codebook Model')
    
    # Model arguments
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--freq_codebook_size', type=int, default=512)
    parser.add_argument('--freq_codebook_dim', type=int, default=64)
    parser.add_argument('--pixel_codebook_size', type=int, default=512)
    parser.add_argument('--pixel_codebook_dim', type=int, default=64)
    parser.add_argument('--vq_commitment_cost', type=float, default=0.25)
    parser.add_argument('--vq_decay', type=float, default=0.99)
    
    # Training arguments
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'imagenet', 'coco'])
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--vq_weight', type=float, default=1.0, help='Weight for VQ loss')
    parser.add_argument('--clip_grad', type=float, default=1.0)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Logging and checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_vq')
    parser.add_argument('--log_dir', type=str, default='./logs_vq')
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=10)
    
    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create model
    model = create_vq_codebook_model(
        in_channels=args.in_channels,
        hidden_dim=args.hidden_dim,
        freq_codebook_size=args.freq_codebook_size,
        freq_codebook_dim=args.freq_codebook_dim,
        pixel_codebook_size=args.pixel_codebook_size,
        pixel_codebook_dim=args.pixel_codebook_dim,
        vq_commitment_cost=args.vq_commitment_cost,
        vq_decay=args.vq_decay
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Create dataloaders
    train_loader = get_dataloader(
        dataset_name=args.dataset,
        data_root=args.data_root,
        split='train',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=32
    )
    
    val_loader = get_dataloader(
        dataset_name=args.dataset,
        data_root=args.data_root,
        split='val',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=32
    )
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=1e-6
    )
    
    # TensorBoard
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_recon, train_vq_freq, train_vq_pixel = train_epoch(
            model, train_loader, optimizer, device, epoch, writer, args
        )
        
        # Validate
        val_loss, val_recon, val_vq_freq, val_vq_pixel = validate(
            model, val_loader, device, epoch, writer, args
        )
        
        # Update learning rate
        scheduler.step()
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, VQ: {train_vq_freq+train_vq_pixel:.4f})")
        print(f"Val Loss: {val_loss:.4f} (Recon: {val_recon:.4f}, VQ: {val_vq_freq+val_vq_pixel:.4f})")
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'args': args
            }
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'best_model.pth'))
            print(f"Saved best model with val loss: {val_loss:.4f}")
        
        # Periodic checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'args': args
            }
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    writer.close()
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()

