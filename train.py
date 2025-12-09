"""
Multi-GPU Training Script for Frequency Prior Autoregressive Image Generation
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import time
from tqdm import tqdm
import numpy as np

from models.frequency_prior_model import create_model
from datasets.dataset_loaders import get_dataloader


def setup(rank: int, world_size: int, port: str = "12355"):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)


def cleanup():
    """Clean up distributed training"""
    dist.destroy_process_group()


def compute_loss(pred_logits: torch.Tensor, target_images: torch.Tensor) -> torch.Tensor:
    """
    Compute cross-entropy loss for pixel prediction
    Args:
        pred_logits: [B, C, 256, H, W] logits for each pixel value
        target_images: [B, C, H, W] target images with values in [0, 255]
    Returns:
        Loss value
    """
    B, C, H, W = target_images.shape
    target = target_images.long()  # [B, C, H, W]
    
    # Reshape for cross-entropy: [B*C*H*W, 256] and [B*C*H*W]
    pred_flat = pred_logits.view(B * C * H * W, 256)
    target_flat = target.view(B * C * H * W)
    
    loss = nn.functional.cross_entropy(pred_flat, target_flat, reduction='mean')
    
    return loss


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    rank: int = 0,
    use_prompt: bool = False
):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    if rank == 0:
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    else:
        pbar = dataloader
    
    for batch_idx, batch in enumerate(pbar):
        if use_prompt:
            images, labels, prompt_embeds = batch
            prompt_embeds = prompt_embeds.to(device)
        else:
            images, labels = batch
            prompt_embeds = None
        
        images = images.to(device)
        labels = labels.to(device) if isinstance(labels, torch.Tensor) else None
        
        # Forward pass
        optimizer.zero_grad()
        
        if use_prompt:
            pred_logits = model(images, prompt_embeds, labels)
        else:
            pred_logits = model(images, labels)
        
        # Compute loss
        loss = compute_loss(pred_logits, images)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate loss
        total_loss += loss.item()
        num_batches += 1
        
        if rank == 0:
            pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    rank: int = 0,
    use_prompt: bool = False
):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if use_prompt:
                images, labels, prompt_embeds = batch
                prompt_embeds = prompt_embeds.to(device)
            else:
                images, labels = batch
                prompt_embeds = None
            
            images = images.to(device)
            labels = labels.to(device) if isinstance(labels, torch.Tensor) else None
            
            # Forward pass
            if use_prompt:
                pred_logits = model(images, prompt_embeds, labels)
            else:
                pred_logits = model(images, labels)
            
            # Compute loss
            loss = compute_loss(pred_logits, images)
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def train_distributed(
    rank: int,
    world_size: int,
    args: argparse.Namespace
):
    """Main training function for distributed training"""
    # Setup distributed training (skip if single GPU/CPU)
    if world_size > 1:
        setup(rank, world_size, args.port)
    
    # Determine device
    if world_size > 1:
        device = torch.device(f'cuda:{rank}')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = create_model(
        model_type='frequency_prior',
        in_channels=3,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        use_prompt_conditioning=args.use_prompt,
        prompt_embed_dim=args.prompt_embed_dim
    )
    
    model = model.to(device)
    
    # Use DDP only for multi-GPU, otherwise regular model
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=args.learning_rate * 0.01
    )
    
    # Create dataloaders
    train_loader, num_classes = get_dataloader(
        dataset_name=args.dataset,
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        train=True,
        max_samples=args.max_samples,
        shuffle=True
    )
    
    val_loader, _ = get_dataloader(
        dataset_name=args.dataset,
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        train=False,
        max_samples=args.max_samples,
        shuffle=False
    )
    
    # Update num_classes if not set
    if args.num_classes is None:
        args.num_classes = num_classes
    
    # TensorBoard writer (only on rank 0)
    if rank == 0:
        writer = SummaryWriter(log_dir=args.log_dir)
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        # Set epoch for distributed sampler
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, device, epoch, rank, args.use_prompt
        )
        
        # Validate
        val_loss = validate(model, val_loader, device, rank, args.use_prompt)
        
        # Update learning rate
        scheduler.step()
        
        # Log and save (only on rank 0)
        if rank == 0:
            current_lr = scheduler.get_last_lr()[0]
            
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Validation', val_loss, epoch)
            writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, "
                  f"Val Loss = {val_loss:.4f}, LR = {current_lr:.6f}")
            
            # Get model state dict (handle both DDP and regular model)
            model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            
            # Save checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model_state,
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
                    'model_state_dict': model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                    'args': args
                }
                torch.save(checkpoint, 
                          os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    if rank == 0:
        writer.close()
    
    # Only cleanup if using distributed training
    if world_size > 1:
        cleanup()


def main():
    parser = argparse.ArgumentParser(description='Train Frequency Prior Autoregressive Model')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='frequency_prior',
                       choices=['frequency_prior', 'prompt_conditional'],
                       help='Model type to train')
    parser.add_argument('--num_layers', type=int, default=8,
                       help='Number of autoregressive layers')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension')
    parser.add_argument('--num_classes', type=int, default=None,
                       help='Number of classes (None for auto-detect)')
    
    # Prompt conditioning
    parser.add_argument('--use_prompt', action='store_true',
                       help='Use prompt-based conditioning')
    parser.add_argument('--prompt_embed_dim', type=int, default=512,
                       help='Prompt embedding dimension')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['cifar100', 'imagenet', 'imagenet10k', 'coco'],
                       help='Dataset name')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of dataset')
    parser.add_argument('--image_size', type=int, default=32,
                       help='Image size (32 for CIFAR-100, 64/128 for others)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples (for ImageNet-10k)')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size per GPU')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Distributed training
    parser.add_argument('--world_size', type=int, default=torch.cuda.device_count(),
                       help='Number of GPUs')
    parser.add_argument('--port', type=str, default='12355',
                       help='Port for distributed training')
    
    # Logging and checkpoints
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='TensorBoard log directory')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--save_freq', type=int, default=10,
                       help='Checkpoint save frequency (epochs)')
    
    args = parser.parse_args()
    
    # Start distributed training
    if args.world_size > 1:
        mp.spawn(
            train_distributed,
            args=(args.world_size, args),
            nprocs=args.world_size,
            join=True
        )
    else:
        # Single GPU training
        train_distributed(0, 1, args)


if __name__ == '__main__':
    main()

