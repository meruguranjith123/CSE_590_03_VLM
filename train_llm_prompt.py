"""
Multi-GPU Training Script for LLM-based Prompt Image Generation
Separate training script for text-conditioned image generation
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

from models.frequency_prior_model import FrequencyPriorAutoregressiveModel
from models.llm_text_encoder import LLMPromptConditionalModel, create_llm_prompt_model, SimpleTokenizer
from datasets.text_image_dataset import get_text_image_dataloader


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
    rank: int = 0
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
        images = batch['images'].to(device)
        labels = batch['labels'].to(device)
        token_ids = batch['token_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        pred_logits = model(
            images,
            token_ids=token_ids,
            attention_mask=attention_mask,
            class_labels=labels
        )
        
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
    rank: int = 0
):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['images'].to(device)
            labels = batch['labels'].to(device)
            token_ids = batch['token_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            pred_logits = model(
                images,
                token_ids=token_ids,
                attention_mask=attention_mask,
                class_labels=labels
            )
            
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
    # Setup distributed training
    setup(rank, world_size, args.port)
    
    device = torch.device(f'cuda:{rank}')
    
    # Create tokenizer
    tokenizer = SimpleTokenizer(
        vocab_size=args.vocab_size,
        max_length=args.max_text_length
    )
    
    # Create base model
    base_model = FrequencyPriorAutoregressiveModel(
        in_channels=3,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        use_frequency_prior=args.use_frequency_prior
    )
    
    # Create LLM-based prompt model
    model = create_llm_prompt_model(
        base_model=base_model,
        vocab_size=args.vocab_size,
        text_embed_dim=args.text_embed_dim,
        num_text_layers=args.num_text_layers,
        text_nhead=args.text_nhead,
        max_text_length=args.max_text_length
    )
    
    model = model.to(device)
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
    train_loader, num_classes = get_text_image_dataloader(
        dataset_name=args.dataset,
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        train=True,
        max_samples=args.max_samples,
        tokenizer=tokenizer,
        max_text_length=args.max_text_length,
        shuffle=True
    )
    
    val_loader, _ = get_text_image_dataloader(
        dataset_name=args.dataset,
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        train=False,
        max_samples=args.max_samples,
        tokenizer=tokenizer,
        max_text_length=args.max_text_length,
        shuffle=False
    )
    
    # Update num_classes if not set
    if args.num_classes is None:
        args.num_classes = num_classes
        # Update model's class embedding if needed
        if hasattr(base_model, 'class_embed'):
            base_model.class_embed = nn.Embedding(num_classes, args.hidden_dim).to(device)
    
    # TensorBoard writer (only on rank 0)
    if rank == 0:
        writer = SummaryWriter(log_dir=args.log_dir)
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        # Set epoch for distributed sampler
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, device, epoch, rank
        )
        
        # Validate
        val_loss = validate(model, val_loader, device, rank)
        
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
            
            # Save checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                    'args': args,
                    'tokenizer_config': {
                        'vocab_size': args.vocab_size,
                        'max_length': args.max_text_length
                    }
                }
                torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'best_model.pth'))
                print(f"Saved best model with val loss: {val_loss:.4f}")
            
            # Periodic checkpoint
            if (epoch + 1) % args.save_freq == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                    'args': args,
                    'tokenizer_config': {
                        'vocab_size': args.vocab_size,
                        'max_length': args.max_text_length
                    }
                }
                torch.save(checkpoint, 
                          os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    if rank == 0:
        writer.close()
    
    cleanup()


def main():
    parser = argparse.ArgumentParser(description='Train LLM-based Prompt Image Generation Model')
    
    # Model arguments
    parser.add_argument('--num_layers', type=int, default=8,
                       help='Number of autoregressive layers')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension for image model')
    parser.add_argument('--num_classes', type=int, default=None,
                       help='Number of classes (None for auto-detect)')
    parser.add_argument('--use_frequency_prior', action='store_true', default=True,
                       help='Use frequency prior module')
    
    # LLM Text Encoder arguments
    parser.add_argument('--vocab_size', type=int, default=50257,
                       help='Vocabulary size for tokenizer')
    parser.add_argument('--text_embed_dim', type=int, default=512,
                       help='Text embedding dimension')
    parser.add_argument('--num_text_layers', type=int, default=6,
                       help='Number of transformer layers in text encoder')
    parser.add_argument('--text_nhead', type=int, default=8,
                       help='Number of attention heads in text encoder')
    parser.add_argument('--max_text_length', type=int, default=512,
                       help='Maximum text sequence length')
    
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
    parser.add_argument('--log_dir', type=str, default='./logs_llm',
                       help='TensorBoard log directory')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_llm',
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

