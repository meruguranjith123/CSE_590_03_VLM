"""
Cloud Training Script with Auto Dataset Download
Automatically downloads datasets and runs training on Google Cloud
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import os
import sys
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train import train_distributed, setup, cleanup
from train_llm_prompt import train_distributed as train_llm_distributed
from models.frequency_prior_model import create_model
from models.frequency_prior_model import FrequencyPriorAutoregressiveModel
from models.llm_text_encoder import create_llm_prompt_model


def check_and_download_cifar100(data_root: str):
    """Check if CIFAR-100 exists, download if not"""
    import torchvision.datasets as datasets
    
    print("Checking CIFAR-100 dataset...")
    cifar_path = os.path.join(data_root, 'cifar100')
    
    try:
        # Try to load dataset (will download if not exists)
        print(f"Downloading to: {cifar_path}")
        train_dataset = datasets.CIFAR100(
            root=cifar_path,
            train=True,
            download=True,
            transform=None
        )
        print(f"✓ CIFAR-100 dataset ready at {cifar_path}")
        print(f"  Train samples: {len(train_dataset)}")
        
        val_dataset = datasets.CIFAR100(
            root=cifar_path,
            train=False,
            download=True,
            transform=None
        )
        print(f"  Val samples: {len(val_dataset)}")
        return True
    except Exception as e:
        print(f"✗ Error downloading CIFAR-100: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_gpu_availability():
    """Check GPU availability and print info"""
    if torch.cuda.is_available():
        print(f"✓ CUDA available")
        print(f"  GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        return True
    else:
        print("✗ CUDA not available, will use CPU (slower)")
        return False


def main():
    parser = argparse.ArgumentParser(description='Cloud Training with Auto Dataset Download')
    
    # Model type
    parser.add_argument('--model_type', type=str, default='base',
                       choices=['base', 'llm_prompt'],
                       help='Model type: base or llm_prompt')
    
    # Model arguments
    parser.add_argument('--num_layers', type=int, default=8,
                       help='Number of autoregressive layers')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension')
    parser.add_argument('--num_classes', type=int, default=None)
    
    # LLM arguments (if using llm_prompt)
    parser.add_argument('--vocab_size', type=int, default=50257)
    parser.add_argument('--text_embed_dim', type=int, default=512)
    parser.add_argument('--num_text_layers', type=int, default=6)
    parser.add_argument('--text_nhead', type=int, default=8)
    parser.add_argument('--max_text_length', type=int, default=512)
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='cifar100',
                       choices=['cifar100', 'imagenet', 'imagenet10k', 'coco'],
                       help='Dataset name')
    parser.add_argument('--data_root', type=str, default='./data',
                       help='Root directory for datasets')
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--auto_download', action='store_true', default=True,
                       help='Automatically download datasets')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Distributed training
    parser.add_argument('--world_size', type=int, default=None,
                       help='Number of GPUs (auto-detect if None)')
    parser.add_argument('--port', type=str, default='12355')
    
    # Logging
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--save_freq', type=int, default=5)
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Cloud Training Setup")
    print("=" * 60)
    
    # Check GPU
    has_gpu = check_gpu_availability()
    
    # Auto-detect world size
    if args.world_size is None:
        args.world_size = torch.cuda.device_count() if has_gpu else 1
    
    # Create directories
    os.makedirs(args.data_root, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Auto-download datasets
    if args.auto_download:
        print("\n" + "=" * 60)
        print("Auto-downloading datasets...")
        print("=" * 60)
        
        if args.dataset == 'cifar100':
            # Download CIFAR-100 to data_root/cifar100
            cifar_path = os.path.join(args.data_root, 'cifar100')
            check_and_download_cifar100(args.data_root)
            # Update data_root to point to parent directory for dataloader
            # (dataloader will append 'cifar100' internally)
        elif args.dataset in ['imagenet', 'imagenet10k']:
            print(f"⚠ ImageNet requires manual download.")
            print(f"  Please download ImageNet and place in: {args.data_root}/imagenet")
            print(f"  Structure: {args.data_root}/imagenet/train/class_name/")
        elif args.dataset == 'coco':
            print(f"⚠ COCO requires manual download.")
            print(f"  Please download COCO and place in: {args.data_root}/coco")
            print(f"  Structure: {args.data_root}/coco/images/train2017/")
    
    # Determine num_classes
    if args.num_classes is None:
        if args.dataset == 'cifar100':
            args.num_classes = 100
        elif args.dataset in ['imagenet', 'imagenet10k']:
            args.num_classes = 1000
        elif args.dataset == 'coco':
            args.num_classes = 80
    
    print("\n" + "=" * 60)
    print("Training Configuration")
    print("=" * 60)
    print(f"Model Type: {args.model_type}")
    print(f"Dataset: {args.dataset}")
    print(f"Data Root: {args.data_root}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print(f"World Size (GPUs): {args.world_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Checkpoint Dir: {args.checkpoint_dir}")
    print("=" * 60)
    
    # Start training
    print("\nStarting training...")
    print("-" * 60)
    
    try:
        if args.model_type == 'base':
            if args.world_size > 1:
                mp.spawn(
                    train_distributed,
                    args=(args.world_size, args),
                    nprocs=args.world_size,
                    join=True
                )
            else:
                train_distributed(0, 1, args)
        elif args.model_type == 'llm_prompt':
            if args.world_size > 1:
                mp.spawn(
                    train_llm_distributed,
                    args=(args.world_size, args),
                    nprocs=args.world_size,
                    join=True
                )
            else:
                train_llm_distributed(0, 1, args)
        
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print("=" * 60)
        print(f"Checkpoints saved to: {args.checkpoint_dir}")
        print(f"Logs saved to: {args.log_dir}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\n✗ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

