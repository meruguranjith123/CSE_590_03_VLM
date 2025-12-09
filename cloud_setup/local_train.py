"""
Local Training Script with Auto Dataset Download
Runs training locally with automatic dataset downloading
"""

import torch
import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train import train_distributed
from train_llm_prompt import train_distributed as train_llm_distributed
import torchvision.datasets as datasets


def check_and_download_cifar100(data_root: str):
    """Check if CIFAR-100 exists, download if not"""
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
        print(f"✓ CIFAR-100 dataset ready")
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


def check_gpu():
    """Check GPU availability"""
    if torch.cuda.is_available():
        print(f"✓ CUDA available")
        print(f"  GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        return True
    else:
        print("⚠ CUDA not available, will use CPU (training will be slow)")
        return False


def main():
    parser = argparse.ArgumentParser(description='Local Training with Auto Dataset Download')
    
    # Model type
    parser.add_argument('--model_type', type=str, default='base',
                       choices=['base', 'llm_prompt'],
                       help='Model type: base or llm_prompt')
    
    # Model arguments
    parser.add_argument('--num_layers', type=int, default=4,
                       help='Number of autoregressive layers (reduced for local)')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden dimension (reduced for local)')
    parser.add_argument('--num_classes', type=int, default=100)
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='cifar100',
                       choices=['cifar100'],
                       help='Dataset (CIFAR-100 auto-downloads)')
    parser.add_argument('--data_root', type=str, default='./data',
                       help='Root directory for datasets')
    parser.add_argument('--image_size', type=int, default=32)
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size (reduced for local)')
    parser.add_argument('--num_epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--num_workers', type=int, default=2)
    
    # Logging
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Local Training Setup")
    print("=" * 60)
    
    # Check GPU
    has_gpu = check_gpu()
    args.world_size = torch.cuda.device_count() if has_gpu else 1
    
    # Create directories
    os.makedirs(args.data_root, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Download dataset
    print("\n" + "=" * 60)
    print("Downloading dataset...")
    print("=" * 60)
    
    if args.dataset == 'cifar100':
        if not check_and_download_cifar100(args.data_root):
            print("Failed to download dataset. Exiting.")
            sys.exit(1)
        args.data_root = os.path.join(args.data_root, 'cifar100')
    
    print("\n" + "=" * 60)
    print("Training Configuration")
    print("=" * 60)
    print(f"Model Type: {args.model_type}")
    print(f"Dataset: {args.dataset}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Device: {'GPU' if has_gpu else 'CPU'}")
    print("=" * 60)
    
    # Start training
    print("\nStarting training...")
    print("-" * 60)
    
    try:
        if args.model_type == 'base':
            train_distributed(0, 1, args)
        elif args.model_type == 'llm_prompt':
            train_llm_distributed(0, 1, args)
        
        print("\n" + "=" * 60)
        print("Training completed!")
        print("=" * 60)
        print(f"Checkpoints: {args.checkpoint_dir}")
        print(f"Logs: {args.log_dir}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted")
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

