#!/usr/bin/env python3
"""
GPU Training Script - Auto-runs on any GPU server
Just upload all code and run: python run_gpu_training.py
"""

import torch
import torch.nn as nn
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from train import train_distributed, setup, cleanup
from datasets.dataset_loaders import get_dataloader
from models.frequency_prior_model import create_model
import torchvision.datasets as datasets


def check_gpu():
    """Check and print GPU information"""
    if torch.cuda.is_available():
        print("=" * 60)
        print("GPU DETECTED - Using CUDA")
        print("=" * 60)
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        print("=" * 60)
        return True
    else:
        print("=" * 60)
        print("NO GPU DETECTED - Will use CPU (very slow)")
        print("=" * 60)
        return False


def download_cifar100(data_root: str):
    """Download CIFAR-100 dataset"""
    print("\nDownloading CIFAR-100 dataset...")
    cifar_path = os.path.join(data_root, 'cifar100')
    
    try:
        print(f"Download location: {cifar_path}")
        train_dataset = datasets.CIFAR100(
            root=cifar_path,
            train=True,
            download=True,
            transform=None
        )
        print(f"✓ CIFAR-100 downloaded successfully!")
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


def main():
    print("\n" + "=" * 60)
    print("VLM GPU Training - Auto Setup")
    print("=" * 60)
    
    # Check GPU
    has_gpu = check_gpu()
    
    # Configuration
    config = {
        'dataset': 'cifar100',
        'data_root': './data',
        'image_size': 32,
        'batch_size': 32 if has_gpu else 8,
        'num_epochs': 10,
        'learning_rate': 3e-4,
        'hidden_dim': 256,
        'num_layers': 8,
        'num_workers': 4 if has_gpu else 2,
        'world_size': torch.cuda.device_count() if has_gpu else 1,
        'checkpoint_dir': './checkpoints',
        'log_dir': './logs',
        'num_classes': 100
    }
    
    # Adjust batch size based on GPU memory
    if has_gpu:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory < 8:  # Less than 8GB
            config['batch_size'] = 16
            config['hidden_dim'] = 128
        elif gpu_memory >= 16:  # 16GB or more
            config['batch_size'] = 64
            config['hidden_dim'] = 512
    
    print("\n" + "=" * 60)
    print("Training Configuration")
    print("=" * 60)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
    # Create directories
    os.makedirs(config['data_root'], exist_ok=True)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    # Download dataset
    print("\n" + "=" * 60)
    print("Dataset Setup")
    print("=" * 60)
    if not download_cifar100(config['data_root']):
        print("Failed to download dataset. Exiting.")
        sys.exit(1)
    
    # Update data_root for dataloader
    config['data_root'] = os.path.join(config['data_root'], 'cifar100')
    
    # Create namespace for args
    class Args:
        pass
    
    args = Args()
    for key, value in config.items():
        setattr(args, key, value)
    args.port = '12355'
    args.weight_decay = 1e-4
    args.save_freq = 5
    args.use_prompt = False
    args.use_frequency_prior = True
    
    # Start training
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    print(f"Device: {'GPU' if has_gpu else 'CPU'}")
    print(f"Checkpoints will be saved to: {config['checkpoint_dir']}")
    print(f"Logs will be saved to: {config['log_dir']}")
    print("=" * 60)
    print()
    
    try:
        # Use single GPU or CPU
        train_distributed(0, 1, args)
        
        print("\n" + "=" * 60)
        print("Training Completed Successfully!")
        print("=" * 60)
        print(f"Best checkpoint: {config['checkpoint_dir']}/best_model.pth")
        print(f"TensorBoard logs: {config['log_dir']}")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n\n✗ Training failed with error:")
        print(str(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

