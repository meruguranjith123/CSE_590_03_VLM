#!/bin/bash
# Setup script for Google Cloud Platform
# Run this on a GCP VM instance

set -e

echo "=========================================="
echo "Setting up VLM Training Environment on GCP"
echo "=========================================="

# Update system
echo "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install Python and pip
echo "Installing Python and dependencies..."
sudo apt-get install -y python3 python3-pip python3-venv git wget

# Install CUDA toolkit (if using GPU instances)
if nvidia-smi &> /dev/null; then
    echo "GPU detected. Installing CUDA dependencies..."
    sudo apt-get install -y nvidia-cuda-toolkit
fi

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv ~/venv
source ~/venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch (with CUDA if GPU available)
if nvidia-smi &> /dev/null; then
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "Installing PyTorch CPU version..."
    pip install torch torchvision torchaudio
fi

# Install project dependencies
echo "Installing project dependencies..."
cd ~/VLM
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p ~/VLM/data
mkdir -p ~/VLM/checkpoints
mkdir -p ~/VLM/logs
mkdir -p ~/VLM/generated

# Set permissions
chmod +x ~/VLM/cloud_setup/*.sh
chmod +x ~/VLM/train*.py
chmod +x ~/VLM/generate*.py

echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo "To activate the environment, run:"
echo "  source ~/venv/bin/activate"
echo "To start training, run:"
echo "  cd ~/VLM && python train.py --dataset cifar100 --data_root ~/VLM/data --batch_size 32 --num_epochs 10"

