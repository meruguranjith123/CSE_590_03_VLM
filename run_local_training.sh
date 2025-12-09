#!/bin/bash
# Quick script to run training locally with auto dataset download

echo "=========================================="
echo "Local Training with Auto Dataset Download"
echo "=========================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 not found!"
    exit 1
fi

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies if needed
if ! python -c "import torch" &> /dev/null; then
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install torch torchvision torchaudio
    pip install -r requirements.txt
fi

# Run local training
echo ""
echo "Starting training (will auto-download CIFAR-100)..."
echo ""

python cloud_setup/local_train.py \
    --dataset cifar100 \
    --data_root ./data \
    --batch_size 16 \
    --num_epochs 5 \
    --hidden_dim 128 \
    --num_layers 4

echo ""
echo "Training complete! Checkpoints saved to ./checkpoints"

