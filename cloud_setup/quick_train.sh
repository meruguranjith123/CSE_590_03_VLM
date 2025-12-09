#!/bin/bash
# Quick training script for Google Cloud
# Automatically downloads CIFAR-100 and trains for a few epochs

set -e

echo "=========================================="
echo "Quick Training on Google Cloud"
echo "=========================================="

# Activate virtual environment if it exists
if [ -d "$HOME/venv" ]; then
    source $HOME/venv/bin/activate
fi

# Navigate to project directory
cd ~/VLM || cd /home/$USER/VLM || (echo "Error: VLM directory not found" && exit 1)

# Default parameters
DATASET="${1:-cifar100}"
EPOCHS="${2:-10}"
BATCH_SIZE="${3:-32}"

echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo ""

# Run training
python cloud_setup/train_cloud.py \
    --dataset $DATASET \
    --data_root ./data \
    --batch_size $BATCH_SIZE \
    --num_epochs $EPOCHS \
    --auto_download \
    --checkpoint_dir ./checkpoints \
    --log_dir ./logs

echo ""
echo "=========================================="
echo "Training completed!"
echo "Checkpoints: ./checkpoints"
echo "Logs: ./logs"
echo "=========================================="

