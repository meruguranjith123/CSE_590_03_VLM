#!/bin/bash
# Startup script for Google Cloud VM
# This runs automatically when the VM starts

set -e

# Redirect output to log file
exec > /tmp/vlm_setup.log 2>&1

echo "=========================================="
echo "VLM Training Environment Startup"
echo "Started at: $(date)"
echo "=========================================="

# Update system
apt-get update
apt-get upgrade -y

# Install Python and dependencies
apt-get install -y python3 python3-pip python3-venv git wget curl

# Install CUDA if GPU is available
if lspci | grep -i nvidia > /dev/null; then
    echo "GPU detected. Installing CUDA drivers..."
    apt-get install -y nvidia-cuda-toolkit nvidia-driver-470
    nvidia-smi
fi

# Clone repository (if not already present)
if [ ! -d "/home/$USER/VLM" ]; then
    echo "Cloning VLM repository..."
    cd /home/$USER
    git clone https://github.com/yourusername/VLM.git || {
        # If repo doesn't exist, create directory structure
        mkdir -p VLM
        cd VLM
        # Files will be uploaded separately or synced via gcloud
    }
fi

cd /home/$USER/VLM

# Create virtual environment
if [ ! -d "/home/$USER/venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv /home/$USER/venv
fi

source /home/$USER/venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch
if lspci | grep -i nvidia > /dev/null; then
    echo "Installing PyTorch with CUDA..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "Installing PyTorch CPU version..."
    pip install torch torchvision torchaudio
fi

# Install project dependencies
if [ -f "requirements.txt" ]; then
    echo "Installing project dependencies..."
    pip install -r requirements.txt
fi

# Create directories
mkdir -p data checkpoints logs generated

# Make scripts executable
chmod +x cloud_setup/*.sh 2>/dev/null || true
chmod +x train*.py generate*.py 2>/dev/null || true

echo "=========================================="
echo "Setup completed at: $(date)"
echo "=========================================="

# Optionally auto-start training (uncomment if desired)
# echo "Starting training..."
# source /home/$USER/venv/bin/activate
# cd /home/$USER/VLM
# python cloud_setup/train_cloud.py \
#     --dataset cifar100 \
#     --data_root ./data \
#     --batch_size 32 \
#     --num_epochs 10 \
#     --auto_download

echo "Setup complete. Ready for training!"
echo "SSH into the VM and run:"
echo "  cd ~/VLM"
echo "  source ~/venv/bin/activate"
echo "  python cloud_setup/train_cloud.py --dataset cifar100 --num_epochs 10"

