#!/bin/bash
# Complete deployment and training script for Google Cloud
# This script deploys the VM and automatically starts training

set -e

echo "=========================================="
echo "VLM Cloud Deployment and Training"
echo "=========================================="

# Configuration - EDIT THESE VALUES
PROJECT_ID="${GCP_PROJECT_ID}"
ZONE="${GCP_ZONE:-us-central1-a}"
INSTANCE_NAME="${INSTANCE_NAME:-vlm-training}"
MACHINE_TYPE="${MACHINE_TYPE:-n1-standard-4}"
GPU_TYPE="${GPU_TYPE:-nvidia-tesla-t4}"
GPU_COUNT="${GPU_COUNT:-1}"
EPOCHS="${EPOCHS:-10}"
DATASET="${DATASET:-cifar100}"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud CLI not found!"
    echo "Please install Google Cloud SDK:"
    echo "  https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if PROJECT_ID is set
if [ -z "$PROJECT_ID" ]; then
    echo "Error: GCP_PROJECT_ID not set!"
    echo "Please set your project ID:"
    echo "  export GCP_PROJECT_ID='your-project-id'"
    echo ""
    echo "Or edit this script and set PROJECT_ID directly"
    exit 1
fi

echo "Configuration:"
echo "  Project ID: $PROJECT_ID"
echo "  Zone: $ZONE"
echo "  Instance: $INSTANCE_NAME"
echo "  Machine Type: $MACHINE_TYPE"
echo "  GPU: $GPU_COUNT x $GPU_TYPE"
echo "  Dataset: $DATASET"
echo "  Epochs: $EPOCHS"
echo ""

read -p "Continue with deployment? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled."
    exit 1
fi

# Set project
echo "Setting GCP project..."
gcloud config set project $PROJECT_ID

# Check if instance already exists
if gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE &> /dev/null; then
    echo "Instance $INSTANCE_NAME already exists!"
    read -p "Delete and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Deleting existing instance..."
        gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --quiet
    else
        echo "Using existing instance."
        INSTANCE_EXISTS=true
    fi
fi

if [ "$INSTANCE_EXISTS" != "true" ]; then
    # Create startup script with training command
    cat > /tmp/vlm_startup.sh << 'EOF'
#!/bin/bash
exec > /tmp/vlm_setup.log 2>&1

echo "Starting VLM setup at $(date)"

# Update system
apt-get update -qq
apt-get install -y python3 python3-pip python3-venv git wget curl

# Install CUDA if GPU available
if lspci | grep -i nvidia > /dev/null; then
    echo "Installing CUDA drivers..."
    apt-get install -y nvidia-cuda-toolkit nvidia-driver-470
fi

# Create VLM directory
mkdir -p /home/$USER/VLM
cd /home/$USER/VLM

# Note: Code will be uploaded via gcloud compute scp
# For now, we'll wait for code to be uploaded

echo "Setup complete at $(date)"
EOF

    # Create instance
    echo "Creating VM instance..."
    if [ "$GPU_COUNT" -gt 0 ]; then
        gcloud compute instances create $INSTANCE_NAME \
            --zone=$ZONE \
            --machine-type=$MACHINE_TYPE \
            --accelerator="type=$GPU_TYPE,count=$GPU_COUNT" \
            --maintenance-policy=TERMINATE \
            --image-family=ubuntu-2004-lts \
            --image-project=ubuntu-os-cloud \
            --boot-disk-size=100GB \
            --boot-disk-type=pd-ssd \
            --scopes=https://www.googleapis.com/auth/cloud-platform \
            --metadata-from-file startup-script=/tmp/vlm_startup.sh \
            --tags=http-server,https-server
    else
        gcloud compute instances create $INSTANCE_NAME \
            --zone=$ZONE \
            --machine-type=$MACHINE_TYPE \
            --image-family=ubuntu-2004-lts \
            --image-project=ubuntu-os-cloud \
            --boot-disk-size=100GB \
            --boot-disk-type=pd-ssd \
            --scopes=https://www.googleapis.com/auth/cloud-platform \
            --metadata-from-file startup-script=/tmp/vlm_startup.sh
    fi

    echo "Waiting for instance to be ready..."
    sleep 30
    
    # Wait for SSH to be ready
    echo "Waiting for SSH access..."
    for i in {1..30}; do
        if gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="echo 'SSH ready'" &> /dev/null; then
            break
        fi
        echo "  Attempt $i/30..."
        sleep 5
    done
fi

# Upload code to VM
echo "Uploading code to VM..."
gcloud compute scp --recurse \
    --zone=$ZONE \
    ../{models,datasets,cloud_setup,train.py,train_llm_prompt.py,generate.py,generate_llm_prompt.py,requirements.txt,test_setup.py} \
    $INSTANCE_NAME:~/VLM/ 2>/dev/null || {
    # If that fails, try uploading everything
    gcloud compute scp --recurse \
        --zone=$ZONE \
        . \
        $INSTANCE_NAME:~/VLM/ \
        --exclude="*.pyc" --exclude="__pycache__" --exclude=".git"
}

# Create setup and training script on VM
echo "Setting up environment on VM..."
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="
    cd ~/VLM
    
    # Create venv if not exists
    if [ ! -d ~/venv ]; then
        python3 -m venv ~/venv
    fi
    source ~/venv/bin/activate
    
    # Install dependencies
    pip install --upgrade pip setuptools wheel -q
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q || pip install torch torchvision torchaudio -q
    pip install -r requirements.txt -q
    
    # Make scripts executable
    chmod +x cloud_setup/*.sh 2>/dev/null || true
    
    echo 'Environment setup complete!'
"

# Start training
echo ""
echo "=========================================="
echo "Starting training..."
echo "=========================================="

gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="
    cd ~/VLM
    source ~/venv/bin/activate
    
    echo 'Starting training: Dataset=$DATASET, Epochs=$EPOCHS'
    echo ''
    
    # Run training in background and save output
    nohup python cloud_setup/train_cloud.py \
        --dataset $DATASET \
        --data_root ~/VLM/data \
        --batch_size 32 \
        --num_epochs $EPOCHS \
        --auto_download \
        --checkpoint_dir ~/VLM/checkpoints \
        --log_dir ~/VLM/logs \
        > ~/VLM/training.log 2>&1 &
    
    echo 'Training started in background!'
    echo 'Process ID:' \$!
    echo ''
    echo 'To monitor training:'
    echo '  tail -f ~/VLM/training.log'
    echo '  watch -n 1 nvidia-smi'
    echo ''
    echo 'Training output will be saved to:'
    echo '  ~/VLM/training.log'
    echo '  ~/VLM/checkpoints/'
    echo '  ~/VLM/logs/'
" <<< "$DATASET $EPOCHS"

echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo ""
echo "Instance: $INSTANCE_NAME"
echo "Zone: $ZONE"
echo ""
echo "To monitor training:"
echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
echo "  tail -f ~/VLM/training.log"
echo ""
echo "To view GPU usage:"
echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command='watch -n 1 nvidia-smi'"
echo ""
echo "To download results:"
echo "  gcloud compute scp --recurse $INSTANCE_NAME:~/VLM/checkpoints ./checkpoints --zone=$ZONE"
echo ""
echo "To stop the instance:"
echo "  gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE"
echo ""
echo "To delete the instance:"
echo "  gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE"

