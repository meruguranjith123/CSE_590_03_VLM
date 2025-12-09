#!/bin/bash
# Deploy to Google Cloud Platform
# This script creates a VM instance and sets up the training environment

set -e

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-your-project-id}"
ZONE="${GCP_ZONE:-us-central1-a}"
INSTANCE_NAME="${INSTANCE_NAME:-vlm-training}"
MACHINE_TYPE="${MACHINE_TYPE:-n1-standard-4}"
GPU_TYPE="${GPU_TYPE:-nvidia-tesla-t4}"
GPU_COUNT="${GPU_COUNT:-1}"
DISK_SIZE="${DISK_SIZE:-100GB}"
IMAGE_FAMILY="${IMAGE_FAMILY:-ubuntu-2004-lts}"
IMAGE_PROJECT="${IMAGE_PROJECT:-ubuntu-os-cloud}"

echo "=========================================="
echo "Deploying VLM Training to Google Cloud"
echo "=========================================="
echo "Project ID: $PROJECT_ID"
echo "Zone: $ZONE"
echo "Instance: $INSTANCE_NAME"
echo "Machine Type: $MACHINE_TYPE"
echo "GPU: $GPU_COUNT x $GPU_TYPE"
echo "=========================================="

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud CLI not found. Please install Google Cloud SDK."
    echo "Visit: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Set project
echo "Setting GCP project..."
gcloud config set project $PROJECT_ID

# Create instance with GPU
echo "Creating VM instance with GPU..."
if [ "$GPU_COUNT" -gt 0 ]; then
    gcloud compute instances create $INSTANCE_NAME \
        --zone=$ZONE \
        --machine-type=$MACHINE_TYPE \
        --accelerator="type=$GPU_TYPE,count=$GPU_COUNT" \
        --maintenance-policy=TERMINATE \
        --image-family=$IMAGE_FAMILY \
        --image-project=$IMAGE_PROJECT \
        --boot-disk-size=$DISK_SIZE \
        --boot-disk-type=pd-ssd \
        --scopes=https://www.googleapis.com/auth/cloud-platform \
        --metadata-from-file startup-script=cloud_setup/gcp_startup.sh
else
    # CPU-only instance
    gcloud compute instances create $INSTANCE_NAME \
        --zone=$ZONE \
        --machine-type=$MACHINE_TYPE \
        --image-family=$IMAGE_FAMILY \
        --image-project=$IMAGE_PROJECT \
        --boot-disk-size=$DISK_SIZE \
        --boot-disk-type=pd-ssd \
        --scopes=https://www.googleapis.com/auth/cloud-platform \
        --metadata-from-file startup-script=cloud_setup/gcp_startup.sh
fi

echo "=========================================="
echo "Instance created successfully!"
echo "=========================================="
echo "To SSH into the instance:"
echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
echo ""
echo "To view startup logs:"
echo "  gcloud compute instances get-serial-port-output $INSTANCE_NAME --zone=$ZONE"
echo ""
echo "To delete the instance when done:"
echo "  gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE"

