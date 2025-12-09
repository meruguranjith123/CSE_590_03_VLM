# Google Cloud Platform Deployment Guide

This guide explains how to deploy and run the VLM training pipeline on Google Cloud Platform.

## Prerequisites

1. **Google Cloud Account**: Sign up at https://cloud.google.com
2. **Google Cloud SDK**: Install from https://cloud.google.com/sdk/docs/install
3. **Billing Enabled**: Ensure billing is enabled for your project

## Quick Start

### Option 1: Automated Deployment Script

1. **Set environment variables**:
```bash
export GCP_PROJECT_ID="your-project-id"
export GCP_ZONE="us-central1-a"
export INSTANCE_NAME="vlm-training"
export MACHINE_TYPE="n1-standard-4"  # or n1-standard-8 for more CPUs
export GPU_TYPE="nvidia-tesla-t4"    # or nvidia-tesla-v100 for faster
export GPU_COUNT="1"
```

2. **Run deployment script**:
```bash
cd cloud_setup
chmod +x gcp_deploy.sh
./gcp_deploy.sh
```

3. **SSH into the instance**:
```bash
gcloud compute ssh $INSTANCE_NAME --zone=$GCP_ZONE
```

4. **Start training**:
```bash
cd ~/VLM
source ~/venv/bin/activate
python cloud_setup/train_cloud.py \
    --dataset cifar100 \
    --data_root ~/VLM/data \
    --batch_size 32 \
    --num_epochs 10 \
    --auto_download
```

### Option 2: Manual Setup

1. **Create VM instance**:
```bash
gcloud compute instances create vlm-training \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator="type=nvidia-tesla-t4,count=1" \
    --maintenance-policy=TERMINATE \
    --image-family=ubuntu-2004-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd
```

2. **SSH into instance**:
```bash
gcloud compute ssh vlm-training --zone=us-central1-a
```

3. **Run setup script**:
```bash
# Clone or upload repository
cd ~
git clone <your-repo-url> VLM || mkdir -p VLM && cd VLM
# Upload files using gcloud compute scp or other method

# Run setup
chmod +x cloud_setup/setup_gcp.sh
./cloud_setup/setup_gcp.sh
```

4. **Start training**:
```bash
source ~/venv/bin/activate
cd ~/VLM
python cloud_setup/train_cloud.py \
    --dataset cifar100 \
    --auto_download \
    --num_epochs 10
```

## Instance Types

### GPU Instances (Recommended for Training)

| Instance Type | vCPUs | RAM | GPU | Cost/Hour (approx) |
|--------------|-------|-----|-----|-------------------|
| n1-standard-4 | 4 | 15GB | 1x T4 | $0.35 |
| n1-standard-8 | 8 | 30GB | 1x T4 | $0.70 |
| n1-highmem-4 | 4 | 26GB | 1x V100 | $2.50 |
| a2-highgpu-1g | 12 | 85GB | 1x A100 | $3.50 |

### CPU Instances (For Testing/Small Datasets)

| Instance Type | vCPUs | RAM | Cost/Hour (approx) |
|--------------|-------|-----|-------------------|
| n1-standard-4 | 4 | 15GB | $0.19 |
| n1-standard-8 | 8 | 30GB | $0.38 |

## Training Examples

### Quick Test (CIFAR-100, 10 epochs)
```bash
python cloud_setup/train_cloud.py \
    --dataset cifar100 \
    --batch_size 32 \
    --num_epochs 10 \
    --auto_download
```

### Full Training (CIFAR-100, 100 epochs)
```bash
python cloud_setup/train_cloud.py \
    --dataset cifar100 \
    --batch_size 64 \
    --num_epochs 100 \
    --hidden_dim 512 \
    --num_layers 12 \
    --auto_download
```

### LLM Prompt Model Training
```bash
python cloud_setup/train_cloud.py \
    --model_type llm_prompt \
    --dataset coco \
    --batch_size 8 \
    --num_epochs 50 \
    --image_size 128 \
    --text_embed_dim 512 \
    --num_text_layers 6
```

### Multi-GPU Training (if instance has multiple GPUs)
```bash
python cloud_setup/train_cloud.py \
    --dataset cifar100 \
    --batch_size 32 \
    --num_epochs 100 \
    --world_size 4 \
    --auto_download
```

## Monitoring Training

### View Training Logs
```bash
# On the VM
tail -f ~/VLM/logs/*

# Or view via TensorBoard
tensorboard --logdir=~/VLM/logs --port=6006
# Then access via: http://YOUR_VM_IP:6006
```

### Check GPU Usage
```bash
watch -n 1 nvidia-smi
```

### View System Resources
```bash
htop
```

## Transferring Data

### Upload Code to VM
```bash
gcloud compute scp --recurse ./VLM vlm-training:~/ --zone=us-central1-a
```

### Download Checkpoints
```bash
gcloud compute scp --recurse vlm-training:~/VLM/checkpoints ./local_checkpoints --zone=us-central1-a
```

### Download Generated Images
```bash
gcloud compute scp --recurse vlm-training:~/VLM/generated ./local_generated --zone=us-central1-a
```

## Auto-download Datasets

The `train_cloud.py` script automatically downloads CIFAR-100. For other datasets:

- **CIFAR-100**: Auto-downloaded ✓
- **ImageNet**: Requires manual download (large dataset)
- **COCO**: Requires manual download (large dataset)

To manually download ImageNet/COCO:
```bash
# On the VM
cd ~/VLM/data
# Follow dataset-specific download instructions
```

## Cost Management

### Estimate Costs
- Use GCP Pricing Calculator: https://cloud.google.com/products/calculator
- Monitor usage: https://console.cloud.google.com/billing

### Stop Instance When Not Training
```bash
gcloud compute instances stop vlm-training --zone=us-central1-a
```

### Delete Instance to Avoid Charges
```bash
gcloud compute instances delete vlm-training --zone=us-central1-a
```

### Use Preemptible Instances (60-80% cheaper)
```bash
gcloud compute instances create vlm-training \
    --preemptible \
    --machine-type=n1-standard-4 \
    --accelerator="type=nvidia-tesla-t4,count=1" \
    ...
```

## Troubleshooting

### GPU Not Detected
```bash
# Check if GPU is available
nvidia-smi

# Reinstall drivers if needed
sudo apt-get install --reinstall nvidia-driver-470
sudo reboot
```

### Out of Memory
- Reduce batch size: `--batch_size 16`
- Use smaller model: `--hidden_dim 128 --num_layers 4`
- Use smaller images: `--image_size 32`

### Slow Training
- Use GPU instance (T4 or V100)
- Increase batch size if memory allows
- Use multiple GPUs: `--world_size 2`

### Connection Issues
```bash
# Check instance status
gcloud compute instances describe vlm-training --zone=us-central1-a

# Reset SSH keys if needed
gcloud compute config-ssh
```

## Advanced: Using Google Cloud Storage

### Upload Datasets to GCS
```bash
gsutil -m cp -r ./data/cifar100 gs://your-bucket-name/datasets/
```

### Download from GCS in Training Script
```python
# Add to train_cloud.py
import subprocess
subprocess.run(['gsutil', '-m', 'cp', '-r', 
                'gs://your-bucket-name/datasets/cifar100', 
                './data/'])
```

## Best Practices

1. **Use Spot/Preemptible Instances**: 60-80% cost savings for long training
2. **Monitor Costs**: Set up billing alerts
3. **Save Checkpoints Regularly**: Use `--save_freq 10`
4. **Use Persistent Disks**: Faster than standard disks
5. **Stop Instances**: When not in use to avoid charges
6. **Use Regional Persistent Disks**: For multi-zone redundancy

## Quick Reference

```bash
# Create instance
gcloud compute instances create vlm-training --zone=us-central1-a --machine-type=n1-standard-4 --accelerator="type=nvidia-tesla-t4,count=1"

# SSH
gcloud compute ssh vlm-training --zone=us-central1-a

# Start training
python cloud_setup/train_cloud.py --dataset cifar100 --num_epochs 10 --auto_download

# Stop instance
gcloud compute instances stop vlm-training --zone=us-central1-a

# Delete instance
gcloud compute instances delete vlm-training --zone=us-central1-a
```

