# Quick Start: Google Cloud Training

## Fastest Way to Run Training on GCP

### Step 1: Setup (One-time)

```bash
# Set your project ID
export GCP_PROJECT_ID="your-project-id"
export GCP_ZONE="us-central1-a"

# Deploy VM instance
cd cloud_setup
chmod +x *.sh
./gcp_deploy.sh
```

### Step 2: SSH and Train

```bash
# SSH into the instance
gcloud compute ssh vlm-training --zone=$GCP_ZONE

# Run quick training (auto-downloads CIFAR-100, trains 10 epochs)
cd ~/VLM
source ~/venv/bin/activate
bash cloud_setup/quick_train.sh cifar100 10 32
```

That's it! The script will:
- ✓ Auto-download CIFAR-100 dataset
- ✓ Set up GPU environment
- ✓ Train for 10 epochs
- ✓ Save checkpoints to `./checkpoints`

## Training Options

### Quick Test (10 epochs)
```bash
bash cloud_setup/quick_train.sh cifar100 10 32
```

### Full Training (100 epochs)
```bash
python cloud_setup/train_cloud.py \
    --dataset cifar100 \
    --batch_size 32 \
    --num_epochs 100 \
    --auto_download
```

### Custom Configuration
```bash
python cloud_setup/train_cloud.py \
    --dataset cifar100 \
    --batch_size 64 \
    --num_epochs 50 \
    --hidden_dim 512 \
    --num_layers 12 \
    --learning_rate 1e-4 \
    --auto_download
```

## Monitor Training

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# View logs
tail -f logs/*

# TensorBoard
tensorboard --logdir=logs --port=6006
```

## Download Results

```bash
# From local machine
gcloud compute scp --recurse \
    vlm-training:~/VLM/checkpoints \
    ./local_checkpoints \
    --zone=$GCP_ZONE
```

## Clean Up

```bash
# Stop instance (saves money, keeps data)
gcloud compute instances stop vlm-training --zone=$GCP_ZONE

# Delete instance (removes everything)
gcloud compute instances delete vlm-training --zone=$GCP_ZONE
```

## Troubleshooting

**GPU not detected?**
```bash
nvidia-smi  # Check if GPU is available
sudo reboot # Reboot to initialize GPU drivers
```

**Out of memory?**
- Reduce batch size: `--batch_size 16`
- Use smaller model: `--hidden_dim 128 --num_layers 4`

**Slow download?**
- Dataset download happens automatically on first run
- CIFAR-100 is ~170MB, takes 1-2 minutes

