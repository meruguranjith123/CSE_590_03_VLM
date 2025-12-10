# Frequency Prior for Autoregressive Image Generation

A full trainable repository implementing autoregressive image generation with frequency prior (mid-term method) and prompt-based conditional generation extension.

## Authors

- **Ranjith Merugu** - Roll Number: 116842918
- **Sambhav Shrestha** - Roll Number: 117525285

Course: CSE 590-03 (VLM)

## Architecture Overview

### Main Method: Frequency Prior Autoregressive Model

The core architecture implements autoregressive image generation with frequency prior:

- **Autoregressive Generation**: Uses masked convolutions to generate images pixel-by-pixel
- **Frequency Prior Module**: Captures frequency domain statistics to improve generation quality
- **Residual Blocks**: Deep autoregressive layers with residual connections

### Extension 1: Prompt-Based Conditional Generation

An extension that adds prompt-based conditioning:

- **Text Prompt Conditioning**: Uses text embeddings to guide image generation
- **Gated Fusion**: Intelligently combines prompt features with image features
- **Flexible Architecture**: Can be combined with class conditioning

### Extension 2: LLM-Based Prompt Image Generation (NEW)

A full LLM-based architecture for text-to-image generation:

- **Transformer Text Encoder**: GPT/BERT-style transformer for encoding text prompts
- **Cross-Attention Mechanism**: Image features attend to text for better conditioning
- **Separate Training Pipeline**: Dedicated dataloaders and training script
- **Text-Image Datasets**: Support for CIFAR-100, ImageNet, and COCO with captions

**See [README_LLM.md](README_LLM.md) for detailed LLM prompt generation documentation.**

## Repository Structure

```
VLM/
├── models/
│   ├── frequency_prior_model.py    # Core model architectures
│   └── llm_text_encoder.py         # LLM-based text encoder
├── datasets/
│   ├── dataset_loaders.py          # Dataset loaders (CIFAR-100, ImageNet-10k, COCO)
│   └── text_image_dataset.py       # Text-image pair datasets for LLM training
├── train.py                        # Multi-GPU training script (base model)
├── train_llm_prompt.py             # Multi-GPU training script (LLM prompt model)
├── generate.py                     # Image generation script (base model)
├── generate_llm_prompt.py          # Image generation script (LLM prompt model)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── README_LLM.md                   # LLM prompt generation documentation
```

## Installation

### Local Installation

1. **Clone the repository**:
```bash
cd /Users/meruguranjith/VLM
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Ensure CUDA is available** (for GPU training):
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Google Cloud Platform Deployment

For cloud training with automatic dataset downloading:

1. **See [cloud_setup/README_GCP.md](cloud_setup/README_GCP.md) for full deployment guide**

2. **Quick start on GCP**:
```bash
# Set your GCP project
export GCP_PROJECT_ID="your-project-id"

# Deploy to GCP
cd cloud_setup
./gcp_deploy.sh

# SSH and train
gcloud compute ssh vlm-training --zone=us-central1-a
cd ~/VLM && source ~/venv/bin/activate
python cloud_setup/train_cloud.py --dataset cifar100 --num_epochs 10 --auto_download
```

The cloud setup automatically:
- Downloads CIFAR-100 dataset
- Sets up GPU environment
- Runs training with proper configuration

## Dataset Setup

### CIFAR-100
The dataset will be automatically downloaded when you run training. Set `--data_root` to where you want to store it.

### ImageNet-10k
Download ImageNet and place it in the following structure:
```
data_root/
├── train/
│   ├── class1/
│   │   ├── img1.jpg
│   │   └── ...
│   └── class2/
│       └── ...
└── val/
    └── ...
```

### COCO
Download COCO dataset and place it in:
```
data_root/
├── images/
│   ├── train2017/
│   └── val2017/
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

## Training

### Single GPU Training

```bash
python train.py \
    --dataset cifar100 \
    --data_root ./data \
    --image_size 32 \
    --batch_size 32 \
    --num_epochs 100 \
    --learning_rate 3e-4 \
    --hidden_dim 256 \
    --num_layers 8 \
    --checkpoint_dir ./checkpoints \
    --log_dir ./logs
```

### Multi-GPU Training

The script automatically detects available GPUs and uses distributed data parallel (DDP):

```bash
python train.py \
    --dataset imagenet10k \
    --data_root ./data/imagenet \
    --image_size 64 \
    --batch_size 16 \
    --num_epochs 200 \
    --learning_rate 3e-4 \
    --hidden_dim 512 \
    --num_layers 12 \
    --world_size 4 \
    --port 12355 \
    --checkpoint_dir ./checkpoints \
    --log_dir ./logs
```

### Training with Prompt Conditioning

```bash
python train.py \
    --dataset coco \
    --data_root ./data/coco \
    --image_size 128 \
    --batch_size 8 \
    --num_epochs 150 \
    --use_prompt \
    --prompt_embed_dim 512 \
    --checkpoint_dir ./checkpoints \
    --log_dir ./logs
```

### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--dataset` | Dataset name (cifar100, imagenet, imagenet10k, coco) | Required |
| `--data_root` | Root directory of dataset | Required |
| `--image_size` | Image size (32 for CIFAR-100, 64/128 for others) | 32 |
| `--batch_size` | Batch size per GPU | 32 |
| `--num_epochs` | Number of training epochs | 100 |
| `--learning_rate` | Learning rate | 3e-4 |
| `--hidden_dim` | Hidden dimension | 256 |
| `--num_layers` | Number of autoregressive layers | 8 |
| `--num_workers` | Number of data loading workers | 4 |
| `--world_size` | Number of GPUs | Auto-detect |
| `--use_prompt` | Enable prompt-based conditioning | False |
| `--prompt_embed_dim` | Prompt embedding dimension | 512 |
| `--weight_decay` | Weight decay | 1e-4 |
| `--max_samples` | Maximum samples (for ImageNet-10k) | None |

## Image Generation

Generate images from a trained model:

```bash
python generate.py \
    --checkpoint ./checkpoints/best_model.pth \
    --output_dir ./generated \
    --num_images 10 \
    --image_size 32 \
    --temperature 1.0 \
    --class_label 5
```

### Generation Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--checkpoint` | Path to model checkpoint | Required |
| `--output_dir` | Output directory | ./generated |
| `--num_images` | Number of images to generate | 10 |
| `--image_size` | Image size | 32 |
| `--temperature` | Sampling temperature (higher = more diverse) | 1.0 |
| `--class_label` | Class label for conditional generation | None |
| `--seed` | Random seed | None |

## Model Architecture Details

### Frequency Prior Module

The frequency prior module analyzes image features in the frequency domain to provide additional guidance for autoregressive generation:

- Input: Image features [B, C, H, W]
- Process: Frequency-domain convolution → Normalization → Prior prediction
- Output: Frequency prior features [B, C, H, W]

### Autoregressive Blocks

Each autoregressive block uses:
- **Masked Convolutions**: Ensures causal dependencies (only previous pixels can influence current pixel)
- **Residual Connections**: Enables deep networks
- **Group Normalization**: Stabilizes training

### Prompt Conditioning (Extension)

The prompt-based extension:
- Encodes text prompts into embeddings
- Uses gated fusion to combine prompts with image features
- Maintains compatibility with class conditioning

## Multi-GPU Training Details

The training script uses PyTorch's DistributedDataParallel (DDP):

- **Automatic GPU Detection**: Detects available GPUs automatically
- **Data Parallelism**: Splits batches across GPUs
- **Gradient Synchronization**: Synchronizes gradients across GPUs
- **Mixed Precision**: Can be enabled for faster training (add `torch.cuda.amp`)

### Recommended GPU Configurations

- **CIFAR-100 (32x32)**: 2-4 GPUs with 16-32 batch size per GPU
- **ImageNet-10k (64x64)**: 4-8 GPUs with 8-16 batch size per GPU
- **COCO (128x128)**: 8+ GPUs with 4-8 batch size per GPU

## Monitoring Training

TensorBoard logs are saved in `--log_dir`:

```bash
tensorboard --logdir ./logs
```

Monitor:
- Training/Validation Loss
- Learning Rate Schedule
- Model Parameters

## Checkpoints

Checkpoints are saved in `--checkpoint_dir`:
- `best_model.pth`: Best model based on validation loss
- `checkpoint_epoch_N.pth`: Periodic checkpoints (every `--save_freq` epochs)

Each checkpoint contains:
- Model state dict
- Optimizer state dict
- Scheduler state dict
- Training arguments
- Validation loss

## Example Training Scripts

### Quick Start (CIFAR-100)

```bash
python train.py \
    --dataset cifar100 \
    --data_root ./data \
    --batch_size 32 \
    --num_epochs 50 \
    --hidden_dim 256 \
    --num_layers 8
```

### ImageNet-10k Training

```bash
python train.py \
    --dataset imagenet10k \
    --data_root ./data/imagenet \
    --image_size 64 \
    --batch_size 16 \
    --num_epochs 200 \
    --hidden_dim 512 \
    --num_layers 12 \
    --max_samples 10000
```

### COCO with Prompt Conditioning

```bash
python train.py \
    --dataset coco \
    --data_root ./data/coco \
    --image_size 128 \
    --batch_size 8 \
    --num_epochs 150 \
    --use_prompt \
    --hidden_dim 512 \
    --num_layers 12
```

## Tips for Training

1. **Start Small**: Begin with CIFAR-100 and 32x32 images
2. **Batch Size**: Use largest batch size that fits in memory
3. **Learning Rate**: 3e-4 works well, adjust based on loss curves
4. **Gradient Clipping**: Already enabled (max_norm=1.0)
5. **Warm-up**: Consider adding learning rate warm-up for large models
6. **Mixed Precision**: Can speed up training with minimal accuracy loss

## Troubleshooting

### Out of Memory
- Reduce `--batch_size`
- Reduce `--image_size`
- Reduce `--hidden_dim` or `--num_layers`
- Use gradient accumulation

### Slow Training
- Increase `--num_workers`
- Enable mixed precision training
- Use more GPUs
- Reduce image size for initial experiments

### Poor Generation Quality
- Train for more epochs
- Increase model capacity (`--hidden_dim`, `--num_layers`)
- Adjust temperature during generation
- Check dataset quality

## Citation

If you use this code, please cite the mid-term report on "Frequency Prior for Autoregressive Image Generation".

## Authors

- **Ranjith Merugu** - Roll Number: 116842918
- **Sambhav Shrestha** - Roll Number: 117525285

Course: CSE 590-03 (VLM)

## License

This code is provided for research purposes.

