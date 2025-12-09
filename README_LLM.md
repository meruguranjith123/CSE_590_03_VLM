# LLM-based Prompt Image Generation

This document describes the LLM-based prompt image generation architecture and training pipeline.

## Architecture Overview

The LLM-based prompt model extends the base frequency prior autoregressive model with:

1. **Transformer-based Text Encoder**: Uses a transformer architecture similar to GPT/BERT to encode text prompts
2. **Cross-Attention Mechanism**: Image features attend to text features for better conditioning
3. **Gated Fusion**: Intelligently combines text and image features

## Components

### LLM Text Encoder (`models/llm_text_encoder.py`)

- **LLMTextEncoder**: Transformer-based encoder for text prompts
  - Token embeddings
  - Positional encoding
  - Multi-layer transformer encoder
  - Pooling to fixed-size embeddings

- **SimpleTokenizer**: Basic tokenizer (can be replaced with GPT-2 tokenizer or similar)
  - Word-based tokenization
  - Padding and truncation support
  - Special tokens: `<pad>`, `<unk>`, `<eos>`, `<bos>`

- **LLMPromptConditionalModel**: Complete model combining text encoder with image generator

### Text-Image Datasets (`datasets/text_image_dataset.py`)

Separate dataloaders that provide text-image pairs:

- **CIFAR100TextDataset**: CIFAR-100 with class-based text descriptions
- **ImageNetTextDataset**: ImageNet with class name descriptions
- **COCOTextDataset**: COCO with actual captions from annotations

## Training

### Basic Training

```bash
python train_llm_prompt.py \
    --dataset cifar100 \
    --data_root ./data \
    --image_size 32 \
    --batch_size 32 \
    --num_epochs 100 \
    --hidden_dim 256 \
    --text_embed_dim 512 \
    --num_text_layers 6
```

### Training with COCO (Real Captions)

```bash
python train_llm_prompt.py \
    --dataset coco \
    --data_root ./data/coco \
    --image_size 128 \
    --batch_size 8 \
    --num_epochs 150 \
    --hidden_dim 512 \
    --text_embed_dim 512 \
    --num_text_layers 6 \
    --max_text_length 256
```

### Multi-GPU Training

```bash
python train_llm_prompt.py \
    --dataset imagenet10k \
    --data_root ./data/imagenet \
    --image_size 64 \
    --batch_size 16 \
    --world_size 4 \
    --hidden_dim 512 \
    --text_embed_dim 768 \
    --num_text_layers 8
```

### Key Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--vocab_size` | Vocabulary size for tokenizer | 50257 |
| `--text_embed_dim` | Text embedding dimension | 512 |
| `--num_text_layers` | Number of transformer layers | 6 |
| `--text_nhead` | Number of attention heads | 8 |
| `--max_text_length` | Maximum text sequence length | 512 |
| `--use_frequency_prior` | Use frequency prior module | True |

## Image Generation

### Generate from Text Prompt

```bash
python generate_llm_prompt.py \
    --checkpoint ./checkpoints_llm/best_model.pth \
    --prompt "a beautiful landscape with mountains and a lake" \
    --output_dir ./generated_llm \
    --num_images 5 \
    --temperature 1.0
```

### Generate from Multiple Prompts

Create a text file `prompts.txt`:
```
a cute cat playing with a ball
a red sports car on a highway
a beautiful sunset over the ocean
a green forest with tall trees
```

Then run:
```bash
python generate_llm_prompt.py \
    --checkpoint ./checkpoints_llm/best_model.pth \
    --prompt_file prompts.txt \
    --output_dir ./generated_llm \
    --num_images 3
```

## Model Architecture Details

### Text Encoder

```
Input Text → Tokenization → Token Embeddings → Positional Encoding
    → Transformer Encoder (N layers) → Pooling → Text Embeddings
```

### Cross-Attention Fusion

```
Image Features [H×W, B, hidden_dim] (Query)
    ↓
Cross-Attention ← Text Embeddings [1, B, hidden_dim] (Key/Value)
    ↓
Attended Features + Gated Fusion → Final Features
```

### Training Loss

Same as base model: Cross-entropy loss for pixel prediction, with text conditioning applied throughout.

## Dataset Requirements

### CIFAR-100
- Automatically generates texts from class names: "a photo of a {class_name}"

### ImageNet-10k
- Uses class directory names as descriptions

### COCO
- Requires `captions_train2017.json` or `captions_val2017.json` in annotations folder
- Uses actual image captions from COCO dataset
- If caption file not found, falls back to generic descriptions

## Differences from Base Model

| Feature | Base Model | LLM Prompt Model |
|---------|------------|------------------|
| Input | Images (+ optional class labels) | Images + Text prompts |
| Text Processing | None | Transformer encoder |
| Conditioning | Class labels only | Text + Class labels |
| Datasets | Standard image datasets | Text-image pair datasets |
| Training Script | `train.py` | `train_llm_prompt.py` |

## Performance Tips

1. **Vocabulary Size**: Smaller vocab (10k-30k) can be faster for custom datasets
2. **Text Length**: Shorter sequences (128-256) are often sufficient
3. **Text Layers**: 4-6 layers work well for most cases
4. **Batch Size**: May need smaller batches due to text processing overhead

## Integration with Pretrained Tokenizers

To use GPT-2 or BERT tokenizer:

1. Install transformers: `pip install transformers`
2. Modify `SimpleTokenizer` or replace with:

```python
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
```

3. Update vocab_size in model args to match tokenizer vocab size

## Example Workflow

1. **Train base model** (optional, can start directly with LLM model):
```bash
python train.py --dataset cifar100 --data_root ./data
```

2. **Train LLM prompt model**:
```bash
python train_llm_prompt.py \
    --dataset cifar100 \
    --data_root ./data \
    --checkpoint_dir ./checkpoints_llm
```

3. **Generate images**:
```bash
python generate_llm_prompt.py \
    --checkpoint ./checkpoints_llm/best_model.pth \
    --prompt "a photo of a cat" \
    --num_images 10
```

## Troubleshooting

### Out of Memory
- Reduce `--max_text_length`
- Reduce `--num_text_layers`
- Reduce `--text_embed_dim`
- Reduce batch size

### Poor Text Conditioning
- Increase `--num_text_layers` for better text understanding
- Train for more epochs
- Use larger `--text_embed_dim`
- Ensure text descriptions are meaningful (COCO captions work best)

### Slow Training
- Reduce `--max_text_length`
- Use fewer text layers for faster training
- Enable mixed precision training

