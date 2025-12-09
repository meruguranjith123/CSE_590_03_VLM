"""
Image Generation Script for LLM-based Prompt Image Generation
Generate images from text prompts
"""

import torch
import torch.nn.functional as F
import argparse
import os
from PIL import Image
import numpy as np

from models.frequency_prior_model import FrequencyPriorAutoregressiveModel
from models.llm_text_encoder import LLMPromptConditionalModel, create_llm_prompt_model, SimpleTokenizer


def sample_pixel(pixel_logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Sample pixel value from logits
    Args:
        pixel_logits: [256] logits for pixel values
        temperature: Sampling temperature
    Returns:
        Sampled pixel value [0-255]
    """
    if temperature == 0:
        return pixel_logits.argmax()
    
    probs = F.softmax(pixel_logits / temperature, dim=0)
    return torch.multinomial(probs, 1).item()


def generate_image_from_prompt(
    model: torch.nn.Module,
    prompt: str,
    tokenizer: SimpleTokenizer,
    image_size: int = 32,
    num_channels: int = 3,
    device: torch.device = None,
    class_label: int = None,
    temperature: float = 1.0,
    seed: int = None
) -> np.ndarray:
    """
    Generate image from text prompt
    Args:
        model: Trained LLM prompt model
        prompt: Text prompt string
        tokenizer: Text tokenizer
        image_size: Output image size
        num_channels: Number of color channels
        device: Device to run on
        class_label: Optional class label for conditional generation
        temperature: Sampling temperature
        seed: Random seed
    Returns:
        Generated image as numpy array [H, W, C] in range [0, 255]
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    
    # Tokenize prompt
    tokenized = tokenizer([prompt], padding=True, truncation=True)
    token_ids = tokenized['input_ids'].to(device)
    attention_mask = tokenized['attention_mask'].to(device)
    
    # Initialize image
    image = torch.zeros(1, num_channels, image_size, image_size, dtype=torch.long, device=device)
    
    with torch.no_grad():
        # Generate pixel by pixel, channel by channel
        for h in range(image_size):
            for w in range(image_size):
                for c in range(num_channels):
                    # Forward pass
                    pred_logits = model(
                        image.float(),
                        token_ids=token_ids,
                        attention_mask=attention_mask,
                        class_labels=torch.tensor([class_label], device=device) if class_label else None
                    )
                    
                    # Get logits for current pixel
                    pixel_logits = pred_logits[0, c, :, h, w]  # [256]
                    
                    # Sample pixel value
                    pixel_value = sample_pixel(pixel_logits, temperature)
                    image[0, c, h, w] = pixel_value
    
    # Convert to numpy
    image_np = image[0].cpu().numpy().transpose(1, 2, 0)  # [H, W, C]
    
    return image_np.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description='Generate images from text prompts')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, default=None,
                       help='Text prompt for generation')
    parser.add_argument('--prompt_file', type=str, default=None,
                       help='File with prompts (one per line)')
    parser.add_argument('--output_dir', type=str, default='./generated_llm',
                       help='Output directory for generated images')
    parser.add_argument('--num_images', type=int, default=10,
                       help='Number of images to generate per prompt')
    parser.add_argument('--image_size', type=int, default=32,
                       help='Image size')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature')
    parser.add_argument('--class_label', type=int, default=None,
                       help='Class label for conditional generation')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model_args = checkpoint.get('args', argparse.Namespace())
    tokenizer_config = checkpoint.get('tokenizer_config', {})
    
    # Create tokenizer
    vocab_size = tokenizer_config.get('vocab_size', getattr(model_args, 'vocab_size', 50257))
    max_length = tokenizer_config.get('max_length', getattr(model_args, 'max_text_length', 512))
    tokenizer = SimpleTokenizer(vocab_size=vocab_size, max_length=max_length)
    
    # Create base model
    base_model = FrequencyPriorAutoregressiveModel(
        in_channels=3,
        num_layers=getattr(model_args, 'num_layers', 8),
        hidden_dim=getattr(model_args, 'hidden_dim', 256),
        num_classes=getattr(model_args, 'num_classes', None),
        use_frequency_prior=getattr(model_args, 'use_frequency_prior', True)
    )
    
    # Create LLM prompt model
    model = create_llm_prompt_model(
        base_model=base_model,
        vocab_size=vocab_size,
        text_embed_dim=getattr(model_args, 'text_embed_dim', 512),
        num_text_layers=getattr(model_args, 'num_text_layers', 6),
        text_nhead=getattr(model_args, 'text_nhead', 8),
        max_text_length=max_length
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get prompts
    if args.prompt_file:
        with open(args.prompt_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
    elif args.prompt:
        prompts = [args.prompt]
    else:
        # Default prompts
        prompts = [
            "a beautiful landscape",
            "a cute cat",
            "a red car",
            "a blue sky",
            "a green tree"
        ]
    
    print(f"Generating images for {len(prompts)} prompts...")
    for prompt_idx, prompt in enumerate(prompts):
        print(f"\nPrompt {prompt_idx + 1}/{len(prompts)}: '{prompt}'")
        for img_idx in range(args.num_images):
            seed = args.seed + img_idx if args.seed is not None else None
            image = generate_image_from_prompt(
                model=model,
                prompt=prompt,
                tokenizer=tokenizer,
                image_size=args.image_size,
                num_channels=3,
                device=device,
                class_label=args.class_label,
                temperature=args.temperature,
                seed=seed
            )
            
            # Save image
            image_pil = Image.fromarray(image)
            prompt_safe = prompt.replace(' ', '_').replace('/', '_')[:50]
            output_path = os.path.join(args.output_dir, f'prompt_{prompt_idx:03d}_img_{img_idx:04d}_{prompt_safe}.png')
            image_pil.save(output_path)
            print(f"  Saved: {output_path}")
    
    print("\nGeneration complete!")


if __name__ == '__main__':
    main()

