"""
Image Generation Script
Supports both base model and prompt-conditional generation
"""

import torch
import torch.nn.functional as F
import argparse
import os
from PIL import Image
import numpy as np

from models.frequency_prior_model import create_model
from datasets.dataset_loaders import get_transform


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


def generate_image(
    model: torch.nn.Module,
    image_size: int = 32,
    num_channels: int = 3,
    device: torch.device = None,
    class_label: int = None,
    prompt_embedding: torch.Tensor = None,
    temperature: float = 1.0,
    seed: int = None
) -> np.ndarray:
    """
    Generate image autoregressively
    Args:
        model: Trained model
        image_size: Output image size
        num_channels: Number of color channels
        device: Device to run on
        class_label: Optional class label for conditional generation
        prompt_embedding: Optional prompt embedding for prompt-conditional generation
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
    
    # Initialize image
    image = torch.zeros(1, num_channels, image_size, image_size, dtype=torch.long, device=device)
    
    with torch.no_grad():
        # Generate pixel by pixel, channel by channel
        for h in range(image_size):
            for w in range(image_size):
                for c in range(num_channels):
                    # Forward pass
                    if prompt_embedding is not None:
                        pred_logits = model(image.float(), prompt_embedding, 
                                          torch.tensor([class_label], device=device) if class_label else None)
                    else:
                        pred_logits = model(image.float(),
                                          torch.tensor([class_label], device=device) if class_label else None)
                    
                    # Get logits for current pixel
                    pixel_logits = pred_logits[0, c, :, h, w]  # [256]
                    
                    # Sample pixel value
                    pixel_value = sample_pixel(pixel_logits, temperature)
                    image[0, c, h, w] = pixel_value
    
    # Convert to numpy
    image_np = image[0].cpu().numpy().transpose(1, 2, 0)  # [H, W, C]
    
    return image_np.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description='Generate images with trained model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./generated',
                       help='Output directory for generated images')
    parser.add_argument('--num_images', type=int, default=10,
                       help='Number of images to generate')
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
    
    # Create model
    model = create_model(
        model_type='frequency_prior',
        in_channels=3,
        num_layers=getattr(model_args, 'num_layers', 8),
        hidden_dim=getattr(model_args, 'hidden_dim', 256),
        num_classes=getattr(model_args, 'num_classes', None),
        use_prompt_conditioning=getattr(model_args, 'use_prompt', False),
        prompt_embed_dim=getattr(model_args, 'prompt_embed_dim', 512)
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate images
    prompt_embedding = None
    if getattr(model_args, 'use_prompt', False):
        # For prompt-based generation, create dummy prompt embedding
        prompt_embedding = torch.randn(1, getattr(model_args, 'prompt_embed_dim', 512)).to(device)
    
    print(f"Generating {args.num_images} images...")
    for i in range(args.num_images):
        seed = args.seed + i if args.seed is not None else None
        image = generate_image(
            model=model,
            image_size=args.image_size,
            num_channels=3,
            device=device,
            class_label=args.class_label,
            prompt_embedding=prompt_embedding,
            temperature=args.temperature,
            seed=seed
        )
        
        # Save image
        image_pil = Image.fromarray(image)
        output_path = os.path.join(args.output_dir, f'generated_{i:04d}.png')
        image_pil.save(output_path)
        print(f"Saved: {output_path}")
    
    print("Generation complete!")


if __name__ == '__main__':
    main()

