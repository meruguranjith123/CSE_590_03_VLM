"""
Quick test script to verify model setup and imports
"""

import torch
import sys

def test_imports():
    """Test if all imports work"""
    print("Testing imports...")
    try:
        from models.frequency_prior_model import create_model, FrequencyPriorAutoregressiveModel, PromptConditionalModel
        from datasets.dataset_loaders import get_dataloader
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_model_creation():
    """Test model creation"""
    print("\nTesting model creation...")
    try:
        from models.frequency_prior_model import create_model
        
        # Test base model
        model = create_model(
            model_type='frequency_prior',
            in_channels=3,
            num_layers=4,
            hidden_dim=128,
            num_classes=100,
            use_prompt_conditioning=False
        )
        print("✓ Base model created successfully")
        
        # Test prompt-conditional model
        model_prompt = create_model(
            model_type='frequency_prior',
            in_channels=3,
            num_layers=4,
            hidden_dim=128,
            num_classes=100,
            use_prompt_conditioning=True,
            prompt_embed_dim=512
        )
        print("✓ Prompt-conditional model created successfully")
        
        return True
    except Exception as e:
        print(f"✗ Model creation error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass():
    """Test forward pass"""
    print("\nTesting forward pass...")
    try:
        from models.frequency_prior_model import create_model
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        model = create_model(
            model_type='frequency_prior',
            in_channels=3,
            num_layers=4,
            hidden_dim=128,
            num_classes=100,
            use_prompt_conditioning=False
        )
        model = model.to(device)
        model.eval()
        
        # Create dummy input
        batch_size = 2
        image_size = 32
        x = torch.randint(0, 256, (batch_size, 3, image_size, image_size)).float().to(device)
        labels = torch.randint(0, 100, (batch_size,)).to(device)
        
        # Forward pass
        with torch.no_grad():
            output = model(x, labels)
        
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Expected output shape: ({batch_size}, 3, 256, {image_size}, {image_size})")
        
        assert output.shape == (batch_size, 3, 256, image_size, image_size), \
            f"Output shape mismatch: {output.shape}"
        
        # Test prompt-conditional forward pass
        model_prompt = create_model(
            model_type='frequency_prior',
            in_channels=3,
            num_layers=4,
            hidden_dim=128,
            num_classes=100,
            use_prompt_conditioning=True,
            prompt_embed_dim=512
        )
        model_prompt = model_prompt.to(device)
        model_prompt.eval()
        
        prompt_embeds = torch.randn(batch_size, 512).to(device)
        with torch.no_grad():
            output_prompt = model_prompt(x, prompt_embeds, labels)
        
        print(f"✓ Prompt-conditional forward pass successful")
        print(f"  Output shape: {output_prompt.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Forward pass error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_loaders():
    """Test dataset loaders (without actually downloading)"""
    print("\nTesting dataset loader imports...")
    try:
        from datasets.dataset_loaders import (
            CIFAR100Dataset, ImageNet10kDataset, COCODataset,
            get_transform, get_dataloader
        )
        print("✓ Dataset loader imports successful")
        print("  (Note: Actual dataset loading requires data files)")
        return True
    except Exception as e:
        print(f"✗ Dataset loader import error: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Repository Setup")
    print("=" * 60)
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("Model Creation", test_model_creation()))
    results.append(("Forward Pass", test_forward_pass()))
    results.append(("Dataset Loaders", test_dataset_loaders()))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("✓ All tests passed! Repository is ready for training.")
        sys.exit(0)
    else:
        print("✗ Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == '__main__':
    main()

