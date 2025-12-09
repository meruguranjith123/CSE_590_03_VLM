"""
Frequency Prior for Autoregressive Image Generation
Main architecture implementing the mid-term report method
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class FrequencyPrior(nn.Module):
    """
    Frequency Prior module that captures frequency domain statistics
    for improved autoregressive image generation
    """
    def __init__(self, in_channels: int, hidden_dim: int = 256):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        
        # Frequency analysis layers
        self.freq_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.freq_norm = nn.GroupNorm(8, hidden_dim)
        
        # Frequency prior prediction
        self.prior_net = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, in_channels, kernel_size=1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, H, W]
        Returns:
            Frequency prior tensor [B, C, H, W]
        """
        # Apply frequency-domain convolution
        freq_features = self.freq_conv(x)
        freq_features = self.freq_norm(freq_features)
        freq_features = F.relu(freq_features)
        
        # Generate frequency prior
        prior = self.prior_net(freq_features)
        
        return prior


class MaskedConv2d(nn.Module):
    """
    Masked convolution for autoregressive models
    """
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, mask_type: str = 'A'):
        super().__init__()
        self.mask_type = mask_type
        
        # Create mask
        mask = torch.zeros(out_channels, in_channels, kernel_size, kernel_size)
        mask[:, :, :kernel_size//2, :] = 1
        mask[:, :, kernel_size//2, :kernel_size//2] = 1
        
        if mask_type == 'A':
            mask[:, :, kernel_size//2, kernel_size//2] = 0
        
        self.register_buffer('mask', mask)
        
        # Convolution
        self.conv = nn.Conv2d(in_channels, out_channels, 
                             kernel_size=kernel_size, padding=kernel_size//2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.conv.weight.data *= self.mask
        return self.conv(x)


class AutoregressiveBlock(nn.Module):
    """
    Residual block with masked convolutions for autoregressive generation
    """
    def __init__(self, channels: int, mask_type: str = 'B'):
        super().__init__()
        self.net = nn.Sequential(
            MaskedConv2d(channels, channels, mask_type=mask_type),
            nn.GroupNorm(8, channels),
            nn.ReLU(inplace=True),
            MaskedConv2d(channels, channels, mask_type='B'),
            nn.GroupNorm(8, channels)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class FrequencyPriorAutoregressiveModel(nn.Module):
    """
    Main model: Autoregressive Image Generation with Frequency Prior
    Implements the method from the mid-term report
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_layers: int = 8,
        hidden_dim: int = 256,
        num_classes: Optional[int] = None,
        use_frequency_prior: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.use_frequency_prior = use_frequency_prior
        
        # Input projection
        self.input_conv = MaskedConv2d(in_channels, hidden_dim, mask_type='A')
        self.input_norm = nn.GroupNorm(8, hidden_dim)
        
        # Frequency prior module
        if use_frequency_prior:
            self.freq_prior = FrequencyPrior(hidden_dim, hidden_dim)
        
        # Autoregressive blocks
        self.layers = nn.ModuleList([
            AutoregressiveBlock(hidden_dim, mask_type='B')
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_conv = MaskedConv2d(hidden_dim, in_channels * 256, mask_type='B')
        
        # Optional class conditioning
        if num_classes is not None:
            self.class_embed = nn.Embedding(num_classes, hidden_dim)
            
    def forward(
        self, 
        x: torch.Tensor, 
        class_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input images [B, C, H, W] (values in [0, 255] or normalized)
            class_labels: Optional class labels [B] for conditional generation
        Returns:
            Logits for pixel prediction [B, C*256, H, W]
        """
        B, C, H, W = x.shape
        
        # Input embedding
        x_embed = self.input_conv(x)
        x_embed = self.input_norm(x_embed)
        x_embed = F.relu(x_embed)
        
        # Add class conditioning if provided
        if class_labels is not None and hasattr(self, 'class_embed'):
            class_emb = self.class_embed(class_labels)  # [B, hidden_dim]
            x_embed = x_embed + class_emb.view(B, -1, 1, 1)
        
        # Apply frequency prior if enabled
        if self.use_frequency_prior:
            freq_prior = self.freq_prior(x_embed)
            x_embed = x_embed + freq_prior
        
        # Autoregressive layers
        for layer in self.layers:
            x_embed = layer(x_embed)
        
        # Output projection
        output = self.output_conv(x_embed)
        
        # Reshape to [B, C, 256, H, W] for pixel prediction
        output = output.view(B, C, 256, H, W)
        
        return output


class PromptConditionalModel(nn.Module):
    """
    Extension: Prompt-based Conditional Image Generation
    Uses text prompts to guide image generation
    """
    def __init__(
        self,
        base_model: FrequencyPriorAutoregressiveModel,
        prompt_embed_dim: int = 512,
        text_encoder: Optional[nn.Module] = None
    ):
        super().__init__()
        self.base_model = base_model
        self.prompt_embed_dim = prompt_embed_dim
        hidden_dim = base_model.hidden_dim
        
        # Text encoder (can be replaced with CLIP or BERT)
        if text_encoder is None:
            self.text_encoder = nn.Sequential(
                nn.Linear(prompt_embed_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim)
            )
        else:
            self.text_encoder = text_encoder
            
        # Prompt conditioning layers
        self.prompt_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        x: torch.Tensor,
        prompt_embeddings: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input images [B, C, H, W]
            prompt_embeddings: Text prompt embeddings [B, prompt_embed_dim]
            class_labels: Optional class labels [B]
        Returns:
            Logits for pixel prediction [B, C*256, H, W]
        """
        B, C, H, W = x.shape
        
        # Encode prompt
        prompt_features = self.text_encoder(prompt_embeddings)  # [B, hidden_dim]
        prompt_features = self.prompt_proj(prompt_features)
        
        # Get base model features
        x_embed = self.base_model.input_conv(x)
        x_embed = self.base_model.input_norm(x_embed)
        x_embed = F.relu(x_embed)
        
        # Combine prompt with image features
        prompt_expanded = prompt_features.view(B, -1, 1, 1)
        
        # Gated fusion
        combined = torch.cat([
            x_embed.mean(dim=[2, 3]),  # Global image features
            prompt_features
        ], dim=1)  # [B, hidden_dim * 2]
        
        gate_values = self.gate(combined).view(B, -1, 1, 1)
        
        # Apply gated conditioning
        x_embed = x_embed * (1 - gate_values) + x_embed * gate_values + prompt_expanded * gate_values
        
        # Add class conditioning if provided
        if class_labels is not None and hasattr(self.base_model, 'class_embed'):
            class_emb = self.base_model.class_embed(class_labels)
            x_embed = x_embed + class_emb.view(B, -1, 1, 1)
        
        # Apply frequency prior
        if self.base_model.use_frequency_prior:
            freq_prior = self.base_model.freq_prior(x_embed)
            x_embed = x_embed + freq_prior
        
        # Autoregressive layers
        for layer in self.base_model.layers:
            x_embed = layer(x_embed)
        
        # Output projection
        output = self.base_model.output_conv(x_embed)
        output = output.view(B, C, 256, H, W)
        
        return output


def create_model(
    model_type: str = 'frequency_prior',
    in_channels: int = 3,
    num_layers: int = 8,
    hidden_dim: int = 256,
    num_classes: Optional[int] = None,
    use_prompt_conditioning: bool = False,
    prompt_embed_dim: int = 512,
    use_frequency_prior: bool = True
) -> nn.Module:
    """
    Factory function to create model instances
    """
    base_model = FrequencyPriorAutoregressiveModel(
        in_channels=in_channels,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        use_frequency_prior=use_frequency_prior
    )
    
    if use_prompt_conditioning:
        model = PromptConditionalModel(
            base_model=base_model,
            prompt_embed_dim=prompt_embed_dim
        )
    else:
        model = base_model
        
    return model

