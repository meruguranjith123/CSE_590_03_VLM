"""
Vector Quantized Variational Auto Encoder (VQ-VAE) Model with Dual Codebooks
Implements:
- Frequency Prior Codebook (VQ-VAE): Vector quantized codebook for frequency-domain features
- Pixel Codebook (VQ-VAE): Vector quantized codebook for pixel-level spatial features
- Reconstruction Decoder: Decodes from both quantized codebooks to reconstruct images

Architecture Flow:
1. Input Image → Frequency Encoder → Frequency Features
2. Input Image → Pixel Encoder → Pixel Features
3. Frequency Features → VQ Codebook (Frequency) → Quantized Frequency Features
4. Pixel Features → VQ Codebook (Pixel) → Quantized Pixel Features
5. [Quantized Frequency, Quantized Pixel] → Reconstruction Decoder → Reconstructed Image
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class VectorQuantizer(nn.Module):
    """
    Vector Quantization (VQ) module using EMA (Exponential Moving Average) updates.
    This is the core component of VQ-VAE, used to create discrete codebook representations.
    
    We use TWO separate VectorQuantizers:
    1. Frequency Codebook: Quantizes frequency-domain features
    2. Pixel Codebook: Quantizes pixel-domain features
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25, decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps
        
        # Initialize codebook embeddings
        embed = torch.randn(embedding_dim, num_embeddings)
        self.register_buffer('embedding', embed)
        self.register_buffer('cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('embedding_avg', embed.clone())
        
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: Input tensor to quantize [B, D, H, W] or [B, D, N] where D is embedding_dim
        Returns:
            quantized: Quantized vectors
            vq_loss: Vector quantization loss
            encodings: One-hot encodings
        """
        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)  # [B*H*W, D] or [B*N, D]
        
        # Calculate distances
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True) +
            torch.sum(self.embedding ** 2, dim=0) -
            2 * torch.matmul(flat_input, self.embedding)
        )  # [B*H*W, num_embeddings]
        
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # [B*H*W, 1]
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize
        quantized = torch.matmul(encodings, self.embedding.t())  # [B*H*W, D]
        quantized = quantized.view(inputs.shape)  # [B, D, H, W] or [B, D, N]
        
        # Use EMA to update codebook during training
        if self.training:
            self.cluster_size.data.mul_(self.decay).add_(
                encodings.sum(0).mul_(1 - self.decay)
            )
            
            n = self.cluster_size.sum()
            self.cluster_size.data = (
                (self.cluster_size + self.eps) / (n + self.num_embeddings * self.eps) * n
            )
            
            embed_sum = flat_input.t() @ encodings  # [D, num_embeddings]
            self.embedding_avg.data.mul_(self.decay).add_(embed_sum.t().mul_(1 - self.decay))
            
            n = self.cluster_size.sum()
            self.embedding.data.copy_(
                self.embedding_avg / self.cluster_size.unsqueeze(0)
            )
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, vq_loss, encodings


class FrequencyEncoder(nn.Module):
    """
    Encoder that extracts frequency-domain features using FFT (Fast Fourier Transform).
    Computes FFT on the image to get frequency domain representation, then extracts
    frequency bands and projects to codebook dimension.
    """
    def __init__(self, in_channels: int = 3, hidden_dim: int = 256, codebook_dim: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.codebook_dim = codebook_dim
        
        # Initial feature extraction (before FFT) - optional, can be removed
        # We'll use FFT directly on the image
        self.use_pre_fft = False  # Set to False to use only FFT features
        if self.use_pre_fft:
            self.pre_fft_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim // 2, kernel_size=3, padding=1),
                nn.GroupNorm(8, hidden_dim // 2),
                nn.ReLU(inplace=True),
            )
        
        # Process FFT features (FFT output has 3*in_channels channels: magnitude, phase_sin, phase_cos)
        fft_output_channels = 3 * in_channels
        self.fft_processor = nn.Sequential(
            nn.Conv2d(fft_output_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Downsample to match codebook resolution
        self.downsample = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),  # H -> H//2
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),  # H//2 -> H//4
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Frequency domain projection to codebook dimension
        self.freq_proj = nn.Conv2d(hidden_dim, codebook_dim, kernel_size=1)
        
    def compute_fft_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute FFT on input and extract frequency domain features.
        
        Args:
            x: Input tensor [B, C, H, W]
        Returns:
            fft_features: Frequency domain features [B, hidden_dim, H, W]
        """
        B, C, H, W = x.shape
        
        # Compute 2D FFT for each channel
        # FFT returns complex tensor [B, C, H, W] with complex values
        fft_result = torch.fft.rfft2(x, norm='ortho')  # [B, C, H, W//2+1] complex
        
        # Get magnitude (spectral energy) and phase
        magnitude = torch.abs(fft_result)  # [B, C, H, W//2+1]
        phase = torch.angle(fft_result)    # [B, C, H, W//2+1]
        
        # Convert back to real space by padding and taking IFFT (for visualization/processing)
        # Or use magnitude and phase directly
        
        # Stack magnitude and phase as separate channels
        # For simplicity, we'll use magnitude and pad to full resolution
        # Pad the frequency domain to get full H x W resolution
        magnitude_full = F.pad(magnitude, (0, W - (W//2 + 1)), mode='reflect')  # [B, C, H, W]
        
        # Use both magnitude and phase information
        # Convert phase to sin/cos representation for better learning
        phase_sin = torch.sin(phase)
        phase_cos = torch.cos(phase)
        phase_full_sin = F.pad(phase_sin, (0, W - (W//2 + 1)), mode='reflect')  # [B, C, H, W]
        phase_full_cos = F.pad(phase_cos, (0, W - (W//2 + 1)), mode='reflect')  # [B, C, H, W]
        
        # Concatenate magnitude, phase_sin, phase_cos: [B, 3*C, H, W]
        fft_combined = torch.cat([magnitude_full, phase_full_sin, phase_full_cos], dim=1)
        
        return fft_combined
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images [B, C, H, W]
        Returns:
            freq_features: Frequency features [B, codebook_dim, H//4, W//4]
        """
        # Step 1: Compute FFT on the original image to get frequency domain
        # This is the key step - we explicitly compute FFT to get frequency representation
        fft_features = self.compute_fft_features(x)  # [B, 3*C, H, W] (magnitude, phase_sin, phase_cos)
        
        # Step 2: Process FFT features to extract frequency-domain representations
        fft_processed = self.fft_processor(fft_features)  # [B, hidden_dim, H, W]
        
        # Step 3: Optionally combine with pre-FFT spatial features (if enabled)
        if self.use_pre_fft:
            pre_features = self.pre_fft_conv(x)  # [B, hidden_dim//2, H, W]
            if pre_features.shape[2:] != fft_processed.shape[2:]:
                pre_features = F.interpolate(pre_features, size=fft_processed.shape[2:], 
                                           mode='bilinear', align_corners=False)
            
            # Concatenate and project
            combined = torch.cat([pre_features, fft_processed], dim=1)  # [B, hidden_dim*1.5, H, W]
            if not hasattr(self, 'combine_proj'):
                self.combine_proj = nn.Conv2d(combined.shape[1], self.hidden_dim, kernel_size=1)
                self.combine_proj = self.combine_proj.to(combined.device)
            combined = self.combine_proj(combined)  # [B, hidden_dim, H, W]
        else:
            # Use only FFT-processed features
            combined = fft_processed  # [B, hidden_dim, H, W]
        
        # Step 4: Downsample to codebook resolution
        features = self.downsample(combined)  # [B, hidden_dim, H//4, W//4]
        
        # Step 5: Project to frequency codebook dimension
        freq_features = self.freq_proj(features)  # [B, codebook_dim, H//4, W//4]
        
        return freq_features


class PixelEncoder(nn.Module):
    """
    Encoder that extracts pixel-level spatial features
    """
    def __init__(self, in_channels: int = 3, hidden_dim: int = 256, codebook_dim: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.codebook_dim = codebook_dim
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=4, stride=2, padding=1),  # 32x32 -> 16x16
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),  # 16x16 -> 16x16
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, codebook_dim, kernel_size=3, stride=1, padding=1),  # 16x16 -> 16x16
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images [B, C, H, W]
        Returns:
            pixel_features: Pixel features [B, codebook_dim, H//2, W//2]
        """
        pixel_features = self.encoder(x)  # [B, codebook_dim, H//2, W//2]
        return pixel_features


class ReconstructionDecoder(nn.Module):
    """
    Decoder that reconstructs images from quantized frequency and pixel codebooks
    """
    def __init__(self, freq_codebook_dim: int = 64, pixel_codebook_dim: int = 64, 
                 hidden_dim: int = 256, out_channels: int = 3):
        super().__init__()
        self.freq_codebook_dim = freq_codebook_dim
        self.pixel_codebook_dim = pixel_codebook_dim
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        
        # Fusion layer to combine frequency and pixel features
        self.fusion = nn.Sequential(
            nn.Conv2d(freq_codebook_dim + pixel_codebook_dim, hidden_dim, kernel_size=1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Decoder network
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),  # 8x8 -> 16x16
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),  # 16x16 -> 32x32
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Output in [-1, 1]
        )
        
    def forward(self, quantized_freq: torch.Tensor, quantized_pixel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            quantized_freq: Quantized frequency features [B, freq_codebook_dim, H//4, W//4]
            quantized_pixel: Quantized pixel features [B, pixel_codebook_dim, H//2, W//2]
        Returns:
            reconstructed: Reconstructed images [B, C, H, W]
        """
        # Upsample frequency features to match pixel feature resolution
        B, _, H_freq, W_freq = quantized_freq.shape
        _, _, H_pixel, W_pixel = quantized_pixel.shape
        
        # Upsample freq features from H//4 to H//2
        quantized_freq_upsampled = F.interpolate(
            quantized_freq, size=(H_pixel, W_pixel), mode='bilinear', align_corners=False
        )  # [B, freq_codebook_dim, H//2, W//2]
        
        # Concatenate frequency and pixel features
        combined = torch.cat([quantized_freq_upsampled, quantized_pixel], dim=1)  # [B, freq_dim+pixel_dim, H//2, W//2]
        
        # Fuse features
        fused = self.fusion(combined)  # [B, hidden_dim, H//2, W//2]
        
        # Decode to image
        reconstructed = self.decoder(fused)  # [B, C, H, W]
        
        return reconstructed


class FrequencyPriorCodebookModel(nn.Module):
    """
    Complete VQ-VAE model with dual codebooks:
    1. Frequency Prior Codebook (VQ-VAE): Quantizes frequency-domain features
    2. Pixel Codebook (VQ-VAE): Quantizes pixel-level spatial features
    3. Reconstruction Decoder: Reconstructs images from both quantized codebooks
    
    This model implements a Vector Quantized Variational Auto Encoder architecture
    with two separate codebooks for frequency and pixel features.
    """
    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 256,
        freq_codebook_size: int = 512,
        freq_codebook_dim: int = 64,
        pixel_codebook_size: int = 512,
        pixel_codebook_dim: int = 64,
        vq_commitment_cost: float = 0.25,
        vq_decay: float = 0.99
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        
        # Encoders: Extract features for each domain
        self.freq_encoder = FrequencyEncoder(in_channels, hidden_dim, freq_codebook_dim)
        self.pixel_encoder = PixelEncoder(in_channels, hidden_dim, pixel_codebook_dim)
        
        # VQ-VAE Codebooks: Two separate vector quantizers
        # 1. Frequency Codebook (VQ-VAE): Quantizes frequency-domain features
        self.freq_codebook = VectorQuantizer(
            num_embeddings=freq_codebook_size,
            embedding_dim=freq_codebook_dim,
            commitment_cost=vq_commitment_cost,
            decay=vq_decay
        )
        # 2. Pixel Codebook (VQ-VAE): Quantizes pixel-domain features
        self.pixel_codebook = VectorQuantizer(
            num_embeddings=pixel_codebook_size,
            embedding_dim=pixel_codebook_dim,
            commitment_cost=vq_commitment_cost,
            decay=vq_decay
        )
        
        # Reconstruction decoder
        self.decoder = ReconstructionDecoder(
            freq_codebook_dim=freq_codebook_dim,
            pixel_codebook_dim=pixel_codebook_dim,
            hidden_dim=hidden_dim,
            out_channels=in_channels
        )
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode image using VQ-VAE dual codebook architecture.
        
        Process:
        1. Encode image to frequency features using Frequency Encoder
        2. Encode image to pixel features using Pixel Encoder
        3. Quantize frequency features using Frequency Codebook (VQ-VAE)
        4. Quantize pixel features using Pixel Codebook (VQ-VAE)
        
        Args:
            x: Input images [B, C, H, W]
        Returns:
            quantized_freq: Quantized frequency features from Frequency Codebook (VQ-VAE)
            quantized_pixel: Quantized pixel features from Pixel Codebook (VQ-VAE)
            vq_loss_freq: VQ loss for frequency codebook
            vq_loss_pixel: VQ loss for pixel codebook
        """
        # Step 1 & 2: Encode to features using separate encoders
        freq_features = self.freq_encoder(x)  # Frequency domain features
        pixel_features = self.pixel_encoder(x)  # Pixel domain features
        
        # Step 3 & 4: Quantize using VQ-VAE codebooks
        quantized_freq, vq_loss_freq, _ = self.freq_codebook(freq_features)  # Frequency Codebook (VQ-VAE)
        quantized_pixel, vq_loss_pixel, _ = self.pixel_codebook(pixel_features)  # Pixel Codebook (VQ-VAE)
        
        return quantized_freq, quantized_pixel, vq_loss_freq, vq_loss_pixel
    
    def decode(self, quantized_freq: torch.Tensor, quantized_pixel: torch.Tensor) -> torch.Tensor:
        """
        Decode quantized features to reconstructed image
        """
        return self.decoder(quantized_freq, quantized_pixel)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode, quantize, and decode
        Args:
            x: Input images [B, C, H, W] (values in [-1, 1])
        Returns:
            reconstructed: Reconstructed images [B, C, H, W]
            vq_loss_freq: VQ loss for frequency codebook
            vq_loss_pixel: VQ loss for pixel codebook
        """
        # Encode and quantize
        quantized_freq, quantized_pixel, vq_loss_freq, vq_loss_pixel = self.encode(x)
        
        # Decode
        reconstructed = self.decode(quantized_freq, quantized_pixel)
        
        return reconstructed, vq_loss_freq, vq_loss_pixel


def create_vq_codebook_model(
    in_channels: int = 3,
    hidden_dim: int = 256,
    freq_codebook_size: int = 512,
    freq_codebook_dim: int = 64,
    pixel_codebook_size: int = 512,
    pixel_codebook_dim: int = 64,
    vq_commitment_cost: float = 0.25,
    vq_decay: float = 0.99
) -> FrequencyPriorCodebookModel:
    """
    Factory function to create VQ codebook model
    """
    return FrequencyPriorCodebookModel(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        freq_codebook_size=freq_codebook_size,
        freq_codebook_dim=freq_codebook_dim,
        pixel_codebook_size=pixel_codebook_size,
        pixel_codebook_dim=pixel_codebook_dim,
        vq_commitment_cost=vq_commitment_cost,
        vq_decay=vq_decay
    )

