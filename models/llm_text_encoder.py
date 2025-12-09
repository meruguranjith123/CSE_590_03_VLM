"""
LLM-based Text Encoder for Prompt-based Image Generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block"""
    def __init__(self, d_model: int, nhead: int = 8, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed-forward
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class LLMTextEncoder(nn.Module):
    """
    Lightweight LLM-based text encoder for prompt conditioning
    Uses transformer architecture similar to GPT/BERT
    """
    def __init__(
        self,
        vocab_size: int = 50257,  # GPT-2 vocab size
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 0
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def create_padding_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Create attention mask for padding tokens"""
        return (token_ids == self.pad_token_id).transpose(0, 1)  # [seq_len, batch_size]
    
    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode text tokens into embeddings
        Args:
            token_ids: [batch_size, seq_len] token IDs
            attention_mask: [batch_size, seq_len] attention mask (1 for valid, 0 for padding)
        Returns:
            Text embeddings [batch_size, d_model]
        """
        batch_size, seq_len = token_ids.shape
        
        # Token embeddings
        token_embeds = self.token_embedding(token_ids)  # [batch_size, seq_len, d_model]
        token_embeds = token_embeds.transpose(0, 1)  # [seq_len, batch_size, d_model]
        
        # Positional encoding
        token_embeds = self.pos_encoder(token_embeds)
        
        # Create padding mask if not provided
        if attention_mask is None:
            src_key_padding_mask = self.create_padding_mask(token_ids)
        else:
            # Invert mask: True for padding (to mask), False for valid tokens
            src_key_padding_mask = ~attention_mask.bool()
            src_key_padding_mask = src_key_padding_mask.transpose(0, 1)  # [seq_len, batch_size]
        
        # Transformer encoding
        encoded = self.transformer(
            token_embeds,
            src_key_padding_mask=src_key_padding_mask
        )  # [seq_len, batch_size, d_model]
        
        # Pool: take mean of non-padded tokens, or use last token
        if attention_mask is not None:
            # Weighted average using attention mask
            mask_expanded = attention_mask.unsqueeze(-1).float()  # [batch_size, seq_len, 1]
            encoded = encoded.transpose(0, 1)  # [batch_size, seq_len, d_model]
            sum_embeds = (encoded * mask_expanded).sum(dim=1)  # [batch_size, d_model]
            mask_sum = mask_expanded.sum(dim=1).clamp(min=1e-9)  # [batch_size, 1]
            pooled = sum_embeds / mask_sum
        else:
            # Take mean of all tokens
            encoded = encoded.transpose(0, 1)  # [batch_size, seq_len, d_model]
            pooled = encoded.mean(dim=1)  # [batch_size, d_model]
        
        # Output projection
        output = self.output_proj(pooled)
        output = self.layer_norm(output)
        
        return output


class SimpleTokenizer:
    """
    Simple tokenizer for text prompts
    Can be replaced with GPT-2 tokenizer or other pretrained tokenizers
    """
    def __init__(self, vocab_size: int = 50257, max_length: int = 512):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.eos_token_id = 2
        self.bos_token_id = 3
        
        # Simple word-based tokenization
        # In practice, use a proper tokenizer like GPT-2's
        self.word_to_id = {}
        self.id_to_word = {}
        self._build_vocab()
    
    def _build_vocab(self):
        """Build basic vocabulary"""
        special_tokens = ['<pad>', '<unk>', '<eos>', '<bos>']
        for i, token in enumerate(special_tokens):
            self.word_to_id[token] = i
            self.id_to_word[i] = token
        
        # Add common words (simplified - use real tokenizer in practice)
        common_words = ['a', 'an', 'the', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 
                       'to', 'for', 'of', 'with', 'by', 'from', 'as', 'be', 'been',
                       'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                       'cat', 'dog', 'bird', 'car', 'house', 'tree', 'sky', 'blue', 'red',
                       'green', 'yellow', 'big', 'small', 'happy', 'sad', 'beautiful']
        
        for word in common_words:
            if len(self.word_to_id) < self.vocab_size:
                idx = len(self.word_to_id)
                self.word_to_id[word] = idx
                self.id_to_word[idx] = word
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs"""
        # Simple word splitting (use proper tokenizer in practice)
        words = text.lower().split()
        
        token_ids = []
        if add_special_tokens:
            token_ids.append(self.bos_token_id)
        
        for word in words:
            word = word.strip('.,!?;:')
            token_id = self.word_to_id.get(word, self.unk_token_id)
            token_ids.append(token_id)
        
        if add_special_tokens and len(token_ids) < self.max_length:
            token_ids.append(self.eos_token_id)
        
        return token_ids[:self.max_length]
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text"""
        words = []
        for token_id in token_ids:
            if token_id in [self.pad_token_id, self.eos_token_id]:
                break
            if token_id in self.id_to_word:
                word = self.id_to_word[token_id]
                if word not in ['<pad>', '<unk>', '<bos>']:
                    words.append(word)
        return ' '.join(words)
    
    def __call__(self, texts: List[str], padding: bool = True, truncation: bool = True) -> dict:
        """
        Tokenize batch of texts
        Returns:
            dict with 'input_ids' and 'attention_mask'
        """
        batch_token_ids = []
        batch_attention_mask = []
        
        for text in texts:
            token_ids = self.encode(text, add_special_tokens=True)
            
            if truncation and len(token_ids) > self.max_length:
                token_ids = token_ids[:self.max_length]
            
            attention_mask = [1] * len(token_ids)
            
            if padding:
                pad_length = self.max_length - len(token_ids)
                token_ids.extend([self.pad_token_id] * pad_length)
                attention_mask.extend([0] * pad_length)
            
            batch_token_ids.append(token_ids)
            batch_attention_mask.append(attention_mask)
        
        return {
            'input_ids': torch.tensor(batch_token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(batch_attention_mask, dtype=torch.long)
        }


class LLMPromptConditionalModel(nn.Module):
    """
    LLM-based Prompt Conditional Image Generation Model
    Uses transformer-based text encoder to condition image generation on text prompts
    """
    def __init__(
        self,
        base_model: nn.Module,  # FrequencyPriorAutoregressiveModel
        vocab_size: int = 50257,
        text_embed_dim: int = 512,
        num_text_layers: int = 6,
        text_nhead: int = 8,
        max_text_length: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.base_model = base_model
        hidden_dim = base_model.hidden_dim
        
        # LLM text encoder
        self.text_encoder = LLMTextEncoder(
            vocab_size=vocab_size,
            d_model=text_embed_dim,
            nhead=text_nhead,
            num_layers=num_text_layers,
            max_seq_len=max_text_length,
            dropout=dropout
        )
        
        # Project text embeddings to model hidden dimension
        self.text_proj = nn.Sequential(
            nn.Linear(text_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Cross-attention for text-image fusion
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=False
        )
        
        # Gated fusion mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input images [B, C, H, W]
            token_ids: Text token IDs [B, seq_len]
            attention_mask: Attention mask [B, seq_len] (1 for valid, 0 for padding)
            class_labels: Optional class labels [B]
        Returns:
            Logits for pixel prediction [B, C*256, H, W]
        """
        B, C, H, W = x.shape
        device = x.device
        
        # Encode text prompt with LLM
        text_embeds = self.text_encoder(token_ids, attention_mask)  # [B, text_embed_dim]
        text_features = self.text_proj(text_embeds)  # [B, hidden_dim]
        
        # Get base model features
        x_embed = self.base_model.input_conv(x)
        x_embed = self.base_model.input_norm(x_embed)
        x_embed = F.relu(x_embed)
        
        # Reshape for cross-attention: [seq_len, batch, hidden_dim]
        # Use spatial features as query, text as key/value
        B, C_embed, H_feat, W_feat = x_embed.shape
        x_flat = x_embed.view(B, C_embed, H_feat * W_feat).permute(2, 0, 1)  # [H*W, B, hidden_dim]
        text_expanded = text_features.unsqueeze(0)  # [1, B, hidden_dim]
        
        # Cross-attention: image queries attend to text keys/values
        attended, _ = self.cross_attn(
            query=x_flat,
            key=text_expanded,
            value=text_expanded
        )  # [H*W, B, hidden_dim]
        
        attended = attended.permute(1, 2, 0).view(B, C_embed, H_feat, W_feat)
        
        # Gated fusion
        x_global = x_embed.mean(dim=[2, 3])  # [B, hidden_dim]
        combined = torch.cat([x_global, text_features], dim=1)  # [B, hidden_dim * 2]
        gate_values = self.gate(combined).view(B, -1, 1, 1)  # [B, hidden_dim, 1, 1]
        
        # Fuse features
        x_embed = x_embed * (1 - gate_values) + attended * gate_values
        x_embed = self.layer_norm(x_embed.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
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


def create_llm_prompt_model(
    base_model: nn.Module,
    vocab_size: int = 50257,
    text_embed_dim: int = 512,
    num_text_layers: int = 6,
    text_nhead: int = 8,
    max_text_length: int = 512
) -> LLMPromptConditionalModel:
    """Factory function to create LLM-based prompt model"""
    return LLMPromptConditionalModel(
        base_model=base_model,
        vocab_size=vocab_size,
        text_embed_dim=text_embed_dim,
        num_text_layers=num_text_layers,
        text_nhead=text_nhead,
        max_text_length=max_text_length
    )

