"""
models.py - Basic Transformer Architecture (Encoder-Decoder) from Scratch in PyTorch.
Foundations of Generative AI (ICT4403) - Assignment 3
Implements:
  1. Sinusoidal Positional Encoding
  2. Multi-Head Scaled Dot-Product Attention
  3. Position-Wise Feed-Forward Networks
  4. EncoderLayer and Encoder
  5. DecoderLayer and Decoder
  6. Full Seq2SeqTransformer with Autoregressive Generation & Temperature Sampling
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding:
    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Scaled Dot-Product Attention:
    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k) + Mask) V
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Project and reshape to [batch_size, n_heads, seq_len, d_k]
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 2) Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # Mask shape broadcastable: e.g. [batch_size, 1, 1, seq_len] or [1, 1, seq_len, seq_len]
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 3) Weighted sum and project
        context = torch.matmul(attention_weights, V)
        # Context shape: [batch_size, seq_len, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(context)
        return output, attention_weights


class PositionwiseFeedForward(nn.Module):
    """Two-layer feed-forward network with ReLU activation and dropout."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """Single Transformer Encoder layer with self-attention and FFN with residual connections."""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Multi-Head Self-Attention with residual connection & layer norm
        attn_out, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Positionwise FFN with residual connection & layer norm
        ffn_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class DecoderLayer(nn.Module):
    """
    Single Transformer Decoder layer:
    1. Masked Self-Attention (causal lookahead)
    2. Cross-Attention over Encoder representations
    3. Feed-Forward Network
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        # 1) Masked Self-Attention
        self_attn_out, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_out))

        # 2) Cross-Attention over encoder memory (queries from decoder, keys/values from encoder)
        cross_attn_out, _ = self.cross_attn(x, memory, memory, memory_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))

        # 3) Feed-Forward Network
        ffn_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x


class Encoder(nn.Module):
    """Transformer Encoder composed of an embedding, positional encoding, and N layers."""
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.scale = math.sqrt(d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask=None):
        x = self.embedding(src) * self.scale
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Transformer Decoder composed of embedding, positional encoding, N layers, and head."""
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.scale = math.sqrt(d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        x = self.embedding(tgt) * self.scale
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        x = self.norm(x)
        logits = self.fc_out(x)
        return logits


class Seq2SeqTransformer(nn.Module):
    """
    Complete Basic Transformer (Encoder-Decoder) Architecture for Machine Translation.
    Handles masking, forward sequence prediction with teacher forcing, and autoregressive inference.
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=128, n_layers=2, n_heads=4, d_ff=256, dropout=0.1, pad_idx=0):
        super(Seq2SeqTransformer, self).__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

        self.encoder = Encoder(src_vocab_size, d_model, n_layers, n_heads, d_ff, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, n_layers, n_heads, d_ff, dropout)

    def make_src_mask(self, src):
        # src shape: [batch_size, src_len]
        # returns mask: [batch_size, 1, 1, src_len]
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_tgt_mask(self, tgt):
        # tgt shape: [batch_size, tgt_len]
        batch_size, tgt_len = tgt.size()
        # Padding mask: [batch_size, 1, 1, tgt_len]
        pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        # Subsequent lookahead mask: [1, 1, tgt_len, tgt_len]
        subsequent_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool().unsqueeze(0).unsqueeze(1)
        tgt_mask = pad_mask & subsequent_mask
        return tgt_mask

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        
        # Cross-attention mask from decoder to encoder
        memory_mask = src_mask

        memory = self.encoder(src, src_mask)
        logits = self.decoder(tgt, memory, tgt_mask, memory_mask)
        return logits

    @torch.no_grad()
    def translate_sentence(self, src_tensor, src_vocab, tgt_vocab, max_len=35, temperature=1.0, sos_idx=1, eos_idx=2):
        """
        Autoregressive generation for a single source sentence.
        Supports greedy decoding (temperature <= 0.2) or temperature sampling.
        """
        self.eval()
        device = next(self.parameters()).device
        
        if src_tensor.dim() == 1:
            src_tensor = src_tensor.unsqueeze(0).to(device)
        else:
            src_tensor = src_tensor.to(device)

        src_mask = self.make_src_mask(src_tensor)
        memory = self.encoder(src_tensor, src_mask)

        tgt_indices = [sos_idx]

        for _ in range(max_len):
            tgt_tensor = torch.tensor(tgt_indices, dtype=torch.long, device=device).unsqueeze(0)
            tgt_mask = self.make_tgt_mask(tgt_tensor)
            logits = self.decoder(tgt_tensor, memory, tgt_mask, src_mask)
            
            # Take logits of the last token
            next_token_logits = logits[0, -1, :]

            if temperature <= 0.2:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits).item()
            else:
                # Scaled temperature sampling
                scaled_logits = next_token_logits / max(temperature, 1e-4)
                probs = F.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

            if next_token == eos_idx:
                break
            tgt_indices.append(next_token)

        translated_text = tgt_vocab.decode(tgt_indices[1:], remove_special=True)
        return translated_text

def count_parameters(model):
    """Count trainable and total parameters in the model."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
