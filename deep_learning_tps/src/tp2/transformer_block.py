import torch
import torch.nn as nn

print("="*60)
print("TP2 - PARTIE 4: TRANSFORMER BLOCK")
print("="*60)

# LayerNorm
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

# GELU
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))

# FeedForward
class FeedForward(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            GELU(),
            nn.Linear(4 * emb_dim, emb_dim)
        )
    
    def forward(self, x):
        return self.layers(x)

# MultiHeadAttention (simplifié sans qkv_bias)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, dropout=0.0, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        
        # Register causal mask as buffer
        self.register_buffer('mask', None)
    
    def forward(self, x):
        batch_size, seq_len, d_in = x.shape
        
        # Create causal mask if needed
        if self.mask is None or self.mask.size(0) != seq_len:
            self.mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            self.mask = self.mask.to(x.device)
        
        # Linear projections
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        
        # Reshape for multi-head: [batch, seq, d_out] -> [batch, seq, heads, head_dim]
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim)
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose: [batch, heads, seq, head_dim]
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # Attention scores
        attn_scores = queries @ keys.transpose(-2, -1)
        attn_scores = attn_scores / (self.head_dim ** 0.5)
        
        # Apply causal mask
        attn_scores = attn_scores.masked_fill(self.mask, float('-inf'))
        
        # Softmax
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Weighted values
        context = attn_weights @ values  # [batch, heads, seq, head_dim]
        
        # Merge heads: [batch, heads, seq, head_dim] -> [batch, seq, heads, head_dim]
        context = context.transpose(1, 2)
        
        # Concatenate: [batch, seq, d_out]
        context = context.contiguous().view(batch_size, seq_len, self.d_out)
        
        # Output projection
        output = self.out_proj(context)
        
        return output

# TransformerBlock
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg.get("qkv_bias", False)
        )
        self.ff = FeedForward(cfg["emb_dim"])
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
    
    def forward(self, x):
        # Block 1: Multi-head attention with residual
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Residual connection 1
        
        # Block 2: Feed-forward with residual
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Residual connection 2
        
        return x

print("\n1. Classe TransformerBlock créée")
print("\nArchitecture:")
print("  Input")
print("    ↓")
print("  LayerNorm 1")
print("    ↓")
print("  Multi-Head Attention")
print("    ↓")
print("  Dropout + Residual 1")
print("    ↓")
print("  LayerNorm 2")
print("    ↓")
print("  FeedForward (GELU)")
print("    ↓")
print("  Dropout + Residual 2")
print("    ↓")
print("  Output")

# 2. Test avec config GPT-2
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

print("\n2. Test TransformerBlock:")
torch.manual_seed(123)
block = TransformerBlock(GPT_CONFIG_124M)

test_input = torch.rand(2, 4, 768)  # [batch=2, seq=4, emb=768]
print(f"Input shape:  {test_input.shape}")

output = block(test_input)
print(f"Output shape: {output.shape}")
print(f"\nOutput:\n{output}")

# Compter paramètres
total_params = sum(p.numel() for p in block.parameters())
print(f"\n✓ Paramètres du TransformerBlock: {total_params:,}")

# Détail
print("\nDétail par composant:")
att_params = sum(p.numel() for p in block.att.parameters())
ff_params = sum(p.numel() for p in block.ff.parameters())
norm1_params = sum(p.numel() for p in block.norm1.parameters())
norm2_params = sum(p.numel() for p in block.norm2.parameters())

print(f"  - Multi-Head Attention: {att_params:,}")
print(f"  - FeedForward:          {ff_params:,}")
print(f"  - LayerNorm 1:          {norm1_params:,}")
print(f"  - LayerNorm 2:          {norm2_params:,}")
print(f"  - Total:                {total_params:,}")

print("\n✓ Shape préservée: [2, 4, 768] → [2, 4, 768]")