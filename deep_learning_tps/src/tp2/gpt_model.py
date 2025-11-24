import torch
import torch.nn as nn

print("="*60)
print("TP2 - PARTIE 5: GPT MODEL COMPLET")
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

# MultiHeadAttention
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
        
        self.register_buffer('mask', None)
    
    def forward(self, x):
        batch_size, seq_len, d_in = x.shape
        
        if self.mask is None or self.mask.size(0) != seq_len:
            self.mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            self.mask = self.mask.to(x.device)
        
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim)
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        attn_scores = queries @ keys.transpose(-2, -1)
        attn_scores = attn_scores / (self.head_dim ** 0.5)
        attn_scores = attn_scores.masked_fill(self.mask, float('-inf'))
        
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = attn_weights @ values
        context = context.transpose(1, 2)
        context = context.contiguous().view(batch_size, seq_len, self.d_out)
        
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
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        
        return x

# GPTModel
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
    
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

print("\n1. Classe GPTModel créée")
print("\nArchitecture complète:")
print("  Token IDs [batch, seq_len]")
print("    ↓")
print("  Token Embeddings + Position Embeddings")
print("    ↓")
print("  Dropout")
print("    ↓")
print("  12x Transformer Blocks")
print("    ↓")
print("  LayerNorm final")
print("    ↓")
print("  Linear (inverse embedding)")
print("    ↓")
print("  Logits [batch, seq_len, vocab_size]")

# 2. Créer modèle
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

print("\n2. Création du modèle GPT-2 124M:")
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
print("✓ Modèle créé")

# 3. Test avec batch
print("\n3. Test avec batch de token IDs:")
batch = torch.tensor([[6109, 3626, 6100, 345],
                      [6109, 1110, 6622, 257]])
print(f"Input shape: {batch.shape}")

logits = model(batch)
print(f"Output shape: {logits.shape}")
print(f"\nLogits:\n{logits}")

# 4. Nombre de paramètres
print("\n4. Nombre de paramètres:")
total_params = sum(p.numel() for p in model.parameters())
print(f"Total: {total_params:,} paramètres")

# Détail par composant
print("\nDétail:")
tok_emb_params = model.tok_emb.weight.numel()
pos_emb_params = model.pos_emb.weight.numel()
trf_params = sum(p.numel() for p in model.trf_blocks.parameters())
final_norm_params = sum(p.numel() for p in model.final_norm.parameters())
out_head_params = model.out_head.weight.numel()

print(f"  - Token embeddings:     {tok_emb_params:>12,}")
print(f"  - Position embeddings:  {pos_emb_params:>12,}")
print(f"  - Transformer blocks:   {trf_params:>12,}")
print(f"  - Final LayerNorm:      {final_norm_params:>12,}")
print(f"  - Output head:          {out_head_params:>12,}")
print(f"  {'='*40}")
print(f"  Total:                  {total_params:>12,}")

# 5. Pourquoi 124M?
print("\n5. Pourquoi on dit '124M paramètres'?")
shared_params = tok_emb_params
actual_unique = total_params - shared_params
print(f"  - Paramètres totaux: {total_params:,}")
print(f"  - Token embedding partagé: {shared_params:,}")
print(f"  - Paramètres uniques: {actual_unique:,}")
print(f"  - ~124M est une approximation du modèle complet")

# 6. Data type
print("\n6. Type de données des paramètres:")
dtype = next(model.parameters()).dtype
print(f"  dtype: {dtype}")

# 7. Taille en MB
print("\n7. Taille du modèle:")
if dtype == torch.float32:
    bytes_per_param = 4
elif dtype == torch.float16:
    bytes_per_param = 2
else:
    bytes_per_param = 4

total_size_bytes = total_params * bytes_per_param
total_size_mb = total_size_bytes / (1024 * 1024)
print(f"  - Bytes par paramètre: {bytes_per_param}")
print(f"  - Taille totale: {total_size_mb:.2f} MB")
print(f"  - Soit environ {total_size_mb/1024:.2f} GB")