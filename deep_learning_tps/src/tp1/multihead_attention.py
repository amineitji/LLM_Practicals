import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

print("="*60)
print("TP1 - PARTIE 6: MULTI-HEAD ATTENTION")
print("="*60)

torch.manual_seed(123)

# 1. MultiHeadAttention class
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, dropout=0.0):
        super().__init__()
        assert d_out % num_heads == 0, "d_out doit être divisible par num_heads"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        
        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key = nn.Linear(d_in, d_out, bias=False)
        self.W_value = nn.Linear(d_in, d_out, bias=False)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, num_tokens, d_in = x.shape
        
        # Projections
        keys = self.W_key(x)      # [B, T, d_out]
        queries = self.W_query(x)  # [B, T, d_out]
        values = self.W_value(x)   # [B, T, d_out]
        
        # Split en plusieurs têtes
        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        
        # Transpose pour attention par tête
        keys = keys.transpose(1, 2)      # [B, num_heads, T, head_dim]
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # Attention scores
        attn_scores = queries @ keys.transpose(2, 3)  # [B, num_heads, T, T]
        
        # Masque causal
        mask = torch.triu(torch.ones(num_tokens, num_tokens), diagonal=1).bool()
        attn_scores.masked_fill_(mask, float('-inf'))
        
        # Attention weights
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Appliquer attention
        context_vec = attn_weights @ values  # [B, num_heads, T, head_dim]
        
        # Concatener les têtes
        context_vec = context_vec.transpose(1, 2)  # [B, T, num_heads, head_dim]
        context_vec = context_vec.contiguous().view(batch_size, num_tokens, self.d_out)
        
        # Projection finale
        context_vec = self.out_proj(context_vec)
        
        return context_vec

print("\n1. MultiHeadAttention créée")
print("Architecture:")
print("  - Projections Q, K, V")
print("  - Split en num_heads têtes")
print("  - Attention parallèle par tête")
print("  - Concatenation des têtes")
print("  - Projection finale")

# 2. Test MultiHeadAttention
inputs = torch.tensor(
  [[0.43, 0.15, 0.89],
   [0.55, 0.87, 0.66],
   [0.57, 0.85, 0.64],
   [0.22, 0.58, 0.33],
   [0.77, 0.25, 0.10],
   [0.05, 0.80, 0.55]]
)

batch = torch.stack([inputs, inputs])  # [2, 6, 3]

print("\n2. Test avec batch:")
print(f"Input shape: {batch.shape}")

mha = MultiHeadAttention(d_in=3, d_out=2, num_heads=2, dropout=0.0)
output = mha(batch)

print(f"Output shape: {output.shape}")
print(f"Output:\n{output}")

# Analyse détaillée
print("\n3. Analyse détaillée de l'architecture:")
total_params = sum(p.numel() for p in mha.parameters())
print(f"Nombre total de paramètres: {total_params}")
print("\nDétail:")
for name, param in mha.named_parameters():
    print(f"  {name}: {param.shape} ({param.numel()} params)")

# Test avec différents nombres de têtes
print("\n4. Test avec différents nombres de têtes:")
for num_heads in [1, 2, 4]:
    if 8 % num_heads == 0:
        mha_test = MultiHeadAttention(d_in=3, d_out=8, num_heads=num_heads, dropout=0.0)
        out_test = mha_test(batch)
        print(f"num_heads={num_heads}: head_dim={8//num_heads}, output shape={out_test.shape}")

# Visualisation Multi-Head Attention
print("\n5. Visualisation Multi-Head Attention:")
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)

words = ['Your', 'journey', 'starts', 'with', 'one', 'step']

# Extraire attention weights par tête
mha_vis = MultiHeadAttention(d_in=3, d_out=4, num_heads=2, dropout=0.0)
with torch.no_grad():
    batch_vis = inputs.unsqueeze(0)
    keys = mha_vis.W_key(batch_vis)
    queries = mha_vis.W_query(batch_vis)
    values = mha_vis.W_value(batch_vis)
    
    # Reshape en têtes
    keys = keys.view(1, 6, 2, 2).transpose(1, 2)
    queries = queries.view(1, 6, 2, 2).transpose(1, 2)
    
    # Attention par tête
    for head in range(2):
        attn_scores = queries[:, head] @ keys[:, head].transpose(1, 2)
        mask = torch.triu(torch.ones(6, 6), diagonal=1).bool()
        attn_scores.masked_fill_(mask, float('-inf'))
        attn_weights = torch.softmax(attn_scores / 2**0.5, dim=-1)
        
        ax = fig.add_subplot(gs[0, head*2:head*2+2])
        sns.heatmap(attn_weights[0].numpy(), annot=True, fmt='.2f', cmap='YlOrRd',
                    xticklabels=words, yticklabels=words, ax=ax,
                    cbar_kws={'label': 'Attention'})
        ax.set_title(f'Tête {head+1}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Key', fontsize=11)
        ax.set_ylabel('Query', fontsize=11)

# Comparaison attention patterns
ax_comp = fig.add_subplot(gs[1, :])
head_comparisons = []
for head in range(2):
    attn_scores = queries[:, head] @ keys[:, head].transpose(1, 2)
    mask = torch.triu(torch.ones(6, 6), diagonal=1).bool()
    attn_scores.masked_fill_(mask, float('-inf'))
    attn_weights = torch.softmax(attn_scores / 2**0.5, dim=-1)
    head_comparisons.append(attn_weights[0, 1].numpy())  # Attention de "journey"

x = range(len(words))
width = 0.35
ax_comp.bar([i - width/2 for i in x], head_comparisons[0], width, label='Tête 1', color='coral')
ax_comp.bar([i + width/2 for i in x], head_comparisons[1], width, label='Tête 2', color='steelblue')
ax_comp.set_xlabel('Token', fontsize=11)
ax_comp.set_ylabel('Attention Weight', fontsize=11)
ax_comp.set_title('Comparaison Attention de "journey" entre têtes', fontsize=13, fontweight='bold')
ax_comp.set_xticks(x)
ax_comp.set_xticklabels(words)
ax_comp.legend()
ax_comp.grid(axis='y', alpha=0.3)

# Architecture diagram
ax_arch = fig.add_subplot(gs[2, :])
ax_arch.axis('off')
architecture_text = """
Architecture Multi-Head Attention:

Input [B, T, d_in]
    ↓
Linear Projections (Q, K, V) → [B, T, d_out]
    ↓
Reshape → [B, T, num_heads, head_dim]
    ↓
Transpose → [B, num_heads, T, head_dim]
    ↓
Attention par tête (parallèle)
    ↓
Transpose → [B, T, num_heads, head_dim]
    ↓
Concat → [B, T, d_out]
    ↓
Output Projection → [B, T, d_out]
"""
ax_arch.text(0.5, 0.5, architecture_text, transform=ax_arch.transAxes,
             fontsize=11, verticalalignment='center', horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
             family='monospace')

plt.savefig('outputs/tp1_multihead_attention.png', dpi=150, bbox_inches='tight')
print("✓ Visualisation sauvegardée: outputs/tp1_multihead_attention.png")
plt.close()

print("\n" + "="*60)
print("MULTI-HEAD ATTENTION TERMINÉ")
print("="*60)
print("\n✓ TP1 COMPLET TERMINÉ")