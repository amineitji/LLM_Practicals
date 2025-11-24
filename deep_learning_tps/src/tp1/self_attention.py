import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

print("="*60)
print("TP1 - PARTIE 5: SELF-ATTENTION CLASS")
print("="*60)

torch.manual_seed(789)

# 1. SelfAttention class
class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key = nn.Linear(d_in, d_out, bias=False)
        self.W_value = nn.Linear(d_in, d_out, bias=False)
        
    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        
        context_vec = attn_weights @ values
        return context_vec

print("\n1. SelfAttention créée")

inputs = torch.tensor(
  [[0.43, 0.15, 0.89],
   [0.55, 0.87, 0.66],
   [0.57, 0.85, 0.64],
   [0.22, 0.58, 0.33],
   [0.77, 0.25, 0.10],
   [0.05, 0.80, 0.55]]
)

# 2. Test SelfAttention
print("\n2. Test SelfAttention:")
sa = SelfAttention(d_in=3, d_out=2)
output = sa(inputs)
print(f"Input shape: {inputs.shape}")
print(f"Output shape: {output.shape}")
print(f"Output:\n{output}")

# 3. Causal mask
print("\n3. Causal Attention avec masque:")

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, dropout=0.0):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key = nn.Linear(d_in, d_out, bias=False)
        self.W_value = nn.Linear(d_in, d_out, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, num_tokens, d_in = x.shape
        
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        attn_scores = queries @ keys.transpose(1, 2)
        
        # Masque causal
        mask = torch.triu(torch.ones(num_tokens, num_tokens), diagonal=1).bool()
        attn_scores.masked_fill_(mask, float('-inf'))
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context_vec = attn_weights @ values
        return context_vec

print("✓ CausalAttention créée avec masque triangulaire supérieur")

# 4. Visualisation du masque
print("\n4. Visualisation du masque causal:")
num_tokens = 6
mask = torch.triu(torch.ones(num_tokens, num_tokens), diagonal=1).bool()
print(f"Masque causal (True = masqué):\n{mask.int()}")

# 5. Test avec batch
torch.manual_seed(123)
print("\n5. Test CausalAttention avec batch:")
batch = torch.stack([inputs, inputs])  # [2, 6, 3]
print(f"Batch shape: {batch.shape}")

ca = CausalAttention(d_in=3, d_out=2, dropout=0.0)
output_causal = ca(batch)
print(f"Output shape: {output_causal.shape}")
print(f"Output:\n{output_causal}")

# Visualisation comparaison
print("\n6. Visualisation Self-Attention vs Causal Attention:")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
words = ['Your', 'journey', 'starts', 'with', 'one', 'step']

# Self-Attention (sans masque)
sa_test = SelfAttention(d_in=3, d_out=2)
with torch.no_grad():
    keys = sa_test.W_key(inputs)
    queries = sa_test.W_query(inputs)
    attn_scores = queries @ keys.T
    attn_weights_self = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

sns.heatmap(attn_weights_self.numpy(), annot=True, fmt='.2f', cmap='Blues',
            xticklabels=words, yticklabels=words, ax=axes[0, 0],
            cbar_kws={'label': 'Attention'})
axes[0, 0].set_title('Self-Attention (sans masque)', fontsize=13, fontweight='bold')
axes[0, 0].set_ylabel('Query', fontsize=11)

# Masque causal
mask_vis = torch.triu(torch.ones(6, 6), diagonal=1)
sns.heatmap(mask_vis.numpy(), annot=False, cmap='Reds', ax=axes[0, 1],
            xticklabels=words, yticklabels=words, cbar_kws={'label': 'Masqué'})
axes[0, 1].set_title('Masque Causal', fontsize=13, fontweight='bold')

# Causal Attention
ca_test = CausalAttention(d_in=3, d_out=2, dropout=0.0)
with torch.no_grad():
    keys_c = ca_test.W_key(inputs.unsqueeze(0))
    queries_c = ca_test.W_query(inputs.unsqueeze(0))
    attn_scores_c = queries_c @ keys_c.transpose(1, 2)
    mask_c = torch.triu(torch.ones(6, 6), diagonal=1).bool()
    attn_scores_c.masked_fill_(mask_c, float('-inf'))
    attn_weights_causal = torch.softmax(attn_scores_c / keys_c.shape[-1]**0.5, dim=-1)

sns.heatmap(attn_weights_causal[0].numpy(), annot=True, fmt='.2f', cmap='Greens',
            xticklabels=words, yticklabels=words, ax=axes[0, 2],
            cbar_kws={'label': 'Attention'})
axes[0, 2].set_title('Causal Attention (avec masque)', fontsize=13, fontweight='bold')

# Attention par position
for i in range(3):
    axes[1, i].bar(words, attn_weights_causal[0, i*2].numpy(), color='steelblue')
    axes[1, i].set_ylabel('Attention Weight', fontsize=10)
    axes[1, i].set_title(f'Attention de "{words[i*2]}"', fontsize=12, fontweight='bold')
    axes[1, i].set_ylim([0, 1])
    axes[1, i].grid(axis='y', alpha=0.3)
    plt.setp(axes[1, i].xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.savefig('outputs/tp1_self_attention.png', dpi=150, bbox_inches='tight')
print("✓ Visualisation sauvegardée: outputs/tp1_self_attention.png")
plt.close()

print("\n" + "="*60)
print("SELF-ATTENTION TERMINÉ")
print("="*60)