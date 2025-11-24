import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

print("="*60)
print("TP2 - PARTIE 1: LAYER NORMALIZATION")
print("="*60)

torch.manual_seed(123)

# Batch example
batch_example = torch.randn(2, 5)
print("\nBatch example (2 samples, 5 features):")
print(batch_example)

# 1. Linear layer avec ReLU
print("\n1. Passage dans Linear layer + ReLU:")
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
output = layer(batch_example)
print(f"Output shape: {output.shape}")
print(f"Output:\n{output}")

# 2. Mean et variance par exemple
print("\n2. Mean et variance pour chaque exemple:")
mean = output.mean(dim=-1, keepdim=True)
var = output.var(dim=-1, keepdim=True)
print(f"Mean shape: {mean.shape}")
print(f"Mean:\n{mean}")
print(f"Variance:\n{var}")

# 3. Normalisation manuelle
print("\n3. Normalisation manuelle:")
normalized = (output - mean) / torch.sqrt(var)
print(f"Normalized layer outputs:\n{normalized}")
print(f"\nMean après normalisation:\n{normalized.mean(dim=-1)}")
print(f"Variance après normalisation:\n{normalized.var(dim=-1, unbiased=False)}")

# 4. LayerNorm class
print("\n4. Classe LayerNorm:")

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

# 5. Test LayerNorm
print("\n5. Test LayerNorm avec batch_example:")
ln = LayerNorm(emb_dim=5)
normalized_ln = ln(batch_example)
print(f"Output:\n{normalized_ln}")
print(f"\nMean: {normalized_ln.mean(dim=-1)}")
print(f"Var: {normalized_ln.var(dim=-1, unbiased=False)}")

print("\n✓ Pourquoi identique à step 3?")
print("  Scale=1 et shift=0 initialement → pas de transformation supplémentaire")

# Visualisation
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Avant normalisation
sns.heatmap(output.detach().numpy(), annot=True, fmt='.2f', cmap='RdBu_r', 
            center=0, ax=axes[0], cbar_kws={'label': 'Value'})
axes[0].set_title('Avant LayerNorm', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Features')
axes[0].set_ylabel('Samples')

# Après normalisation
sns.heatmap(normalized.detach().numpy(), annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, ax=axes[1], cbar_kws={'label': 'Value'})
axes[1].set_title('Après LayerNorm', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Features')
axes[1].set_ylabel('Samples')

# Distribution
axes[2].hist(output.detach().flatten().numpy(), bins=20, alpha=0.5, 
             label='Avant', color='red')
axes[2].hist(normalized.detach().flatten().numpy(), bins=20, alpha=0.5,
             label='Après', color='blue')
axes[2].set_xlabel('Valeur', fontsize=11)
axes[2].set_ylabel('Fréquence', fontsize=11)
axes[2].set_title('Distribution des valeurs', fontsize=13, fontweight='bold')
axes[2].legend()
axes[2].axvline(0, color='black', linestyle='--', linewidth=1)

plt.tight_layout()
plt.savefig('outputs/tp2_layernorm.png', dpi=150, bbox_inches='tight')
print("\n✓ Visualisation sauvegardée: outputs/tp2_layernorm.png")
plt.close()