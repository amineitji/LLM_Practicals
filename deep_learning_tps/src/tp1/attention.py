import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print("="*60)
print("TP1 - PARTIE 4: ATTENTION MECHANISM")
print("="*60)

torch.manual_seed(123)

# Input sequences
inputs = torch.tensor(
  [[0.43, 0.15, 0.89],  # Your    (x1)
   [0.55, 0.87, 0.66],  # journey (x2)
   [0.57, 0.85, 0.64],  # starts  (x3)
   [0.22, 0.58, 0.33],  # with    (x4)
   [0.77, 0.25, 0.10],  # one     (x5)
   [0.05, 0.80, 0.55]]  # step    (x6)
)

print(f"Input shape: {inputs.shape}")
print(f"Inputs:\n{inputs}")

# 1. Matrices W_query, W_key, W_value
print("\n1. Création des matrices de projection:")
d_in = 3
d_out = 2

W_query = nn.Parameter(torch.rand(d_in, d_out))
W_key = nn.Parameter(torch.rand(d_in, d_out))
W_value = nn.Parameter(torch.rand(d_in, d_out))

print(f"W_query:\n{W_query}")
print(f"W_key:\n{W_key}")
print(f"W_value:\n{W_value}")

# 2. Query pour "journey" (x2)
print("\n2. Query pour 'journey' (x2):")
x2 = inputs[1]
query_2 = x2 @ W_query
print(f"x2: {x2}")
print(f"query_2: {query_2}")

# 3. Keys et Values
print("\n3. Keys et Values pour tous les inputs:")
keys = inputs @ W_key
values = inputs @ W_value
print(f"Keys shape: {keys.shape}")
print(f"Keys:\n{keys}")
print(f"\nValues shape: {values.shape}")
print(f"Values:\n{values}")

# 4. Attention scores
print("\n4. Attention scores:")
attn_scores_2 = query_2 @ keys.T
print(f"Attention scores for x2: {attn_scores_2}")

# 5. Normalisation
print("\n5. Normalisation des scores:")
d_k = keys.shape[-1]
attn_scores_2_normalized = attn_scores_2 / (d_k ** 0.5)
print(f"Scores divisés par sqrt(d_k={d_k}): {attn_scores_2_normalized}")

attn_weights_2 = torch.softmax(attn_scores_2_normalized, dim=0)
print(f"Attention weights (après softmax): {attn_weights_2}")
print(f"Somme: {attn_weights_2.sum().item():.6f}")

# 6. Context vector
print("\n6. Context vector pour 'journey':")
context_vec_2 = attn_weights_2 @ values
print(f"Context vector: {context_vec_2}")

# Visualisation mécanisme attention
print("\n7. Visualisation du mécanisme d'attention:")
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

words = ['Your', 'journey', 'starts', 'with', 'one', 'step']

# Attention scores heatmap
ax1 = fig.add_subplot(gs[0, :2])
scores_matrix = torch.zeros(6, 6)
with torch.no_grad():
    W_query_detached = W_query.detach()
    W_key_detached = W_key.detach()
    W_value_detached = W_value.detach()
    
    for i in range(6):
        query_i = inputs[i] @ W_query_detached
        keys_detached = inputs @ W_key_detached
        scores_i = query_i @ keys_detached.T / (d_k ** 0.5)
        scores_matrix[i] = torch.softmax(scores_i, dim=0)

sns.heatmap(scores_matrix.numpy(), annot=True, fmt='.2f', cmap='YlOrRd',
            xticklabels=words, yticklabels=words, ax=ax1, cbar_kws={'label': 'Attention Weight'})
ax1.set_title('Matrice d\'Attention Complète', fontsize=14, fontweight='bold')
ax1.set_xlabel('Keys (vers)', fontsize=11)
ax1.set_ylabel('Queries (depuis)', fontsize=11)

# Attention pour "journey"
ax2 = fig.add_subplot(gs[0, 2])
ax2.barh(words, attn_weights_2.detach().numpy(), color='coral')
ax2.set_xlabel('Attention Weight', fontsize=11)
ax2.set_title('Attention de "journey"', fontsize=12, fontweight='bold')
ax2.invert_yaxis()
ax2.grid(axis='x', alpha=0.3)

# Queries
ax3 = fig.add_subplot(gs[1, 0])
queries = inputs @ W_query_detached
sns.heatmap(queries.T.numpy(), cmap='Blues', ax=ax3, cbar_kws={'label': 'Value'}, 
            yticklabels=['Dim 1', 'Dim 2'], xticklabels=words)
ax3.set_ylabel('Dimension', fontsize=11)
ax3.set_title('Queries', fontsize=12, fontweight='bold')

# Keys
ax4 = fig.add_subplot(gs[1, 1])
keys_vis = inputs @ W_key_detached
sns.heatmap(keys_vis.T.numpy(), cmap='Greens', ax=ax4, cbar_kws={'label': 'Value'},
            yticklabels=['Dim 1', 'Dim 2'], xticklabels=words)
ax4.set_ylabel('Dimension', fontsize=11)
ax4.set_title('Keys', fontsize=12, fontweight='bold')

# Values
ax5 = fig.add_subplot(gs[1, 2])
values_vis = inputs @ W_value_detached
sns.heatmap(values_vis.T.numpy(), cmap='Oranges', ax=ax5, cbar_kws={'label': 'Value'},
            yticklabels=['Dim 1', 'Dim 2'], xticklabels=words)
ax5.set_ylabel('Dimension', fontsize=11)
ax5.set_title('Values', fontsize=12, fontweight='bold')

# Context vectors
ax6 = fig.add_subplot(gs[2, :])
context_vecs = scores_matrix @ values_vis
sns.heatmap(context_vecs.T.numpy(), cmap='RdPu', ax=ax6, cbar_kws={'label': 'Value'},
            yticklabels=['Dim 1', 'Dim 2'], xticklabels=words)
ax6.set_ylabel('Dimension', fontsize=11)
ax6.set_title('Context Vectors (après attention)', fontsize=13, fontweight='bold')

plt.savefig('outputs/tp1_attention_mechanism.png', dpi=150, bbox_inches='tight')
print("✓ Visualisation sauvegardée: outputs/tp1_attention_mechanism.png")
plt.close()

print("\n" + "="*60)
print("ATTENTION MECHANISM TERMINÉ")
print("="*60)
