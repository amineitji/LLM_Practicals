import torch
import torch.nn as nn
import tiktoken
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
sys.path.append('src/tp1')
from dataset_dataloader import create_dataloader_v1

print("="*60)
print("TP1 - PARTIE 3: EMBEDDINGS")
print("="*60)

torch.manual_seed(123)

# 1. Embedding layer simple
print("\n1. Embedding layer (vocab_size=6, embed_dim=3):")
vocab_size = 6
embed_dim = 3

embedding_layer = nn.Embedding(vocab_size, embed_dim)
print(f"Embedding weights shape: {embedding_layer.weight.shape}")
print(f"\nWeights:\n{embedding_layer.weight}")

# 2. Token unique
print("\n2. Embedding d'un token (id=0):")
token_id = torch.tensor([0])
embedded = embedding_layer(token_id)
print(f"Token id: {token_id.item()}")
print(f"Embedding: {embedded}")
print(f"Shape: {embedded.shape}")

# 3. Séquence de tokens
print("\n3. Embedding d'une séquence:")
token_ids = torch.tensor([2, 3, 5, 1])
embeddings = embedding_layer(token_ids)
print(f"Token ids: {token_ids.tolist()}")
print(f"Embeddings shape: {embeddings.shape}")
print(f"Embeddings:\n{embeddings}")

# 4. GPT2 token embeddings
print("\n4. GPT2 Token Embeddings:")
vocab_size_gpt2 = 50257
embed_dim_gpt2 = 256

token_embedding_layer = nn.Embedding(vocab_size_gpt2, embed_dim_gpt2)
print(f"Vocab size: {vocab_size_gpt2}")
print(f"Embedding dimension: {embed_dim_gpt2}")
print(f"Total parameters: {vocab_size_gpt2 * embed_dim_gpt2:,}")

# Charger un batch
with open('data/the-verdict.txt', 'r') as f:
    raw_text = f.read()

dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, 
                                  stride=4, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

print(f"\nBatch inputs shape: {inputs.shape}")  # [8, 4]

token_embeddings = token_embedding_layer(inputs)
print(f"Token embeddings shape: {token_embeddings.shape}")  # [8, 4, 256]
print(f"Token embeddings:\n{token_embeddings}")

# 5. Position embeddings
print("\n5. Position Embeddings:")
context_length = 4
pos_embedding_layer = nn.Embedding(context_length, embed_dim_gpt2)
print(f"Context length: {context_length}")
print(f"Position embedding weights shape: {pos_embedding_layer.weight.shape}")

# Créer position ids
pos_ids = torch.arange(context_length)
print(f"Position ids: {pos_ids}")

pos_embeddings = pos_embedding_layer(pos_ids)
print(f"Position embeddings shape: {pos_embeddings.shape}")

# Combiner token et position embeddings
print("\n6. Combinaison Token + Position Embeddings:")
input_embeddings = token_embeddings + pos_embeddings
print(f"Input embeddings shape: {input_embeddings.shape}")  # [8, 4, 256]
print("✓ Token et position embeddings combinés")

# Visualisation
print("\n7. Visualisation des embeddings:")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Token embedding d'une séquence
seq_idx = 0
token_emb_vis = token_embeddings[seq_idx].detach().numpy()
sns.heatmap(token_emb_vis.T, cmap='RdBu_r', center=0, ax=axes[0, 0], 
            cbar_kws={'label': 'Value'})
axes[0, 0].set_xlabel('Token position', fontsize=11)
axes[0, 0].set_ylabel('Embedding dimension', fontsize=11)
axes[0, 0].set_title('Token Embeddings (séquence 1)', fontsize=13, fontweight='bold')

# Position embeddings
pos_emb_vis = pos_embeddings.detach().numpy()
sns.heatmap(pos_emb_vis.T, cmap='RdBu_r', center=0, ax=axes[0, 1],
            cbar_kws={'label': 'Value'})
axes[0, 1].set_xlabel('Position', fontsize=11)
axes[0, 1].set_ylabel('Embedding dimension', fontsize=11)
axes[0, 1].set_title('Position Embeddings', fontsize=13, fontweight='bold')

# Combined embeddings
combined_vis = input_embeddings[seq_idx].detach().numpy()
sns.heatmap(combined_vis.T, cmap='RdBu_r', center=0, ax=axes[1, 0],
            cbar_kws={'label': 'Value'})
axes[1, 0].set_xlabel('Token position', fontsize=11)
axes[1, 0].set_ylabel('Embedding dimension', fontsize=11)
axes[1, 0].set_title('Combined Embeddings (Token + Position)', fontsize=13, fontweight='bold')

# Distribution des valeurs
axes[1, 1].hist(token_emb_vis.flatten(), bins=50, alpha=0.5, label='Token', color='blue')
axes[1, 1].hist(pos_emb_vis.flatten(), bins=50, alpha=0.5, label='Position', color='red')
axes[1, 1].hist(combined_vis.flatten(), bins=50, alpha=0.5, label='Combined', color='green')
axes[1, 1].set_xlabel('Valeur', fontsize=11)
axes[1, 1].set_ylabel('Fréquence', fontsize=11)
axes[1, 1].set_title('Distribution des Valeurs d\'Embedding', fontsize=13, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/tp1_embeddings.png', dpi=150, bbox_inches='tight')
print("✓ Visualisation sauvegardée: outputs/tp1_embeddings.png")
plt.close()

print("\n" + "="*60)
print("EMBEDDINGS TERMINÉ")
print("="*60)
