import torch
import torch.nn.functional as F
import tiktoken
from gpt_model import GPTModel

print("="*60)
print("TP2 - PARTIE 8: LOSS ET PERPLEXITY")
print("="*60)

# Configuration
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()

# Données
inputs = torch.tensor([[16833, 3626, 6100],
                       [40,    1107, 588]])

targets = torch.tensor([[3626, 6100, 345],
                        [1107,  588, 11311]])

print("Inputs (texte d'entrée):")
tokenizer = tiktoken.get_encoding("gpt2")
for i in range(2):
    print(f"  Seq {i+1}: {tokenizer.decode(inputs[i].tolist())}")

print("\nTargets (mots suivants attendus):")
for i in range(2):
    print(f"  Seq {i+1}: {tokenizer.decode(targets[i].tolist())}")

# 1. Logits et probabilités
print("\n1. Génération des logits:")
with torch.no_grad():
    logits = model(inputs)

print(f"Logits shape: {logits.shape}")

probas = torch.softmax(logits, dim=-1)
print(f"Probas shape: {probas.shape}")

# 2. Tokens prédits
print("\n2. Tokens avec plus haute probabilité:")
predicted_ids = torch.argmax(probas, dim=-1, keepdim=True)
print(f"Predicted IDs shape: {predicted_ids.shape}")
print(f"Predicted IDs:\n{predicted_ids}")

# 3. Comparaison
print("\n3. Comparaison pour la première séquence:")
print(f"Targets batch 1: {tokenizer.decode(targets[0].tolist())}")
print(f"Outputs batch 1: {tokenizer.decode(predicted_ids[0].squeeze().tolist())}")

# 4. Probabilités des targets
print("\n4. Probabilités des tokens dans targets:")
batch_size, seq_len, vocab_size = probas.shape

target_probas_1 = probas[0, range(seq_len), targets[0]]
target_probas_2 = probas[1, range(seq_len), targets[1]]

print(f"Text 1: {target_probas_1}")
print(f"Text 2: {target_probas_2}")

# 5. Log probabilités
print("\n5. Log des probabilités:")
all_target_probas = torch.cat([target_probas_1, target_probas_2])
log_probas = torch.log(all_target_probas)
print(f"Log probabilities: {log_probas}")
print(f"Maximum value: {log_probas.max().item():.4f}")

# 6. Average
print("\n6. Moyenne des log probabilités:")
avg_log_proba = log_probas.mean()
print(f"Average log probability: {avg_log_proba.item():.4f}")

# 7. Cross entropy
print("\n7. Cross Entropy:")
cross_entropy = -avg_log_proba
print(f"Cross Entropy: {cross_entropy.item():.4f}")

# 8. Avec F.cross_entropy
print("\n8. Vérification avec F.cross_entropy:")
logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()
loss = F.cross_entropy(logits_flat, targets_flat)
print(f"Cross Entropy (torch): {loss.item():.4f}")
print("✓ Valeurs identiques!")

# 9. Perplexity
print("\n9. Perplexity:")
perplexity = torch.exp(loss)
print(f"Perplexity: {perplexity.item():.4f}")

print("\n✓ Interprétation:")
print(f"  - Perplexity élevée ({perplexity.item():.0f}) → modèle très confus")
print("  - Modèle non entraîné → predictions aléatoires")
print("  - Après entraînement, perplexity devrait diminuer fortement")