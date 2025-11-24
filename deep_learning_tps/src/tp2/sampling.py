import torch
import matplotlib.pyplot as plt
import numpy as np

print("="*60)
print("TP2 - PARTIE 10-11: TEMPERATURE & TOP-K SAMPLING")
print("="*60)

# Vocabulaire et logits
vocab = { 
    "closer": 0, "every": 1, "effort": 2, "forward": 3,
    "inches": 4, "moves": 5, "pizza": 6, "toward": 7, "you": 8,
}

next_token_logits = torch.tensor(
    [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
)

print("\nVocabulaire:")
for word, idx in vocab.items():
    print(f"  {word}: {idx} (logit: {next_token_logits[idx]:.2f})")

# 1. Probabilités sans température
print("\n1. Probabilités (température = 1.0):")
probas = torch.softmax(next_token_logits, dim=0)
print(probas)

best_idx = torch.argmax(probas)
print(f"\nMot avec plus haute probabilité: '{list(vocab.keys())[best_idx]}' (prob: {probas[best_idx]:.4f})")

# 2. Temperature scaling
print("\n2. Temperature scaling:")

def apply_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)

temp_1 = apply_temperature(next_token_logits, 1.0)
temp_01 = apply_temperature(next_token_logits, 0.1)
temp_5 = apply_temperature(next_token_logits, 5.0)

print(f"Temperature 1.0:  {temp_1}")
print(f"Temperature 0.1:  {temp_01}")
print(f"Temperature 5.0:  {temp_5}")

# 3. Visualisation
print("\n3. Visualisation des températures:")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

words = list(vocab.keys())
x = np.arange(len(words))
width = 0.25

# Graphique 1: Barres groupées
ax1.bar(x - width, temp_01.numpy(), width, label='Temp=0.1 (conservatif)', color='blue', alpha=0.7)
ax1.bar(x, temp_1.numpy(), width, label='Temp=1.0 (normal)', color='green', alpha=0.7)
ax1.bar(x + width, temp_5.numpy(), width, label='Temp=5.0 (créatif)', color='red', alpha=0.7)
ax1.set_xlabel('Mots', fontsize=11)
ax1.set_ylabel('Probabilité', fontsize=11)
ax1.set_title('Impact de la Température sur les Probabilités', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(words, rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Graphique 2: Courbes
temps = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
top_word_probs = []
for temp in temps:
    probs = apply_temperature(next_token_logits, temp)
    top_word_probs.append(probs.max().item())

ax2.plot(temps, top_word_probs, 'bo-', linewidth=2, markersize=8)
ax2.set_xlabel('Temperature', fontsize=11)
ax2.set_ylabel('Probabilité du mot le plus probable', fontsize=11)
ax2.set_title('Température vs Confiance', fontsize=13, fontweight='bold')
ax2.set_xscale('log')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=1/len(vocab), color='r', linestyle='--', 
            label='Uniforme (1/9)', linewidth=1)
ax2.legend()

plt.tight_layout()
plt.savefig('outputs/tp2_temperature.png', dpi=150, bbox_inches='tight')
print("✓ Visualisation sauvegardée: outputs/tp2_temperature.png")
plt.close()

print("\n✓ Observations:")
print("  - Temp 0.1: Distribution très concentrée → toujours le même mot")
print("  - Temp 1.0: Distribution normale")
print("  - Temp 5.0: Distribution aplatie → plus de diversité/créativité")

# Top-k sampling
print("\n" + "="*60)
print("TOP-K SAMPLING")
print("="*60)

# 1. Top-k sampling
print("\n1. Top-k sampling (k=3):")
k = 3

top_k_values, top_k_indices = torch.topk(next_token_logits, k)
print(f"Top-{k} logits: {top_k_values}")
print(f"Top-{k} indices: {top_k_indices}")
print(f"Top-{k} mots: {[list(vocab.keys())[i] for i in top_k_indices]}")

logits_masked = torch.full_like(next_token_logits, float('-inf'))
logits_masked[top_k_indices] = next_token_logits[top_k_indices]

probas_topk = torch.softmax(logits_masked, dim=0)
print(f"\nProbabilités après top-k:\n{probas_topk}")

print("\n✓ Seuls les 3 meilleurs tokens ont une probabilité non-nulle")

# Visualisation comparaison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Sans top-k
ax1.bar(words, probas.numpy(), color='steelblue', alpha=0.7)
ax1.set_xlabel('Mots', fontsize=11)
ax1.set_ylabel('Probabilité', fontsize=11)
ax1.set_title('Sans Top-K (toutes les options)', fontsize=13, fontweight='bold')
ax1.tick_params(axis='x', rotation=45)
ax1.grid(axis='y', alpha=0.3)

# Avec top-k
ax2.bar(words, probas_topk.numpy(), color='coral', alpha=0.7)
ax2.set_xlabel('Mots', fontsize=11)
ax2.set_ylabel('Probabilité', fontsize=11)
ax2.set_title(f'Avec Top-K (k={k})', fontsize=13, fontweight='bold')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/tp2_topk.png', dpi=150, bbox_inches='tight')
print("\n✓ Visualisation sauvegardée: outputs/tp2_topk.png")
plt.close()