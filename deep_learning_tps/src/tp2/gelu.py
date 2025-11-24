import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

print("="*60)
print("TP2 - PARTIE 2: GELU ACTIVATION")
print("="*60)

# 1. GELU class
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))

print("\n1. Classe GELU implémentée")

# 2. Test avec vecteur
print("\n2. Test GELU avec 100 valeurs entre -3 et 3:")
x = torch.linspace(-3, 3, 100)
gelu = GELU()
gelu_output = gelu(x)
print(f"Output:\n{gelu_output}")

# 3. Visualisation GELU
print("\n3. Visualisation GELU:")
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x.numpy(), gelu_output.detach().numpy(), 'b-', linewidth=2, label='GELU')
plt.grid(True, alpha=0.3)
plt.xlabel('x', fontsize=12)
plt.ylabel('GELU(x)', fontsize=12)
plt.title('GELU Activation', fontsize=14, fontweight='bold')
plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
plt.axvline(x=0, color='k', linestyle='--', linewidth=0.5)
plt.legend()

# 4. Comparaison avec ReLU
print("\n4. Comparaison GELU vs ReLU:")
relu = nn.ReLU()
relu_output = relu(x)

plt.subplot(1, 2, 2)
plt.plot(x.numpy(), gelu_output.detach().numpy(), 'b-', linewidth=2, label='GELU')
plt.plot(x.numpy(), relu_output.detach().numpy(), 'r-', linewidth=2, label='ReLU')
plt.grid(True, alpha=0.3)
plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
plt.title('GELU vs ReLU', fontsize=14, fontweight='bold')
plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
plt.axvline(x=0, color='k', linestyle='--', linewidth=0.5)
plt.legend()

plt.tight_layout()
plt.savefig('outputs/tp2_gelu.png', dpi=150, bbox_inches='tight')
print("✓ Visualisation sauvegardée: outputs/tp2_gelu.png")
plt.close()

print("\n✓ Différences GELU vs ReLU:")
print("  - ReLU: f(x) = 0 si x<0, f(x)=x si x>=0 (dur)")
print("  - GELU: transition douce autour de 0 (probabiliste)")
print("  - GELU permet un petit gradient pour valeurs négatives")

# 5. FeedForward class
print("\n5. Classe FeedForward:")

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

# Test
print("\nTest FeedForward (768 dim):")
ff = FeedForward(768)
test_input = torch.randn(2, 4, 768)  # [batch, seq, emb]
test_output = ff(test_input)
print(f"Input shape:  {test_input.shape}")
print(f"Output shape: {test_output.shape}")
print("✓ Shape identique (comme attendu)")

# Compter paramètres
total_params = sum(p.numel() for p in ff.parameters())
print(f"\nParamètres FeedForward: {total_params:,}")
print(f"  - Linear 1: 768 × (4×768) + bias = {768 * 4 * 768 + 4*768:,}")
print(f"  - Linear 2: (4×768) × 768 + bias = {4 * 768 * 768 + 768:,}")