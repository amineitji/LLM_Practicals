# src/tp0/gradients.py - Correction ligne 75
import torch
import torch.nn.functional as F
from torch.autograd import grad
import matplotlib.pyplot as plt
import numpy as np

print("="*60)
print("TP0 - PARTIE 2: GRADIENTS")
print("="*60)

# 1. Créer tenseurs
print("\n1. Création des tenseurs:")
y = torch.tensor([1.0])
x_1 = torch.tensor([1.1])
w_1 = torch.tensor([2.2])
b = torch.tensor([0.0])

print(f"y = {y.item()}")
print(f"x_1 = {x_1.item()}")
print(f"w_1 = {w_1.item()}")
print(f"b = {b.item()}")

# 2-3. Forward pass
print("\n2-3. Forward pass:")
z = x_1 * w_1 + b
print(f"z = x_1 * w_1 + b = {z.item()}")

a = torch.sigmoid(z)
print(f"a = sigmoid(z) = {a.item()}")

# 4. Loss
print("\n4. Binary Cross Entropy Loss:")
loss = F.binary_cross_entropy(a, y)
print(f"loss = BCE(a, y) = {loss.item()}")
print(f"Calcul manuel: -[y*log(a) + (1-y)*log(1-a)] = {-(y*torch.log(a) + (1-y)*torch.log(1-a)).item()}")

# 5. Requires_grad
print("\n5. Activation requires_grad:")
w_1 = torch.tensor([2.2], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)
print(f"w_1.requires_grad = {w_1.requires_grad}")
print(f"b.requires_grad = {b.requires_grad}")

# 6. Autograd.grad
print("\n6. Calcul gradients avec torch.autograd.grad:")
z = x_1 * w_1 + b
a = torch.sigmoid(z)
loss = F.binary_cross_entropy(a, y)

grads = grad(loss, [w_1, b], retain_graph=True)
print(f"grad(loss, w_1) = {grads[0].item()}")
print(f"grad(loss, b) = {grads[1].item()}")
print("retain_graph=True permet de recalculer les gradients sans recréer le graphe")

# 7. Avant backward
print("\n7. Gradients avant backward():")
print(f"w_1.grad = {w_1.grad}")
print(f"b.grad = {b.grad}")

# 8. Backward
print("\n8. Calcul gradients avec backward():")
loss.backward()
print(f"w_1.grad = {w_1.grad.item()}")
print(f"b.grad = {b.grad.item()}")

# Visualisation du gradient descent
print("\n9. Visualisation Gradient Descent:")
w_values = np.linspace(0, 4, 100)
losses = []

for w_val in w_values:
    w_temp = torch.tensor([w_val], requires_grad=False, dtype=torch.float32)  # FIX: dtype float32
    z_temp = x_1.float() * w_temp + b.detach().float()  # FIX: forcer float32
    a_temp = torch.sigmoid(z_temp)
    loss_temp = F.binary_cross_entropy(a_temp, y.float())  # FIX: y en float32
    losses.append(loss_temp.item())

plt.figure(figsize=(10, 6))
plt.plot(w_values, losses, 'b-', linewidth=2, label='Loss function')
plt.axvline(x=w_1.item(), color='r', linestyle='--', label=f'w_1 initial = {w_1.item():.2f}')
plt.scatter([w_1.item()], [loss.item()], color='r', s=100, zorder=5)

# Flèche gradient
gradient_direction = -w_1.grad.item()
plt.arrow(w_1.item(), loss.item(), gradient_direction*0.3, 
          -abs(gradient_direction)*0.02, head_width=0.1, head_length=0.01, 
          fc='green', ec='green', linewidth=2, label='Gradient direction')

plt.xlabel('w_1', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Binary Cross Entropy Loss vs w_1', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('outputs/tp0_gradients.png', dpi=150, bbox_inches='tight')
print("✓ Visualisation sauvegardée: outputs/tp0_gradients.png")
plt.close()

print(f"\nGradient w_1: {w_1.grad.item():.4f} (négatif car w_1 doit diminuer)")
print(f"Gradient b: {b.grad.item():.4f}")