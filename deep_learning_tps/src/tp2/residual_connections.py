import torch
import torch.nn as nn
import matplotlib.pyplot as plt

print("="*60)
print("TP2 - PARTIE 3: RESIDUAL CONNECTIONS")
print("="*60)

# GELU pour ce fichier
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))

# 1. ExampleDeepNeuralNetwork
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, use_shortcut=False):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(3, 3), GELU()),
            nn.Sequential(nn.Linear(3, 3), GELU()),
            nn.Sequential(nn.Linear(3, 3), GELU()),
            nn.Sequential(nn.Linear(3, 3), GELU()),
            nn.Sequential(nn.Linear(3, 1), GELU())
        ])
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if self.use_shortcut and i < len(self.layers) - 1:
                x = layer(x) + x  # Residual connection
            else:
                x = layer(x)
        return x

print("\n1. Classe ExampleDeepNeuralNetwork créée")

# 2. Test sans shortcut
print("\n2. Test sans residual connections:")
torch.manual_seed(123)
model_no_shortcut = ExampleDeepNeuralNetwork(use_shortcut=False)
input_tensor = torch.tensor([[1., 0., -1.]])
output_no_shortcut = model_no_shortcut(input_tensor)
print(f"Output: {output_no_shortcut}")

# 3. Loss et backward
print("\n3. Calcul de la loss (target=0):")
target = torch.tensor([[0.]])
loss_fn = nn.MSELoss()
loss = loss_fn(output_no_shortcut, target)
print(f"The loss is: {loss}")

print("\nBackward pass...")
loss.backward()

# 4. Gradients par layer
print("\n4. Mean des gradients absolus par layer (SANS shortcut):")
gradients_no_shortcut = {}
for name, param in model_no_shortcut.named_parameters():
    if 'weight' in name and param.grad is not None:
        grad_mean = param.grad.abs().mean().item()
        gradients_no_shortcut[name] = grad_mean
        print(f"{name} has gradient mean of {grad_mean}")

# 5. Avec shortcut
print("\n5. Test AVEC residual connections:")
torch.manual_seed(123)
model_with_shortcut = ExampleDeepNeuralNetwork(use_shortcut=True)
output_with_shortcut = model_with_shortcut(input_tensor)
print(f"Output: {output_with_shortcut}")

loss_with = loss_fn(output_with_shortcut, target)
print(f"The loss is: {loss_with}")

print("\nBackward pass...")
loss_with.backward()

print("\nMean des gradients absolus par layer (AVEC shortcut):")
gradients_with_shortcut = {}
for name, param in model_with_shortcut.named_parameters():
    if 'weight' in name and param.grad is not None:
        grad_mean = param.grad.abs().mean().item()
        gradients_with_shortcut[name] = grad_mean
        print(f"{name} has gradient mean of {grad_mean}")

# Visualisation comparaison
print("\n✓ Comparaison:")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

layers = [f"Layer {i+1}" for i in range(5)]
grads_no = list(gradients_no_shortcut.values())
grads_with = list(gradients_with_shortcut.values())

# Sans shortcut
ax1.bar(layers, grads_no, color='coral', alpha=0.7)
ax1.set_ylabel('Mean Absolute Gradient', fontsize=11)
ax1.set_title('Sans Residual Connections', fontsize=13, fontweight='bold')
ax1.set_ylim([0, max(max(grads_no), max(grads_with)) * 1.1])
ax1.grid(axis='y', alpha=0.3)
for i, v in enumerate(grads_no):
    ax1.text(i, v + 0.0001, f'{v:.5f}', ha='center', fontsize=9)

# Avec shortcut
ax2.bar(layers, grads_with, color='steelblue', alpha=0.7)
ax2.set_ylabel('Mean Absolute Gradient', fontsize=11)
ax2.set_title('Avec Residual Connections', fontsize=13, fontweight='bold')
ax2.set_ylim([0, max(max(grads_no), max(grads_with)) * 1.1])
ax2.grid(axis='y', alpha=0.3)
for i, v in enumerate(grads_with):
    ax2.text(i, v + 0.0001, f'{v:.5f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('outputs/tp2_residual.png', dpi=150, bbox_inches='tight')
print("\n✓ Visualisation sauvegardée: outputs/tp2_residual.png")
plt.close()

print("\n✓ Observations:")
print("  SANS shortcut: Gradient diminue dans les couches profondes (vanishing gradient)")
print("  AVEC shortcut: Gradients plus uniformes → meilleur apprentissage des couches profondes")