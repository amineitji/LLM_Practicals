import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

print("="*60)
print("TP0 - PARTIE 3: NEURAL NETWORKS")
print("="*60)

# 1. Classe NeuralNetwork
class NeuralNetwork(nn.Module):
    def __init__(self, num_input, num_outputs):
        super(NeuralNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(num_input, 30),
            nn.ReLU(),
            nn.Linear(30, 20),
            nn.ReLU(),
            nn.Linear(20, num_outputs)
        )
    
    def forward(self, x):
        return self.network(x)

print("\n1. Classe NeuralNetwork créée")
print("Architecture:")
print("  - Hidden layer 1: num_input -> 30 (ReLU)")
print("  - Hidden layer 2: 30 -> 20 (ReLU)")
print("  - Output layer: 20 -> num_outputs")

# 3. Nombre de paramètres
model_test = NeuralNetwork(50, 3)
total_params = sum(p.numel() for p in model_test.parameters() if p.requires_grad)
print(f"\n3. Nombre total de paramètres (50->3): {total_params:,}")

# Détail par couche
print("\nDétail par couche:")
for i, (name, param) in enumerate(model_test.named_parameters()):
    print(f"  {name}: {param.numel():,} paramètres, shape {list(param.shape)}")

# 4. Paramètres première couche
print("\n4. Paramètres première hidden layer:")
first_layer = model_test.network[0]
print(f"Weights shape: {first_layer.weight.shape}")
print(f"Bias shape: {first_layer.bias.shape}")
print(f"\nPremiers weights:\n{first_layer.weight[:3, :5]}")
print(f"\nPremiers bias:\n{first_layer.bias[:5]}")

# 5. Forward pass
print("\n5. Forward pass avec input aléatoire:")
torch.manual_seed(42)
X = torch.randn(50)
print(f"Input shape: {X.shape}")

with torch.no_grad():
    output = model_test(X)
print(f"Output shape: {output.shape}")
print(f"Output values: {output}")

# 6. Softmax
print("\n6. Application softmax:")
softmax_output = torch.softmax(output, dim=0)
print(f"Softmax output: {softmax_output}")
print(f"Somme des probabilités: {softmax_output.sum().item():.6f}")

# 7-8. Dataset custom
print("\n7-8. Création Dataset et DataLoader:")

X_train = torch.tensor([
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5]
])
y_train = torch.tensor([0, 0, 0, 1, 1])

X_test = torch.tensor([
    [-0.8, 2.8],
    [2.6, -1.6],
])
y_test = torch.tensor([0, 0])

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

print("\nBatches du train_loader (shuffle=True):")
for i, (batch_X, batch_y) in enumerate(train_loader):
    print(f"  Batch {i+1}: X shape {batch_X.shape}, y shape {batch_y.shape}")
    print(f"    X:\n{batch_X}")
    print(f"    y: {batch_y}")

# 9. Drop_last
print("\n9. Effet de drop_last:")
train_loader_drop = DataLoader(train_dataset, batch_size=2, shuffle=False, drop_last=True)
print(f"Avec drop_last=True: {len(train_loader_drop)} batches (dernier batch incomplet supprimé)")
print(f"Sans drop_last: {len(train_loader)} batches")

# 10-12. Training
print("\n10-12. Training du modèle:")
torch.manual_seed(42)
model = NeuralNetwork(2, 2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

train_losses = []
train_accs = []

for epoch in range(3):
    epoch_loss = 0
    correct = 0
    total = 0
    
    for batch_X, batch_y in train_loader_drop:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)
    
    avg_loss = epoch_loss / len(train_loader_drop)
    accuracy = 100 * correct / total
    train_losses.append(avg_loss)
    train_accs.append(accuracy)
    
    print(f"Epoch {epoch+1}/3: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")

# 11. Prédictions
print("\n11. Prédictions sur train set:")
model.eval()
with torch.no_grad():
    train_outputs = model(X_train)
    _, train_preds = torch.max(train_outputs, 1)
    print(f"True labels: {y_train.tolist()}")
    print(f"Predictions: {train_preds.tolist()}")

# 12. Accuracy
print("\n12. Accuracy finale:")
with torch.no_grad():
    train_outputs = model(X_train)
    _, train_preds = torch.max(train_outputs, 1)
    train_acc = (train_preds == y_train).float().mean().item() * 100
    
    test_outputs = model(X_test)
    _, test_preds = torch.max(test_outputs, 1)
    test_acc = (test_preds == y_test).float().mean().item() * 100
    
print(f"Train Accuracy: {train_acc:.2f}%")
print(f"Test Accuracy: {test_acc:.2f}%")

# Visualisation
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loss curve
axes[0].plot(range(1, 4), train_losses, 'b-o', linewidth=2, markersize=8)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Accuracy curve
axes[1].plot(range(1, 4), train_accs, 'g-o', linewidth=2, markersize=8)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy (%)', fontsize=12)
axes[1].set_title('Training Accuracy', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# Decision boundary
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

with torch.no_grad():
    Z = model(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]))
    _, Z = torch.max(Z, 1)
    Z = Z.reshape(xx.shape)

axes[2].contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
scatter = axes[2].scatter(X_train[:, 0], X_train[:, 1], c=y_train, 
                          cmap='RdYlBu', edgecolors='black', s=100, linewidth=1.5)
axes[2].set_xlabel('Feature 1', fontsize=12)
axes[2].set_ylabel('Feature 2', fontsize=12)
axes[2].set_title('Decision Boundary', fontsize=14, fontweight='bold')
plt.colorbar(scatter, ax=axes[2])

plt.tight_layout()
plt.savefig('outputs/tp0_training.png', dpi=150, bbox_inches='tight')
print("\n✓ Visualisation sauvegardée: outputs/tp0_training.png")
plt.close()

# 13. Sauvegarder modèle
torch.save(model.state_dict(), 'models/model.pth')
print("\n13. Modèle sauvegardé: models/model.pth")

# 14. Charger modèle
print("\n14. Chargement du modèle:")
model_loaded = NeuralNetwork(2, 2)
model_loaded.load_state_dict(torch.load('models/model.pth'))
model_loaded.eval()
print("✓ Modèle chargé avec succès")

with torch.no_grad():
    loaded_outputs = model_loaded(X_train)
    _, loaded_preds = torch.max(loaded_outputs, 1)
    print(f"Prédictions modèle chargé: {loaded_preds.tolist()}")
    print(f"Prédictions originales:     {train_preds.tolist()}")
    print(f"Identiques: {torch.equal(loaded_preds, train_preds)}")