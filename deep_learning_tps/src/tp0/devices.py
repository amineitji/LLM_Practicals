import torch
import torch.nn as nn
import time

print("="*60)
print("TP0 - PARTIE 4: DEVICES")
print("="*60)

# 1. Vérifier devices disponibles
print("\n1. Devices disponibles:")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version: {torch.version.cuda}")

print(f"MPS available (MacOS): {torch.backends.mps.is_available()}")
print(f"CPU available: {torch.cpu.is_available()}")

# Sélectionner device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"\n✓ Utilisation: CUDA")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"\n✓ Utilisation: MPS (Apple Silicon)")
else:
    device = torch.device("cpu")
    print(f"\n✓ Utilisation: CPU")

print(f"Device sélectionné: {device}")

# 2. Training sur device
print("\n2. Training avec device:")

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 30),
            nn.ReLU(),
            nn.Linear(30, 20),
            nn.ReLU(),
            nn.Linear(20, 2)
        )
    
    def forward(self, x):
        return self.network(x)

# Données
X_train = torch.tensor([
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5]
]).to(device)

y_train = torch.tensor([0, 0, 0, 1, 1]).to(device)

print(f"X_train device: {X_train.device}")
print(f"y_train device: {y_train.device}")

# Modèle
torch.manual_seed(42)
model = SimpleNet().to(device)
print(f"Model device: {next(model.parameters()).device}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Training
print("\nTraining...")
start_time = time.time()

for epoch in range(50):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            _, preds = torch.max(outputs, 1)
            acc = (preds == y_train).float().mean().item() * 100
        print(f"Epoch {epoch+1}/50: Loss = {loss.item():.4f}, Acc = {acc:.2f}%")

training_time = time.time() - start_time
print(f"\nTemps d'entraînement sur {device}: {training_time:.4f}s")

# 3. Comparaison CPU
print("\n3. Comparaison avec CPU:")

if device.type != "cpu":
    torch.manual_seed(42)
    model_cpu = SimpleNet().to("cpu")
    X_train_cpu = X_train.to("cpu")
    y_train_cpu = y_train.to("cpu")
    
    optimizer_cpu = torch.optim.SGD(model_cpu.parameters(), lr=0.1)
    
    print("Training sur CPU...")
    start_time_cpu = time.time()
    
    for epoch in range(50):
        optimizer_cpu.zero_grad()
        outputs_cpu = model_cpu(X_train_cpu)
        loss_cpu = criterion(outputs_cpu, y_train_cpu)
        loss_cpu.backward()
        optimizer_cpu.step()
        
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                _, preds_cpu = torch.max(outputs_cpu, 1)
                acc_cpu = (preds_cpu == y_train_cpu).float().mean().item() * 100
            print(f"Epoch {epoch+1}/50: Loss = {loss_cpu.item():.4f}, Acc = {acc_cpu:.2f}%")
    
    training_time_cpu = time.time() - start_time_cpu
    print(f"\nTemps d'entraînement sur CPU: {training_time_cpu:.4f}s")
    
    speedup = training_time_cpu / training_time
    print(f"\nSpeedup {device} vs CPU: {speedup:.2f}x")
else:
    print("Déjà sur CPU, pas de comparaison nécessaire")

print("\n" + "="*60)
print("TP0 TERMINÉ")
print("="*60)
