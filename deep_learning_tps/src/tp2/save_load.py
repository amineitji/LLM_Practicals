import torch
import os
from gpt_model import GPTModel

print("="*60)
print("TP2 - PARTIE 13: SAUVEGARDER/CHARGER MODÈLE")
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

# Charger modèle entraîné
print("\nChargement du modèle entraîné...")
model = GPTModel(GPT_CONFIG_124M)
try:
    model.load_state_dict(torch.load('models/gpt2_trained.pth', weights_only=True), strict=False)
    print("✓ Modèle entraîné chargé")
except:
    print("⚠ Modèle entraîné non trouvé, création d'un nouveau modèle")
    torch.manual_seed(123)

# 1. Sauvegarder state_dict
print("\n1. Sauvegarde du state_dict du modèle:")
save_path_model = 'models/gpt2_model_only.pth'
torch.save(model.state_dict(), save_path_model)

# Vérifier taille
file_size_mb = os.path.getsize(save_path_model) / (1024 * 1024)
print(f"✓ Modèle sauvegardé: {save_path_model}")
print(f"  Taille du fichier: {file_size_mb:.2f} MB")

# Taille théorique
total_params = sum(p.numel() for p in model.parameters())
dtype_size = 4  # float32
expected_size_mb = (total_params * dtype_size) / (1024 * 1024)
print(f"  Taille théorique: {expected_size_mb:.2f} MB")
print(f"  Différence: {abs(file_size_mb - expected_size_mb):.2f} MB (overhead fichier)")

# 2. Charger et vérifier
print("\n2. Test de chargement du modèle:")
model_loaded = GPTModel(GPT_CONFIG_124M)
model_loaded.load_state_dict(torch.load(save_path_model, weights_only=True), strict=False)
model_loaded.eval()
print("✓ Modèle rechargé avec succès")

# Vérifier poids
print("\nVérification des poids:")
params_original = list(model.parameters())
params_loaded = list(model_loaded.parameters())

all_equal = True
for p1, p2 in zip(params_original, params_loaded):
    if not torch.equal(p1, p2):
        all_equal = False
        break

print(f"✓ Poids identiques: {all_equal}")

# Test avec input
test_input = torch.randint(0, 50257, (1, 10))
model.eval()
with torch.no_grad():
    output1 = model(test_input)
    output2 = model_loaded(test_input)

print(f"✓ Outputs identiques: {torch.allclose(output1, output2)}")

# 3. Sauvegarder avec optimizer
print("\n3. Sauvegarde du modèle + optimizer:")
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

# Simuler un step
dummy_input = torch.randint(0, 50257, (2, 256))
dummy_target = torch.randint(0, 50257, (2, 256))
model.train()
output = model(dummy_input)
loss = torch.nn.functional.cross_entropy(
    output.flatten(0, 1), 
    dummy_target.flatten()
)
loss.backward()
optimizer.step()

checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': 10,
    'loss': loss.item(),
}

save_path_checkpoint = 'models/gpt2_checkpoint.pth'
torch.save(checkpoint, save_path_checkpoint)

file_size_checkpoint_mb = os.path.getsize(save_path_checkpoint) / (1024 * 1024)
print(f"✓ Checkpoint sauvegardé: {save_path_checkpoint}")
print(f"  Taille du fichier: {file_size_checkpoint_mb:.2f} MB")
print(f"  Différence vs modèle seul: {file_size_checkpoint_mb - file_size_mb:.2f} MB")
print(f"  → Overhead de l'optimizer (états des gradients)")

# 4. Charger checkpoint
print("\n4. Chargement du checkpoint:")
model_from_checkpoint = GPTModel(GPT_CONFIG_124M)
optimizer_from_checkpoint = torch.optim.AdamW(
    model_from_checkpoint.parameters(), 
    lr=0.0004, 
    weight_decay=0.1
)

checkpoint_loaded = torch.load(save_path_checkpoint, weights_only=False)

# Charger avec strict=False pour ignorer les buffers (mask)
model_from_checkpoint.load_state_dict(checkpoint_loaded['model_state_dict'], strict=False)
optimizer_from_checkpoint.load_state_dict(checkpoint_loaded['optimizer_state_dict'])
epoch = checkpoint_loaded['epoch']
loss_value = checkpoint_loaded['loss']

print(f"✓ Checkpoint chargé")
print(f"  Epoch: {epoch}")
print(f"  Loss: {loss_value:.4f}")

# Vérifier optimizer state
print("\n✓ État de l'optimizer préservé:")
print(f"  Learning rate: {optimizer_from_checkpoint.param_groups[0]['lr']}")
print(f"  Weight decay: {optimizer_from_checkpoint.param_groups[0]['weight_decay']}")

print("\n✓ Avec le checkpoint, on peut:")
print("  - Reprendre l'entraînement exactement où on s'était arrêté")
print("  - L'optimizer a gardé ses moments (Adam)")
print("  - Utile pour entraînements longs avec interruptions")

# Résumé
print("\n" + "="*60)
print("RÉSUMÉ DES FICHIERS SAUVEGARDÉS")
print("="*60)
print(f"\n1. gpt2_trained.pth")
print(f"   - Modèle après entraînement complet")
print(f"   - Taille: {file_size_mb:.2f} MB")
print(f"\n2. gpt2_model_only.pth")
print(f"   - State dict du modèle uniquement")
print(f"   - Taille: {file_size_mb:.2f} MB")
print(f"\n3. gpt2_checkpoint.pth")
print(f"   - Modèle + optimizer + métadonnées")
print(f"   - Taille: {file_size_checkpoint_mb:.2f} MB")

print("\n✓ Note: strict=False ignore les buffers non-paramètres (comme mask)")