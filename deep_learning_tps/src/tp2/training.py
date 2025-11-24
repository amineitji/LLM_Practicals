import torch
import torch.nn.functional as F
import tiktoken
import matplotlib.pyplot as plt
from gpt_model import GPTModel
from text_generation import generate_text_simple

print("="*60)
print("TP2 - PARTIE 9: PRÉ-ENTRAÎNEMENT")
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

# 1. Charger et split les données
print("\n1. Chargement et split des données:")
with open('data/the-verdict.txt', 'r', encoding='utf-8') as f:
    text_data = f.read()

total_chars = len(text_data)
split_idx = int(0.9 * total_chars)

train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

print(f"Total caractères: {total_chars}")
print(f"Train: {len(train_data)} caractères (90%)")
print(f"Val:   {len(val_data)} caractères (10%)")

# 2. Créer dataloaders (simple version)
print("\n2. Création des dataloaders:")

class GPTDataset(torch.utils.data.Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        
        token_ids = tokenizer.encode(txt, allowed_special={'<|endoftext|>'})
        
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

tokenizer = tiktoken.get_encoding("gpt2")

train_dataset = GPTDataset(
    train_data, 
    tokenizer, 
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"]
)

val_dataset = GPTDataset(
    val_data,
    tokenizer,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"]
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    drop_last=True
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False,
    drop_last=False
)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")

# 3. Taille des batches
print("\n3. Taille des batches:")
if len(train_loader) > 0:
    train_batch_x, train_batch_y = next(iter(train_loader))
    print(f"Batch shape: {train_batch_x.shape}")
    print(f"Batch size: 2 séquences de 256 tokens")

# 4. Nombre total de tokens
print("\n4. Nombre de tokens:")
train_tokens = len(train_dataset) * GPT_CONFIG_124M["context_length"]
val_tokens = len(val_dataset) * GPT_CONFIG_124M["context_length"]
total_tokens = train_tokens + val_tokens

total_tokens_actual = len(tokenizer.encode(text_data))

print(f"Train tokens: {train_tokens:,}")
print(f"Val tokens: {val_tokens:,}")
print(f"Total (dataset): {total_tokens:,}")
print(f"Total (actual): {total_tokens_actual:,}")

# 5. Fonction calc_loss_batch
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

print("\n5. Fonction calc_loss_batch créée")

# 6. Fonction calc_loss_loader
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
    
    return total_loss / num_batches

print("6. Fonction calc_loss_loader créée")

# 7. Déterminer device
print("\n7. Détermination du device:")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Device: {device}")

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
model.eval()

print("\nCalcul des losses initiales (avant entraînement):")
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)

print(f"Train loss: {train_loss:.3f}")
print(f"Val loss: {val_loss:.3f}")

# 8. Fonction train_model_simple
def train_model_simple(model, train_loader, val_loader, optimizer, device, 
                       num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    
    for epoch in range(num_epochs):
        model.train()
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1
            
            # Evaluation
            if global_step % eval_freq == 0:
                train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
                val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
        
        # Génération après chaque epoch
        model.eval()
        context = torch.tensor(tokenizer.encode(start_context)).unsqueeze(0).to(device)
        with torch.no_grad():
            output = generate_text_simple(
                model=model, idx=context, max_new_tokens=50,
                context_size=GPT_CONFIG_124M["context_length"]
            )
        output_text = tokenizer.decode(output.squeeze(0).tolist())
        print(output_text)
        model.train()
    
    return train_losses, val_losses, track_tokens_seen

print("\n8. Fonction train_model_simple créée")

# 9. Entraînement
print("\n9. Lancement de l'entraînement:")
print("="*60)

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

num_epochs = 10
eval_freq = 5
eval_iter = 5
start_context = "Every effort moves you"

train_losses, val_losses, tokens_seen = train_model_simple(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device,
    num_epochs=num_epochs,
    eval_freq=eval_freq,
    eval_iter=eval_iter,
    start_context=start_context,
    tokenizer=tokenizer
)

print("="*60)
print("\n✓ Entraînement terminé!")

# 10. Visualisation
print("\n10. Visualisation des losses:")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss vs epochs
epochs_seen = [i // (len(train_loader) // eval_freq + 1) for i in range(len(train_losses))]
ax1.plot(epochs_seen, train_losses, 'b-o', label='Train loss', linewidth=2, markersize=6)
ax1.plot(epochs_seen, val_losses, 'r-s', label='Val loss', linewidth=2, markersize=6)
ax1.set_xlabel('Epochs', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Loss vs Epochs', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Loss vs tokens
ax2.plot(tokens_seen, train_losses, 'b-o', label='Train loss', linewidth=2, markersize=6)
ax2.plot(tokens_seen, val_losses, 'r-s', label='Val loss', linewidth=2, markersize=6)
ax2.set_xlabel('Tokens seen', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.set_title('Loss vs Tokens Seen', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/tp2_training.png', dpi=150, bbox_inches='tight')
print("✓ Visualisation sauvegardée: outputs/tp2_training.png")
plt.close()

# 11. Test génération
print("\n11. Test de génération après entraînement:")
model.eval()

test_prompts = [
    "Every effort moves you",
    "I had always thought",
    "The picture was"
]

for prompt in test_prompts:
    print(f"\n→ Prompt: '{prompt}'")
    context = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    with torch.no_grad():
        output = generate_text_simple(
            model=model, idx=context, max_new_tokens=30,
            context_size=GPT_CONFIG_124M["context_length"]
        )
    print(f"   {tokenizer.decode(output.squeeze(0).tolist())}")

print("\n✓ Observations:")
print("  - La génération est DÉTERMINISTE (toujours le même output)")
print("  - Le modèle utilise argmax → prend toujours le token le plus probable")
print("  - Prochaine étape: ajouter du sampling pour plus de diversité!")

# Sauvegarder le modèle
torch.save(model.state_dict(), 'models/gpt2_trained.pth')
print("\n✓ Modèle entraîné sauvegardé: models/gpt2_trained.pth")