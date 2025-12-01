import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken
import os
from gpt_model import GPTModel
from generate_advanced import generate

print("="*60)
print("TP2 - PARTIE 9: PRÉ-ENTRAÎNEMENT")
print("="*60)

# --- CONFIGURATION ---
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256, # Réduit pour le TP comme demandé
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

# --- 1. DATASET CLASS ---
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenization complète du texte
        token_ids = tokenizer.encode(txt)

        # Création des fenêtres glissantes (Sliding windows)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

# --- 2. FONCTIONS UTILITAIRES DE LOSS ---
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    # Aplatir pour cross_entropy: (batch*seq_len, vocab_size) vs (batch*seq_len)
    loss = nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
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

# --- 3. BOUCLE D'ENTRAÎNEMENT ---
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Textes pour suivre l'évolution
    input_text = start_context
    encoded = tokenizer.encode(input_text)
    token_ids = torch.tensor(encoded).unsqueeze(0).to(device)

    for epoch in range(num_epochs):
        model.train()
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            
            tokens_seen += input_batch.numel()
            global_step += 1

            # Evaluation périodique
            if global_step % eval_freq == 0:
                model.eval()
                with torch.no_grad():
                    train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
                    val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
                
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                model.train() # Retour en mode train
        
        # Génération fin d'époque pour voir le progrès
        model.eval()
        print(f"\n--- Génération fin Epoch {epoch+1} ---")
        with torch.no_grad():
             # On utilise la fonction generate que vous avez déjà dans generate_advanced.py
             # On génère un peu moins de tokens pour que ce soit rapide
            gen_ids = generate(model, token_ids, max_new_tokens=20, context_size=GPT_CONFIG_124M['context_length'])
            decoded_text = tokenizer.decode(gen_ids.squeeze(0).tolist())
            print(f"Prompt: '{input_text}' -> ...{decoded_text[len(input_text):]}\n")
            
    return train_losses, val_losses, track_tokens_seen

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Setup Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device utilisé: {device}")

    # 2. Chargement données
    tokenizer = tiktoken.get_encoding("gpt2")
    file_path = "the-verdict.txt" # Assurez-vous d'avoir ce fichier !
    
    if not os.path.exists(file_path):
        print(f"⚠ Fichier {file_path} introuvable. Téléchargement d'un exemple...")
        import urllib.request
        url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
        urllib.request.urlretrieve(url, file_path)
    
    with open(file_path, "r", encoding="utf-8") as f:
        text_data = f.read()

    # Split Train/Val
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    print(f"Train data len: {len(train_data)}")
    print(f"Val data len:   {len(val_data)}")

    # DataLoaders
    batch_size = 2
    context_length = GPT_CONFIG_124M["context_length"]
    
    train_dataset = GPTDatasetV1(train_data, tokenizer, max_length=context_length, stride=context_length)
    val_dataset = GPTDatasetV1(val_data, tokenizer, max_length=context_length, stride=context_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)

    # 3. Modèle et Optimizer
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

    # 4. Lancement Entraînement
    print("\nDémarrage de l'entraînement...")
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=10, eval_freq=5, eval_iter=5,
        start_context="Every effort moves you", tokenizer=tokenizer
    )

    # 5. Sauvegarde
    print("\nSauvegarde du modèle...")
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/gpt2_trained.pth")
    print("✓ Modèle sauvegardé dans models/gpt2_trained.pth")