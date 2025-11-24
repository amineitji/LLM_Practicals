import torch
import tiktoken
from gpt_model import GPTModel

print("="*60)
print("TP2 - PARTIE 12: GÉNÉRATION AVANCÉE")
print("="*60)

# 1. Fonction generate avec temperature et top-k
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """
    Génère du texte avec température et top-k sampling
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        
        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1, :]
        
        # Top-k filtering
        if top_k is not None:
            top_k_values, top_k_indices = torch.topk(logits, top_k)
            logits_filtered = torch.full_like(logits, float('-inf'))
            logits_filtered.scatter_(1, top_k_indices, top_k_values)
            logits = logits_filtered
        
        # Temperature sampling
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        
        # Stop si eos_id
        if eos_id is not None and idx_next.item() == eos_id:
            break
        
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx

print("\n1. Fonction generate créée")
print("\nAméliorations vs generate_text_simple:")
print("  - Temperature sampling (diversité contrôlée)")
print("  - Top-k sampling (évite tokens improbables)")
print("  - Support eos_id (stop automatique)")

# 2. Test génération
print("\n2. Génération avec temperature et top-k:")

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

model = GPTModel(GPT_CONFIG_124M)
try:
    model.load_state_dict(torch.load('models/gpt2_trained.pth'))
    print("✓ Modèle entraîné chargé")
except:
    print("⚠ Modèle entraîné non trouvé, utilisation modèle aléatoire")
    torch.manual_seed(123)

model.eval()

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model.to(device)

tokenizer = tiktoken.get_encoding("gpt2")
start_text = "Every effort moves you"

print(f"\nPrompt: '{start_text}'")
print(f"Paramètres: max_new_tokens=15, top_k=25, temperature=1.4")

# Génération 1
print("\n→ Output text 1:")
torch.manual_seed(123)
token_ids = torch.tensor(tokenizer.encode(start_text)).unsqueeze(0).to(device)
output_ids = generate(
    model=model,
    idx=token_ids,
    max_new_tokens=15,
    context_size=GPT_CONFIG_124M["context_length"],
    temperature=1.4,
    top_k=25
)
output_text = tokenizer.decode(output_ids.squeeze(0).tolist())
print(f" {output_text}")

# Génération 2
print("\n→ Output text 2:")
torch.manual_seed(456)
token_ids = torch.tensor(tokenizer.encode(start_text)).unsqueeze(0).to(device)
output_ids = generate(
    model=model,
    idx=token_ids,
    max_new_tokens=15,
    context_size=GPT_CONFIG_124M["context_length"],
    temperature=1.4,
    top_k=25
)
output_text = tokenizer.decode(output_ids.squeeze(0).tolist())
print(f" {output_text}")

print("\n✓ Les deux générations sont DIFFÉRENTES (non-déterministe)")
print("  → Temperature > 0 introduit du sampling aléatoire")

# Comparaison températures
print("\n3. Comparaison de températures:")
temps = [0.1, 0.5, 1.0, 1.5, 2.0]

for temp in temps:
    print(f"\n→ Temperature = {temp}:")
    torch.manual_seed(123)
    token_ids = torch.tensor(tokenizer.encode(start_text)).unsqueeze(0).to(device)
    output_ids = generate(
        model=model,
        idx=token_ids,
        max_new_tokens=20,
        context_size=GPT_CONFIG_124M["context_length"],
        temperature=temp,
        top_k=25
    )
    output_text = tokenizer.decode(output_ids.squeeze(0).tolist())
    print(f"   {output_text}")

print("\n✓ Observations:")
print("  - Temp faible (0.1): Texte répétitif, conservatif")
print("  - Temp moyenne (1.0): Équilibre créativité/cohérence")
print("  - Temp élevée (2.0): Très créatif mais peut perdre cohérence")