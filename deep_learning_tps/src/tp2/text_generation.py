import torch
import tiktoken
from gpt_model import GPTModel

print("="*60)
print("TP2 - PARTIE 6: GÉNÉRATION DE TEXTE (Step 1)")
print("="*60)

# Configuration
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

# 1. Fonction generate_text_simple
def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    Génère du texte token par token (approche greedy)
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        
        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx

print("\n1. Fonction generate_text_simple créée")
print("\nPrincipe:")
print("  1. Calculer logits pour le contexte actuel")
print("  2. Prendre logits du dernier token")
print("  3. Convertir en probabilités (softmax)")
print("  4. Sélectionner token avec max probabilité (argmax)")
print("  5. Ajouter au contexte")
print("  6. Répéter jusqu'à max_new_tokens")

# 2. Test génération
print("\n2. Test de génération:")
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()

tokenizer = tiktoken.get_encoding("gpt2")
start_text = "Hello, I am"
encoded = tokenizer.encode(start_text)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)

print(f"Input: '{start_text}'")
print(f"Encoded: {encoded}")
print(f"Shape: {encoded_tensor.shape}")

print("\nGénération de 6 nouveaux tokens...")
output_ids = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)

# 3. Décoder
print("\n3. Résultat:")
output_text = tokenizer.decode(output_ids.squeeze(0).tolist())
print(f"Output: '{output_text}'")

print("\n✓ Analyse:")
print("  - Le texte généré est probablement incohérent")
print("  - Normal: le modèle n'est PAS entraîné")
print("  - Les poids sont aléatoires (torch.manual_seed(123))")
print("  - C'est déterministe (même seed → même output)")