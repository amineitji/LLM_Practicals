import torch
import tiktoken
from gpt_model import GPTModel
from text_generation import generate_text_simple

print("="*60)
print("TP2 - PARTIE 7: GÉNÉRATION DE TEXTE (Step 2)")
print("="*60)

# Configuration réduite
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

# 1. Créer modèle
print("\n1. Création modèle avec context_length réduit (256):")
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()
print("✓ Modèle créé en mode eval")

# 2. Fonctions encode/decode
tokenizer = tiktoken.get_encoding("gpt2")

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

print("\n2. Fonctions text_to_token_ids et token_ids_to_text créées")

# 3. Génération
print("\n3. Génération de 10 tokens:")
start_context = "Every effort moves you"
print(f"Input: '{start_context}'")

token_ids = text_to_token_ids(start_context, tokenizer)
print(f"Token IDs: {token_ids}")

print("\nGénération en cours...")
output_ids = generate_text_simple(
    model=model,
    idx=token_ids,
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
)

output_text = token_ids_to_text(output_ids, tokenizer)
print(f"\nOutput text:\n {output_text}")

print("\n✓ Observation:")
print("  - Texte généré toujours incohérent (modèle non entraîné)")
print("  - Tokens ressemblent à du bruit aléatoire")
print("  - Prochaine étape: ENTRAÎNER le modèle!")