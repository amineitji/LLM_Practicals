import torch
import torch.nn as nn
import numpy as np
import tiktoken
import os
import sys

# --- GESTION DES IMPORTS ROBUSTE ---
# On ajoute le dossier courant (src/tp2) au sys.path pour trouver gpt_download.py
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

print("="*60)
print("TP2 - PARTIE 14: CHARGER POIDS OPENAI GPT-2")
print("="*60)

try:
    # On importe tensorflow ici juste pour vérifier qu'il est installé
    import tensorflow as tf
    # On importe le script de téléchargement
    import gpt_download
except ImportError as e:
    print("\n❌ ERREUR CRITIQUE : Manque de dépendances")
    print(f"Détail : {e}")
    print("👉 Vous devez installer tensorflow pour lire les poids d'OpenAI.")
    print("👉 Commande : pip install tensorflow")
    sys.exit(1)

from gpt_model import GPTModel
from generate_advanced import generate

# 1. Téléchargement et chargement des poids bruts
print("\n1. Téléchargement des poids OpenAI (124M)...")
model_size = "124M"
# On télécharge dans un dossier 'gpt2_openai' à la racine du projet
# On remonte de deux niveaux depuis src/tp2 pour arriver à la racine
root_dir = os.path.dirname(os.path.dirname(current_dir))
models_dir = os.path.join(root_dir, "gpt2_openai")

try:
    # Appel de la fonction du fichier gpt_download.py
    hparams, params = gpt_download.download_and_load_gpt2(model_size=model_size, models_dir=models_dir)
    print("✓ Poids téléchargés et chargés en mémoire")
except Exception as e:
    print(f"❌ Erreur lors du téléchargement/chargement : {e}")
    sys.exit(1)

# 2. Configuration du modèle
print("\n2. Création de l'architecture GPT-2:")
# NOTE: OpenAI utilise qkv_bias=True, contrairement à notre config par défaut
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024, # OpenAI utilise 1024
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": True  # IMPORTANT !
}

model = GPTModel(GPT_CONFIG_124M)
model.eval()
print("✓ Architecture créée (avec qkv_bias=True)")

# 3. Fonction de transfert des poids
def assign(left_param, right_numpy_array, name="param"):
    """Convertit numpy -> torch et vérifie les dimensions"""
    if right_numpy_array.ndim == 1:
        # Cas des vecteurs (bias, layer norm)
        right_tensor = torch.tensor(right_numpy_array)
    else:
        # Cas des matrices (weights)
        right_tensor = torch.tensor(right_numpy_array)
        
    if left_param.shape != right_tensor.shape:
        raise ValueError(f"Shape mismatch for {name}: {left_param.shape} vs {right_tensor.shape}")
        
    return nn.Parameter(right_tensor)

# 4. Chargement des poids dans le modèle PyTorch
print("\n3. Transfert des poids OpenAI -> PyTorch Model...")

try:
    # a. Embeddings
    model.tok_emb.weight = assign(model.tok_emb.weight, params["wte"], "wte")
    model.pos_emb.weight = assign(model.pos_emb.weight, params["wpe"], "wpe")

    # b. Transformer Blocks
    for b in range(len(model.trf_blocks)):
        block_params = params["blocks"][b]
        pt_block = model.trf_blocks[b]
        
        # 1. Attention - OpenAI stocke q,k,v concaténés dans c_attn
        qkv_w = block_params['attn']['c_attn']['w']
        qkv_b = block_params['attn']['c_attn']['b']
        
        # On doit splitter en 3 parties égales
        q_w, k_w, v_w = np.split(qkv_w, 3, axis=-1)
        q_b, k_b, v_b = np.split(qkv_b, 3, axis=-1)
        
        # Note: Dans PyTorch Linear, les poids sont [out_features, in_features] (transposés)
        pt_block.att.W_query.weight = assign(pt_block.att.W_query.weight, q_w.T, f"h{b}.attn.q_w")
        pt_block.att.W_query.bias   = assign(pt_block.att.W_query.bias,   q_b,   f"h{b}.attn.q_b")
        
        pt_block.att.W_key.weight   = assign(pt_block.att.W_key.weight,   k_w.T, f"h{b}.attn.k_w")
        pt_block.att.W_key.bias     = assign(pt_block.att.W_key.bias,     k_b,   f"h{b}.attn.k_b")
        
        pt_block.att.W_value.weight = assign(pt_block.att.W_value.weight, v_w.T, f"h{b}.attn.v_w")
        pt_block.att.W_value.bias   = assign(pt_block.att.W_value.bias,   v_b,   f"h{b}.attn.v_b")
        
        # 2. Attention Projection (c_proj)
        pt_block.att.out_proj.weight = assign(pt_block.att.out_proj.weight, block_params['attn']['c_proj']['w'].T, f"h{b}.attn.c_proj.w")
        pt_block.att.out_proj.bias   = assign(pt_block.att.out_proj.bias,   block_params['attn']['c_proj']['b'],   f"h{b}.attn.c_proj.b")
        
        # 3. Feed Forward (c_fc -> GELU -> c_proj)
        # Layer 1 (c_fc)
        pt_block.ff.layers[0].weight = assign(pt_block.ff.layers[0].weight, block_params['mlp']['c_fc']['w'].T, f"h{b}.mlp.c_fc.w")
        pt_block.ff.layers[0].bias   = assign(pt_block.ff.layers[0].bias,   block_params['mlp']['c_fc']['b'],   f"h{b}.mlp.c_fc.b")
        
        # Layer 2 (c_proj) - index 2 car index 1 est GELU
        pt_block.ff.layers[2].weight = assign(pt_block.ff.layers[2].weight, block_params['mlp']['c_proj']['w'].T, f"h{b}.mlp.c_proj.w")
        pt_block.ff.layers[2].bias   = assign(pt_block.ff.layers[2].bias,   block_params['mlp']['c_proj']['b'],   f"h{b}.mlp.c_proj.b")
        
        # 4. Layer Norms
        pt_block.norm1.scale = assign(pt_block.norm1.scale, block_params['ln_1']['g'], f"h{b}.ln_1.g")
        pt_block.norm1.shift = assign(pt_block.norm1.shift, block_params['ln_1']['b'], f"h{b}.ln_1.b")
        pt_block.norm2.scale = assign(pt_block.norm2.scale, block_params['ln_2']['g'], f"h{b}.ln_2.g")
        pt_block.norm2.shift = assign(pt_block.norm2.shift, block_params['ln_2']['b'], f"h{b}.ln_2.b")

    # c. Final Layer Norm
    model.final_norm.scale = assign(model.final_norm.scale, params["ln_f_g"], "ln_f.g")
    model.final_norm.shift = assign(model.final_norm.shift, params["ln_f_b"], "ln_f.b")

    # d. Output Head (Weight Tying)
    model.out_head.weight = assign(model.out_head.weight, params["wte"], "wte (head)")

    print("✓ Tous les poids ont été chargés avec succès!")

except KeyError as e:
    print(f"❌ Erreur de structure des poids (clé manquante) : {e}")
    sys.exit(1)
except ValueError as e:
    print(f"❌ Erreur de dimension des poids : {e}")
    sys.exit(1)

# 5. Test de génération
print("\n4. Test de génération (GPT-2 Officiel):")

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Device: {device}")
model.to(device)

tokenizer = tiktoken.get_encoding("gpt2")
prompts = ["The future of Artificial Intelligence is", "Once upon a time in a"]

for prompt in prompts:
    print(f"\nPrompt: '{prompt}'")
    token_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output_ids = generate(
            model=model,
            idx=token_ids,
            max_new_tokens=30,
            context_size=1024,
            temperature=0.8,
            top_k=50
        )
    
    output_text = tokenizer.decode(output_ids.squeeze(0).tolist())
    # Affichage en couleur pour distinguer
    print(f"GPT-2:  \033[92m{output_text}\033[0m") 
    print("-" * 50)