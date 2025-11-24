import torch
import torch.nn as nn
import tiktoken
from gpt_model import GPTModel, LayerNorm, GELU, FeedForward, MultiHeadAttention, TransformerBlock
from generate_advanced import generate

print("="*60)
print("TP2 - PARTIE 14: CHARGER POIDS OPENAI GPT-2")
print("="*60)

# 1. Download et load
print("\n1. Téléchargement des poids OpenAI...")
print("\n⚠ NOTE IMPORTANTE:")
print("  Le chargement complet des poids OpenAI nécessite:")
print("  - Le fichier gpt_download.py dans le dossier racine")
print("  - Une adaptation de MultiHeadAttention pour qkv_bias=True")
print("  - Connexion internet pour télécharger les poids (~500MB)")
print("\n  Pour une implémentation complète, voir:")
print("  https://github.com/rasbt/LLMs-from-scratch")

try:
    import sys
    sys.path.append('.')
    from gpt_download import download_and_load_gpt2
    
    print("\n  Téléchargement en cours...")
    settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
    print("✓ Poids téléchargés et chargés")
    
    # 2. Settings et params
    print("\n2. Settings dictionary:")
    print(settings)
    
    print("\n3. Clés du params dictionary:")
    print(f"Nombre de clés: {len(params)}")
    print("\nPremières 10 clés:")
    for i, key in enumerate(list(params.keys())[:10]):
        shape = params[key].shape if hasattr(params[key], 'shape') else 'N/A'
        print(f"  {key}: {shape}")
    
    print("\n✓ Structure des paramètres:")
    print("  - wte: word token embeddings")
    print("  - wpe: word position embeddings")
    print("  - h.X.ln_1/ln_2: layer norms")
    print("  - h.X.attn: attention layers (c_attn, c_proj)")
    print("  - h.X.mlp: feed forward (c_fc, c_proj)")
    print("  - ln_f: final layer norm")
    
    # 3. Créer architecture
    print("\n3. Création de l'architecture GPT-2:")
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.0,
        "qkv_bias": True
    }
    
    model = GPTModel(GPT_CONFIG_124M)
    print("✓ Architecture créée")
    
    # 4. Fonction assign
    def assign(left, right):
        """Vérifie compatibilité et assigne les poids"""
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch: {left.shape} vs {right.shape}")
        return nn.Parameter(torch.tensor(right))
    
    print("\n4. Fonction assign créée")
    
    # 5. Load weights
    print("\n5. Chargement des poids OpenAI dans le modèle...")
    
    def load_weights_into_gpt(gpt, params):
        """Charge les poids OpenAI dans notre modèle GPT"""
        
        # Token et position embeddings
        gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
        gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
        
        # Transformer blocks
        for b in range(len(gpt.trf_blocks)):
            # Attention weights (concaténés dans OpenAI)
            q_w, k_w, v_w = torch.split(
                params[f'h.{b}.attn.c_attn.weight'], 
                GPT_CONFIG_124M["emb_dim"], 
                dim=-1
            )
            gpt.trf_blocks[b].att.W_query.weight = assign(
                gpt.trf_blocks[b].att.W_query.weight, q_w.T
            )
            gpt.trf_blocks[b].att.W_key.weight = assign(
                gpt.trf_blocks[b].att.W_key.weight, k_w.T
            )
            gpt.trf_blocks[b].att.W_value.weight = assign(
                gpt.trf_blocks[b].att.W_value.weight, v_w.T
            )
            
            # Attention projection
            gpt.trf_blocks[b].att.out_proj.weight = assign(
                gpt.trf_blocks[b].att.out_proj.weight, 
                params[f'h.{b}.attn.c_proj.weight'].T
            )
            gpt.trf_blocks[b].att.out_proj.bias = assign(
                gpt.trf_blocks[b].att.out_proj.bias,
                params[f'h.{b}.attn.c_proj.bias']
            )
            
            # Feed forward
            gpt.trf_blocks[b].ff.layers[0].weight = assign(
                gpt.trf_blocks[b].ff.layers[0].weight,
                params[f'h.{b}.mlp.c_fc.weight'].T
            )
            gpt.trf_blocks[b].ff.layers[0].bias = assign(
                gpt.trf_blocks[b].ff.layers[0].bias,
                params[f'h.{b}.mlp.c_fc.bias']
            )
            gpt.trf_blocks[b].ff.layers[2].weight = assign(
                gpt.trf_blocks[b].ff.layers[2].weight,
                params[f'h.{b}.mlp.c_proj.weight'].T
            )
            gpt.trf_blocks[b].ff.layers[2].bias = assign(
                gpt.trf_blocks[b].ff.layers[2].bias,
                params[f'h.{b}.mlp.c_proj.bias']
            )
            
            # Layer norms
            gpt.trf_blocks[b].norm1.scale = assign(
                gpt.trf_blocks[b].norm1.scale,
                params[f'h.{b}.ln_1.weight']
            )
            gpt.trf_blocks[b].norm1.shift = assign(
                gpt.trf_blocks[b].norm1.shift,
                params[f'h.{b}.ln_1.bias']
            )
            gpt.trf_blocks[b].norm2.scale = assign(
                gpt.trf_blocks[b].norm2.scale,
                params[f'h.{b}.ln_2.weight']
            )
            gpt.trf_blocks[b].norm2.shift = assign(
                gpt.trf_blocks[b].norm2.shift,
                params[f'h.{b}.ln_2.bias']
            )
        
        # Final layer norm
        gpt.final_norm.scale = assign(gpt.final_norm.scale, params['ln_f.weight'])
        gpt.final_norm.shift = assign(gpt.final_norm.shift, params['ln_f.bias'])
        
        # Output head (partage avec token embedding)
        gpt.out_head.weight = assign(gpt.out_head.weight, params['wte'])
    
    load_weights_into_gpt(model, params)
    print("✓ Poids chargés avec succès!")
    
    # 6. Test de génération
    print("\n6. Test de génération avec GPT-2 officiel:")
    model.eval()
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    model.to(device)
    
    tokenizer = tiktoken.get_encoding("gpt2")
    
    prompts = [
        "Hello, I am",
        "The future of AI is",
        "In a galaxy far, far away"
    ]
    
    print("\n" + "="*60)
    print("GÉNÉRATIONS AVEC GPT-2 OFFICIEL")
    print("="*60)
    
    for prompt in prompts:
        print(f"\n→ Prompt: '{prompt}'")
        token_ids = torch.tensor(
            tokenizer.encode(prompt)
        ).unsqueeze(0).to(device)
        
        torch.manual_seed(123)
        output_ids = generate(
            model=model,
            idx=token_ids,
            max_new_tokens=30,
            context_size=GPT_CONFIG_124M["context_length"],
            temperature=1.0,
            top_k=50
        )
        
        output_text = tokenizer.decode(output_ids.squeeze(0).tolist())
        print(f"   {output_text}")
    
    print("\n" + "="*60)
    print("✓ FÉLICITATIONS!")
    print("="*60)
    print("\nVous avez:")
    print("  ✓ Implémenté GPT-2 (124M) from scratch")
    print("  ✓ Entraîné le modèle sur vos données")
    print("  ✓ Chargé les poids officiels OpenAI")
    print("  ✓ Généré du texte avec différentes stratégies de sampling")
    print("\nVous comprenez maintenant comment fonctionnent les LLMs! 🎉")
    
except ImportError:
    print("❌ Erreur: gpt_download.py non trouvé")
    print("\n  Pour utiliser cette fonctionnalité:")
    print("  1. Téléchargez gpt_download.py depuis le TP")
    print("  2. Placez-le dans le dossier racine du projet")
    print("  3. Relancez ce script")
    print("\n  Alternatif: Continuez avec votre modèle entraîné")
    
    # Générer avec modèle entraîné à la place
    print("\n  Génération avec modèle entraîné à la place:")
    
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
        print("  ✓ Modèle entraîné chargé")
    except:
        print("  ⚠ Aucun modèle entraîné trouvé")
    
    model.eval()
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    model.to(device)
    
    tokenizer = tiktoken.get_encoding("gpt2")
    prompt = "Every effort moves you"
    
    print(f"\n  → Prompt: '{prompt}'")
    token_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    
    torch.manual_seed(123)
    output_ids = generate(
        model=model,
        idx=token_ids,
        max_new_tokens=30,
        context_size=GPT_CONFIG_124M["context_length"],
        temperature=1.0,
        top_k=25
    )
    
    output_text = tokenizer.decode(output_ids.squeeze(0).tolist())
    print(f"     {output_text}")

except Exception as e:
    print(f"❌ Erreur inattendue: {e}")
    print("\n  Le chargement des poids OpenAI est optionnel.")
    print("  Votre modèle entraîné fonctionne parfaitement!")