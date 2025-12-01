import torch
import tiktoken
import sys
import time
import os
from gpt_model import GPTModel
from generate_advanced import generate

# --- CONFIGURATION & UTILITAIRES ---

def type_writer(text, speed=0.01):
    """Affiche le texte caractère par caractère pour un effet 'style rétro'"""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(speed)
    print("")

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

print("="*60)
print("DEMO INTERACTIVE - TEST MANUEL GPT-2 (MENU)")
print("="*60)

# Configuration du modèle (Doit être identique à l'entraînement)
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

# Choix du device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Device utilisé : {device}")

# --- CHARGEMENT DU MODÈLE ---
print("\nChargement du modèle...")
model = GPTModel(GPT_CONFIG_124M)
model_path = "models/gpt2_trained.pth"

try:
    # weights_only=True est recommandé pour la sécurité
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    # strict=False permet d'ignorer les buffers 'mask' qui causent l'erreur
    model.load_state_dict(state_dict, strict=False)
    print(f"✓ Modèle chargé avec succès depuis {model_path}")
except FileNotFoundError:
    print(f"❌ Erreur: Le fichier {model_path} n'existe pas.")
    print("👉 Lancez d'abord l'entraînement : python src/tp2/training.py")
    sys.exit(1)

model.to(device)
model.eval()
tokenizer = tiktoken.get_encoding("gpt2")

# --- LISTE DES PROMPTS PRÉDÉFINIS ---
# Ces phrases sont choisies car elles contiennent des mots clés du texte "The Verdict"
prompts_list = [
    "Jack Gisburn",                  # Le personnage principal
    "The picture was",               # Thème central (peinture)
    "Mrs. Gisburn said",             # Personnage secondaire
    "The donkey",                    # Élément récurrent du texte
    "It was a strange",              # Début générique
    "I went to the room",            # Action classique
    "Every effort moves you",        # La phrase d'exemple du TP
    "Rickham looked at",             # Autre personnage
    "The internet is",               # TEST PIÈGE (Mot inconnu du texte du 19e siècle)
    "[ECRIRE MON PROPRE PROMPT]"     # Option personnalisée
]

# --- BOUCLE PRINCIPALE ---
while True:
    print("\n" + "="*40)
    print("CHOISISSEZ UN DÉBUT DE PHRASE :")
    print("="*40)
    
    for i, p in enumerate(prompts_list):
        print(f"  [{i+1}] {p}")
    print("  [q]  Quitter")
    
    choice = input("\nVotre choix (1-10) > ")
    
    if choice.lower() in ['q', 'quit', 'exit']:
        print("Au revoir !")
        break
    
    selected_prompt = ""
    
    # Gestion du choix
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(prompts_list):
            if idx == len(prompts_list) - 1:
                # Option personnalisée
                selected_prompt = input("\nEntrez votre prompt : ")
            else:
                selected_prompt = prompts_list[idx]
        else:
            print("⚠ Choix invalide.")
            continue
    except ValueError:
        print("⚠ Veuillez entrer un nombre.")
        continue

    if not selected_prompt.strip():
        print("Prompt vide, on passe.")
        continue

    # --- GÉNÉRATION ---
    print(f"\nPrompt sélectionné : '{selected_prompt}'")
    print("Génération en cours...", end="\r")
    
    token_ids = torch.tensor(tokenizer.encode(selected_prompt)).unsqueeze(0).to(device)

    with torch.no_grad():
        output_ids = generate(
            model=model,
            idx=token_ids,
            max_new_tokens=40,       # Longueur de la génération
            context_size=GPT_CONFIG_124M["context_length"],
            temperature=0.8,         # 0.8 = bon équilibre créativité/cohérence
            top_k=40                 # Limite aux 40 mots les plus probables
        )

    decoded_text = tokenizer.decode(output_ids.squeeze(0).tolist())
    
    # Affichage du résultat
    print("\n" + "-" * 60)
    print("🤖 GPT-2 (The Verdict edition) :")
    # On met en gras (code ANSI) la partie générée pour la distinguer du prompt
    prompt_len = len(selected_prompt)
    sys.stdout.write(f"\033[1m{selected_prompt}\033[0m") # Prompt en gras
    type_writer(decoded_text[prompt_len:])               # Le reste machine à écrire
    print("-" * 60)
    
    input("\nAppuyez sur Entrée pour continuer...")