import os
import urllib.request
import json
import numpy as np
import tensorflow as tf # Non requis si on lit manuellement, mais ici on fait simple
import torch

def download_and_load_gpt2(model_size, models_dir):
    # Validation de la taille du modèle
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # Création du dossier
    model_dir = os.path.join(models_dir, model_size)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # Téléchargement des fichiers
    print(f"Téléchargement du modèle {model_size} dans {model_dir} ...")
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        
        if not os.path.exists(file_path):
            print(f" -> Téléchargement {filename}...")
            try:
                urllib.request.urlretrieve(file_url, file_path)
            except urllib.error.HTTPError as e:
                print(f"Erreur téléchargement {filename}: {e}")

    # Chargement des hyperparamètres
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    hparams = json.load(open(os.path.join(model_dir, "hparams.json")))
    
    # Chargement des poids (nécessite TensorFlow pour lire le format .ckpt)
    # Note: Si vous n'avez pas TF, c'est plus complexe (lecture binaire manuelle)
    # Pour ce TP, on suppose l'installation possible ou on utilise une astuce numpy
    
    print("Chargement des tenseurs (via lecture binaire numpy/tf)...")
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, hparams)
    
    return hparams, params

def load_gpt2_params_from_tf_ckpt(ckpt_path, hparams):
    def set_in_nested_dict(d, keys, val):
        if not keys:
            return val
        if keys[0] not in d:
            d[keys[0]] = {}
        d[keys[0]] = set_in_nested_dict(d[keys[0]], keys[1:], val)
        return d

    params = {"blocks": [{} for _ in range(hparams["n_layer"])]}
    
    # Lecture brute des variables TF sans installer TF complet si possible
    # Sinon, on utilise tf.train.load_variable
    try:
        import tensorflow as tf
        for name, _ in tf.train.list_variables(ckpt_path):
            array = tf.train.load_variable(ckpt_path, name)
            array = array.squeeze()
            
            parts = name.split("/")
            
            # Mapping des noms de variables OpenAI vers notre structure
            if parts[0].startswith("model"):
                if parts[1].startswith("h"):
                    layer_idx = int(parts[1][1:])
                    # Reconstruction de la hiérarchie pour les blocs
                    # ex: model/h0/attn/c_attn/w -> blocks[0]['attn']['c_attn']['w']
                    target_dict = params["blocks"][layer_idx]
                    key_path = parts[2:]
                    
                    # Fonction récursive simple pour remplir le dictionnaire
                    curr = target_dict
                    for key in key_path[:-1]:
                        if key not in curr: curr[key] = {}
                        curr = curr[key]
                    curr[key_path[-1]] = array
                    
                elif parts[1] == "wte":
                    params["wte"] = array
                elif parts[1] == "wpe":
                    params["wpe"] = array
                elif parts[1] == "ln_f":
                    if parts[2] == "g": params["ln_f_g"] = array
                    if parts[2] == "b": params["ln_f_b"] = array
                    
        return params
        
    except ImportError:
        print("❌ Erreur: TensorFlow est requis pour lire les poids .ckpt d'OpenAI.")
        print("👉 pip install tensorflow")
        raise