# TP2 - GPT-2 Implementation & Pre-training

## Vue d'ensemble

Implémentation complète d'un modèle GPT-2 (124M paramètres) from scratch et entraînement sur corpus texte.

---

## 1. Layer Normalization

**Concept :** Normaliser les activations pour stabiliser l'entraînement.

**Formule :** `(x - mean) / sqrt(variance + eps)`

![LayerNorm](outputs/tp2_layernorm.png)

**Résultat :**
- Avant : valeurs dispersées (0 à 0.5)
- Après : valeurs centrées autour de 0
- Mean ≈ 0, Variance ≈ 1

**Pourquoi ?** Réduit le covariate shift, permet learning rates plus élevés.

---

## 2. GELU Activation

**Formule :** GELU(x) = 0.5 × x × (1 + tanh[...])

![GELU](outputs/tp2_gelu.png)

**GELU vs ReLU :**
- **ReLU** : Coupure dure à 0
- **GELU** : Transition douce (probabiliste)
- **Avantage** : Pas de "dying neurons", gradient même pour valeurs négatives

**FeedForward :** Linear(768→3072) → GELU → Linear(3072→768)

---

## 3. Residual Connections

**Concept :** `output = layer(x) + x` (skip connection)

![Residual](outputs/tp2_residual.png)

**Impact sur les gradients :**

**Sans residual :**
```
Layer 1: 0.00020  ← Très faible
Layer 2: 0.00012
Layer 5: 0.00505
```

**Avec residual :**
```
Layer 1: 0.222    ← Beaucoup plus fort !
Layer 2: 0.207
Layer 5: 1.326
```

**Pourquoi ?** Résout le vanishing gradient, permet réseaux profonds (12+ couches).

---

## 4. Transformer Block

**Architecture :**
```
Input → LayerNorm → Attention → Dropout → +Residual
      → LayerNorm → FeedForward → Dropout → +Residual → Output
```

**Paramètres par bloc :** 7,085,568
- Attention : 2,360,064
- FeedForward : 4,722,432
- LayerNorms : 3,072

---

## 5. GPT-2 Complet

**Pipeline :**
```
Token IDs [batch, seq]
    ↓
Token + Position Embeddings
    ↓
12× Transformer Blocks
    ↓
LayerNorm final
    ↓
Linear (768 → 50257)
    ↓
Logits [batch, seq, vocab_size]
```

**Paramètres totaux :** 163,009,536
- Uniques : 124,412,160 (~124M)
- Taille : 621 MB (float32)

---

## 6-7. Génération de texte

**Méthode greedy :**
1. Calculer logits
2. Softmax → probabilités
3. Argmax → token le plus probable
4. Ajouter au contexte
5. Répéter

**Avant entraînement :**
```
Input:  "Hello, I am"
Output: "Hello, I am Featureiman Byeswickattribute argue"
```
→ Incohérent (poids aléatoires)

---

## 8. Loss & Perplexity

**Test sur modèle non-entraîné :**
- Prédictions : "Armed he Netflix" vs "effort moves you"
- Cross Entropy : 10.794
- **Perplexity : 48,726** (très élevée = très confus)

---

## 9. Entraînement

**Données :**
- Corpus : the-verdict.txt (20K caractères)
- Train : 18K (90%) / Val : 2K (10%)
- 10 epochs, AdamW optimizer

**Évolution :**

| Epoch | Train Loss | Val Loss | Exemple génération |
|-------|-----------|----------|-------------------|
| 1 | 9.83 | 9.98 | "Every effort moves you,,,,,,,,,,,," |
| 2 | 6.81 | 7.06 | "Every effort moves you, and, and, and..." |
| 5 | 4.59 | 6.25 | "Every effort moves you, and, and he was..." |
| 10 | 1.12 | 6.28 | **"Yes--quite insensible to the irony. She wanted him vindicated--and by me!"** |

![Training](outputs/tp2_training.png)

**Résultat :**
- ✅ Train loss : 10.98 → 1.12 (-90%)
- ✅ Génération cohérente et réaliste
- ⚠️ Val loss stagne à 6.28 (overfitting léger)

---

## 10-11. Temperature & Top-K Sampling

**Temperature :** Contrôle la diversité

![Temperature](outputs/tp2_temperature.png)

- **Temp 0.1** : Très conservatif (toujours même mot)
- **Temp 1.0** : Normal
- **Temp 5.0** : Très créatif (mais peut perdre cohérence)

**Top-K :** Garde seulement les k meilleurs tokens

![Top-K](outputs/tp2_topk.png)

- Sans top-k : Tous les tokens possibles
- Avec top-k=3 : Seulement "forward", "toward", "closer"

**Formule :** `logits / temperature` puis softmax + sampling

---

## 12. Génération avancée

**Nouvelle fonction :**
```python
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=25):
    # Temperature scaling + top-k filtering
    # Multinomial sampling au lieu d'argmax
```

**Résultat :**
- Non-déterministe (2 générations différentes)
- Plus de diversité
- Contrôle créativité vs cohérence

**Exemples (temp=1.4, top-k=25) :**
```
Output 1: "Every effort moves you?"
          "Yes--quite insensible to the irony..."

Output 2: "Every effort moves you?"
          "Yes--quite insensible to the portrait..."
```

---

## 13. Sauvegarde/Chargement

**Fichiers créés :**

1. **gpt2_trained.pth** (620 MB)
   - State dict du modèle entraîné

2. **gpt2_checkpoint.pth** (1.86 GB)
   - Modèle + optimizer + métadonnées
   - Permet de reprendre l'entraînement

**Chargement :**
```python
model.load_state_dict(torch.load('model.pth'), strict=False)
```
→ `strict=False` pour ignorer les buffers (mask)

---

## Résultats finaux

### ✅ Accomplissements

1. **Implémenté GPT-2 from scratch** (124M paramètres)
2. **Entraîné le modèle** (loss 10.98 → 1.12)
3. **Génération cohérente** après 10 epochs
4. **Sampling avancé** (temperature + top-k)
5. **Sauvegarde/chargement** fonctionnel

### 📊 Métriques finales

- Train loss : **1.116**
- Val loss : **6.281**
- Perplexity val : **~535** (vs 48,726 au départ)
- Taille modèle : **621 MB**

### 🎯 Exemple de génération (epoch 10)

**Prompt :** "Every effort moves you"

**Output :**
```
"Every effort moves you?"

"Yes--quite insensible to the irony. She wanted him 
vindicated--and by me!"

"Oh, and back his head to look up at the sketch of 
the donkey."
```

→ **Dialogue réaliste avec structure narrative cohérente !**

---

## Visualisations

Toutes les visualisations sont dans `/outputs/` :
- `tp2_layernorm.png` - Normalisation des activations
- `tp2_gelu.png` - Comparaison activations
- `tp2_residual.png` - Impact des skip connections
- `tp2_training.png` - Évolution des losses
- `tp2_temperature.png` - Impact de la température
- `tp2_topk.png` - Filtrage top-k

---

## Concepts clés maîtrisés

✅ **LayerNorm** - Stabilisation de l'entraînement  
✅ **GELU** - Activation moderne pour transformers  
✅ **Residual connections** - Réseaux profonds  
✅ **Transformer blocks** - Architecture modulaire  
✅ **Pre-training** - Entraînement de LLM  
✅ **Sampling** - Génération contrôlée  
✅ **Model persistence** - Sauvegarde/chargement  

---

## Conclusion

**Mission accomplie !** 🎉

Tu as construit et entraîné un vrai modèle GPT-2 from scratch. Le modèle génère du texte cohérent après seulement 10 epochs sur un petit corpus. 

**Tu comprends maintenant comment ChatGPT fonctionne à l'intérieur.**