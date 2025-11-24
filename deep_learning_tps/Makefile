.PHONY: help setup tp0 tp1 tp2 tp2-train tp2-openai all clean interactive

PYTHON := python
SRC_TP0 := src/tp0
SRC_TP1 := src/tp1
SRC_TP2 := src/tp2

help:
	@echo ""
	@echo "════════════════════════════════════════════════════════════════"
	@echo "  Deep Learning TPs - Makefile"
	@echo "════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "Commandes disponibles:"
	@echo "  make setup       - Créer l'environnement et installer dépendances"
	@echo "  make tp0         - Lancer TP0 (PyTorch Basics)"
	@echo "  make tp1         - Lancer TP1 (Tokenizers & Attention)"
	@echo "  make tp2         - Lancer TP2 parties 1-8 (GPT-2 base)"
	@echo "  make tp2-train   - Lancer TP2 parties 9-13 (Entraînement)"
	@echo "  make tp2-openai  - Lancer TP2 partie 14 (Poids OpenAI)"
	@echo "  make all         - Lancer tous les TPs"
	@echo "  make clean       - Nettoyer outputs et models"
	@echo "  make interactive - Menu interactif"
	@echo ""

setup:
	@echo ""
	@echo "════════════════════════════════════════════════════════════════"
	@echo "  SETUP - Création environnement"
	@echo "════════════════════════════════════════════════════════════════"
	@echo ""
	@conda create -n dl_tps python=3.10 -y
	@echo ""
	@echo "✓ Environnement créé: dl_tps"
	@echo ""
	@echo "Pour activer: conda activate dl_tps"
	@echo "Puis installer: pip install torch tiktoken matplotlib numpy pandas seaborn"
	@echo ""

tp0:
	@echo ""
	@echo "════════════════════════════════════════════════════════════════"
	@echo "  LANCEMENT TP0 - PyTorch Basics"
	@echo "════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "▸ Partie 1: Matrices et Tenseurs"
	@echo "────────────────────────────────────────────────────────────────"
	@$(PYTHON) $(SRC_TP0)/matrices_tensors.py
	@echo ""
	@echo "▸ Partie 2: Gradients"
	@echo "────────────────────────────────────────────────────────────────"
	@$(PYTHON) $(SRC_TP0)/gradients.py
	@echo ""
	@echo "▸ Partie 3: Réseaux de Neurones"
	@echo "────────────────────────────────────────────────────────────────"
	@$(PYTHON) $(SRC_TP0)/neural_network.py
	@echo ""
	@echo "▸ Partie 4: Devices"
	@echo "────────────────────────────────────────────────────────────────"
	@$(PYTHON) $(SRC_TP0)/devices.py
	@echo ""
	@echo "════════════════════════════════════════════════════════════════"
	@echo "  TP0 TERMINÉ"
	@echo "════════════════════════════════════════════════════════════════"
	@echo ""

tp1:
	@echo ""
	@echo "════════════════════════════════════════════════════════════════"
	@echo "  LANCEMENT TP1 - Tokenizers & Attention"
	@echo "════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "▸ Partie 1: Tokenizers"
	@echo "────────────────────────────────────────────────────────────────"
	@$(PYTHON) $(SRC_TP1)/tokenizer.py
	@echo ""
	@echo "▸ Partie 2: Dataset & DataLoader"
	@echo "────────────────────────────────────────────────────────────────"
	@$(PYTHON) $(SRC_TP1)/dataset_dataloader.py
	@echo ""
	@echo "▸ Partie 3: Embeddings"
	@echo "────────────────────────────────────────────────────────────────"
	@$(PYTHON) $(SRC_TP1)/embeddings.py
	@echo ""
	@echo "▸ Partie 4: Attention Mechanism"
	@echo "────────────────────────────────────────────────────────────────"
	@$(PYTHON) $(SRC_TP1)/attention.py
	@echo ""
	@echo "▸ Partie 5: Self-Attention"
	@echo "────────────────────────────────────────────────────────────────"
	@$(PYTHON) $(SRC_TP1)/self_attention.py
	@echo ""
	@echo "▸ Partie 6: Multi-Head Attention"
	@echo "────────────────────────────────────────────────────────────────"
	@$(PYTHON) $(SRC_TP1)/multihead_attention.py
	@echo ""
	@echo "════════════════════════════════════════════════════════════════"
	@echo "  TP1 TERMINÉ"
	@echo "════════════════════════════════════════════════════════════════"
	@echo ""

tp2:
	@echo ""
	@echo "════════════════════════════════════════════════════════════════"
	@echo "  LANCEMENT TP2 - GPT-2 Implementation"
	@echo "════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "▸ Partie 1: Layer Normalization"
	@echo "────────────────────────────────────────────────────────────────"
	@$(PYTHON) $(SRC_TP2)/layer_norm.py
	@echo ""
	@echo "▸ Partie 2: GELU Activation"
	@echo "────────────────────────────────────────────────────────────────"
	@$(PYTHON) $(SRC_TP2)/gelu.py
	@echo ""
	@echo "▸ Partie 3: Residual Connections"
	@echo "────────────────────────────────────────────────────────────────"
	@$(PYTHON) $(SRC_TP2)/residual_connections.py
	@echo ""
	@echo "▸ Partie 4: Transformer Block"
	@echo "────────────────────────────────────────────────────────────────"
	@$(PYTHON) $(SRC_TP2)/transformer_block.py
	@echo ""
	@echo "▸ Partie 5: GPT Model Complet"
	@echo "────────────────────────────────────────────────────────────────"
	@$(PYTHON) $(SRC_TP2)/gpt_model.py
	@echo ""
	@echo "▸ Partie 6: Génération de texte (Step 1)"
	@echo "────────────────────────────────────────────────────────────────"
	@$(PYTHON) $(SRC_TP2)/text_generation.py
	@echo ""
	@echo "▸ Partie 7: Génération de texte (Step 2)"
	@echo "────────────────────────────────────────────────────────────────"
	@$(PYTHON) $(SRC_TP2)/text_generation_v2.py
	@echo ""
	@echo "▸ Partie 8: Loss et Perplexity"
	@echo "────────────────────────────────────────────────────────────────"
	@$(PYTHON) $(SRC_TP2)/loss_perplexity.py
	@echo ""
	@echo "════════════════════════════════════════════════════════════════"
	@echo "  TP2 TERMINÉ (Parties 1-8)"
	@echo "════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "Fichiers générés:"
	@echo "  - outputs/tp2_layernorm.png"
	@echo "  - outputs/tp2_gelu.png"
	@echo "  - outputs/tp2_residual.png"
	@echo ""
	@echo "⚠ NOTE: L'entraînement (Partie 9) peut être long!"
	@echo "Pour lancer l'entraînement: make tp2-train"
	@echo ""

tp2-train:
	@echo ""
	@echo "════════════════════════════════════════════════════════════════"
	@echo "  TP2 - ENTRAÎNEMENT (PEUT PRENDRE DU TEMPS)"
	@echo "════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "▸ Partie 9: Pré-entraînement"
	@echo "────────────────────────────────────────────────────────────────"
	@$(PYTHON) $(SRC_TP2)/training.py
	@echo ""
	@echo "▸ Partie 10-11: Temperature & Top-K Sampling"
	@echo "────────────────────────────────────────────────────────────────"
	@$(PYTHON) $(SRC_TP2)/sampling.py
	@echo ""
	@echo "▸ Partie 12: Génération avancée"
	@echo "────────────────────────────────────────────────────────────────"
	@$(PYTHON) $(SRC_TP2)/generate_advanced.py
	@echo ""
	@echo "▸ Partie 13: Sauvegarder/Charger"
	@echo "────────────────────────────────────────────────────────────────"
	@$(PYTHON) $(SRC_TP2)/save_load.py
	@echo ""
	@echo "════════════════════════════════════════════════════════════════"
	@echo "  TP2 ENTRAÎNEMENT TERMINÉ"
	@echo "════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "Fichiers générés:"
	@echo "  - outputs/tp2_training.png"
	@echo "  - outputs/tp2_temperature.png"
	@echo "  - outputs/tp2_topk.png"
	@echo "  - models/gpt2_trained.pth"
	@echo "  - models/gpt2_model_only.pth"
	@echo "  - models/gpt2_checkpoint.pth"
	@echo ""

tp2-openai:
	@echo ""
	@echo "════════════════════════════════════════════════════════════════"
	@echo "  TP2 - CHARGER POIDS OPENAI"
	@echo "════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "▸ Partie 14: Charger GPT-2 officiel"
	@echo "────────────────────────────────────────────────────────────────"
	@$(PYTHON) $(SRC_TP2)/load_openai.py
	@echo ""
	@echo "════════════════════════════════════════════════════════════════"
	@echo "  TP2 COMPLET TERMINÉ!"
	@echo "════════════════════════════════════════════════════════════════"
	@echo ""

all: tp0 tp1 tp2 tp2-train

clean:
	@echo ""
	@echo "════════════════════════════════════════════════════════════════"
	@echo "  NETTOYAGE"
	@echo "════════════════════════════════════════════════════════════════"
	@echo ""
	@rm -rf outputs/*.png
	@rm -rf models/*.pth
	@rm -rf __pycache__ src/__pycache__ src/tp0/__pycache__ src/tp1/__pycache__ src/tp2/__pycache__
	@echo "✓ Fichiers nettoyés"
	@echo ""

interactive:
	@echo ""
	@echo "════════════════════════════════════════════════════════════════"
	@echo "  MENU INTERACTIF - Deep Learning TPs"
	@echo "════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "Choisissez une option:"
	@echo ""
	@echo "  [1] TP0 - PyTorch Basics"
	@echo "  [2] TP1 - Tokenizers & Attention"
	@echo "  [3] TP2 - GPT-2 Implementation (parties 1-8)"
	@echo "  [4] TP2 - Entraînement (parties 9-13)"
	@echo "  [5] TP2 - Charger poids OpenAI (partie 14)"
	@echo "  [6] Tous les TPs"
	@echo "  [7] Nettoyer outputs"
	@echo "  [q] Quitter"
	@echo ""
	@read -p "Votre choix: " choice; \
	case $$choice in \
		1) $(MAKE) tp0 ;; \
		2) $(MAKE) tp1 ;; \
		3) $(MAKE) tp2 ;; \
		4) $(MAKE) tp2-train ;; \
		5) $(MAKE) tp2-openai ;; \
		6) $(MAKE) all ;; \
		7) $(MAKE) clean ;; \
		q|Q) echo ""; echo "Au revoir!"; echo ""; exit 0 ;; \
		*) echo ""; echo "Option invalide"; echo ""; $(MAKE) interactive ;; \
	esac; \
	echo ""; \
	read -p "Appuyez sur Entrée pour revenir au menu..." dummy; \
	$(MAKE) interactive