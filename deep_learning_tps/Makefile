.PHONY: help setup tp0 tp1 all clean interactive

PYTHON := python
SRC_TP0 := src/tp0
SRC_TP1 := src/tp1
OUTPUTS := outputs
MODELS := models
DATA := data

help:
	@echo "════════════════════════════════════════════════════════════════"
	@echo "  DEEP LEARNING TPS - MAKEFILE"
	@echo "════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "Commandes disponibles:"
	@echo ""
	@echo "  make interactive    - Menu interactif pour choisir quoi lancer"
	@echo "  make setup          - Créer l'environnement conda et installer dépendances"
	@echo "  make tp0            - Lancer TP0 (PyTorch Basics)"
	@echo "  make tp1            - Lancer TP1 (Tokenizers & Attention)"
	@echo "  make all            - Lancer TP0 puis TP1"
	@echo "  make clean          - Nettoyer outputs et models"
	@echo "  make help           - Afficher cette aide"
	@echo ""
	@echo "════════════════════════════════════════════════════════════════"

setup:
	@echo "════════════════════════════════════════════════════════════════"
	@echo "  SETUP ENVIRONNEMENT"
	@echo "════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "[1/6] Création de la structure de dossiers..."
	@mkdir -p $(SRC_TP0) $(SRC_TP1) $(DATA) $(MODELS) $(OUTPUTS)
	@echo "✓ Dossiers créés: src/tp0, src/tp1, data, models, outputs"
	@echo ""
	@echo "[2/6] Vérification environnement conda..."
	@if conda env list | grep -q "dl_tps"; then \
		echo "✓ Environnement 'dl_tps' existe déjà"; \
	else \
		echo "Création environnement conda 'dl_tps'..."; \
		conda create -n dl_tps python=3.10 -y; \
		echo "✓ Environnement créé"; \
	fi
	@echo ""
	@echo "[3/6] Installation PyTorch..."
	@bash -c "source $$(conda info --base)/etc/profile.d/conda.sh && conda activate dl_tps && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
	@echo "✓ PyTorch installé"
	@echo ""
	@echo "[4/6] Installation autres packages..."
	@bash -c "source $$(conda info --base)/etc/profile.d/conda.sh && conda activate dl_tps && pip install tiktoken matplotlib numpy pandas seaborn"
	@echo "✓ Packages installés"
	@echo ""
	@echo "[5/6] Génération requirements.txt..."
	@bash -c "source $$(conda info --base)/etc/profile.d/conda.sh && conda activate dl_tps && pip freeze > requirements.txt"
	@echo "✓ requirements.txt créé"
	@echo ""
	@echo "[6/6] Téléchargement des données..."
	@if [ ! -f $(DATA)/the-verdict.txt ]; then \
		wget -q -O $(DATA)/the-verdict.txt https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt; \
		echo "✓ the-verdict.txt téléchargé"; \
	else \
		echo "✓ the-verdict.txt déjà présent"; \
	fi
	@echo ""
	@echo "════════════════════════════════════════════════════════════════"
	@echo "  SETUP TERMINÉ"
	@echo "════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "Pour activer l'environnement:"
	@echo "  conda activate dl_tps"
	@echo ""

tp0:
	@echo ""
	@echo "════════════════════════════════════════════════════════════════"
	@echo "  LANCEMENT TP0 - PYTORCH BASICS"
	@echo "════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "▸ Partie 1: Matrices et Tensors"
	@echo "────────────────────────────────────────────────────────────────"
	@$(PYTHON) $(SRC_TP0)/matrices_tensors.py
	@echo ""
	@echo "▸ Partie 2: Gradients"
	@echo "────────────────────────────────────────────────────────────────"
	@$(PYTHON) $(SRC_TP0)/gradients.py
	@echo ""
	@echo "▸ Partie 3: Neural Networks"
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
	@echo "Fichiers générés:"
	@echo "  - outputs/tp0_matrices.png"
	@echo "  - outputs/tp0_gradients.png"
	@echo "  - outputs/tp0_training.png"
	@echo "  - models/model.pth"
	@echo ""

tp1:
	@echo ""
	@echo "════════════════════════════════════════════════════════════════"
	@echo "  LANCEMENT TP1 - TOKENIZERS & ATTENTION"
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
	@echo "Fichiers générés:"
	@echo "  - outputs/tp1_tokenizer_stats.png"
	@echo "  - outputs/tp1_stride_overlap.png"
	@echo "  - outputs/tp1_embeddings.png"
	@echo "  - outputs/tp1_attention_mechanism.png"
	@echo "  - outputs/tp1_self_attention.png"
	@echo "  - outputs/tp1_multihead_attention.png"
	@echo ""

all:
	@echo ""
	@echo "════════════════════════════════════════════════════════════════"
	@echo "  LANCEMENT COMPLET - TP0 + TP1"
	@echo "════════════════════════════════════════════════════════════════"
	@$(MAKE) tp0
	@$(MAKE) tp1
	@echo ""
	@echo "════════════════════════════════════════════════════════════════"
	@echo "  TOUS LES TPS TERMINÉS"
	@echo "════════════════════════════════════════════════════════════════"
	@echo ""

clean:
	@echo ""
	@echo "════════════════════════════════════════════════════════════════"
	@echo "  NETTOYAGE"
	@echo "════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "Suppression des fichiers générés..."
	@rm -rf $(OUTPUTS)/*.png
	@rm -rf $(MODELS)/*.pth
	@rm -rf __pycache__ src/__pycache__ $(SRC_TP0)/__pycache__ $(SRC_TP1)/__pycache__
	@echo "✓ Outputs nettoyés"
	@echo "✓ Models supprimés"
	@echo "✓ Cache Python supprimé"
	@echo ""
	@echo "════════════════════════════════════════════════════════════════"
	@echo "  NETTOYAGE TERMINÉ"
	@echo "════════════════════════════════════════════════════════════════"
	@echo ""

interactive:
	@echo ""
	@echo "════════════════════════════════════════════════════════════════"
	@echo "  MENU INTERACTIF - DEEP LEARNING TPS"
	@echo "════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "Que voulez-vous faire?"
	@echo ""
	@echo "  [0] Setup - Créer environnement et installer dépendances"
	@echo "  [1] TP0 - PyTorch Basics (matrices, gradients, NN, devices)"
	@echo "  [2] TP1 - Tokenizers & Attention (embeddings, self-attention, MHA)"
	@echo "  [3] Tout lancer (TP0 + TP1)"
	@echo "  [4] Nettoyer les outputs"
	@echo "  [5] Afficher l'aide"
	@echo "  [q] Quitter"
	@echo ""
	@echo "════════════════════════════════════════════════════════════════"
	@read -p "Votre choix: " choice; \
	case $$choice in \
		0) $(MAKE) setup ;; \
		1) $(MAKE) tp0 ;; \
		2) $(MAKE) tp1 ;; \
		3) $(MAKE) all ;; \
		4) $(MAKE) clean ;; \
		5) $(MAKE) help ;; \
		q|Q) echo ""; echo "Au revoir!"; echo ""; exit 0 ;; \
		*) echo ""; echo "❌ Choix invalide"; echo ""; $(MAKE) interactive ;; \
	esac; \
	echo ""; \
	read -p "Appuyez sur Entrée pour revenir au menu..." dummy; \
	$(MAKE) interactive