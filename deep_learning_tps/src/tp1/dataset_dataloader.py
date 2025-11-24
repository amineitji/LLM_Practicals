import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
import matplotlib.pyplot as plt
import numpy as np

print("="*60)
print("TP1 - PARTIE 2: DATASET & DATALOADER")
print("="*60)

# Charger texte
with open('data/the-verdict.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()

# 2. GPTDatasetV1
print("\n2. Création GPTDatasetV1:")

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []
        
        # Encoder le texte
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        
        # Fenêtre glissante
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

# 3. Dataloader
print("\n3. Fonction create_dataloader_v1:")

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, 
                         shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                           drop_last=drop_last, num_workers=num_workers)
    return dataloader

# 4. Test dataloader
print("\n4. Test dataloader (max_length=stride=4):")
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, 
                                  stride=4, shuffle=False)

print(f"Nombre de batches: {len(dataloader)}")
print(f"Taille dataset: {len(dataloader.dataset)}")

# Premier batch
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

print(f"\nPremier batch:")
print(f"Inputs shape: {inputs.shape}")
print(f"Targets shape: {targets.shape}")
print(f"\nInputs:\n{inputs}")
print(f"\nTargets:\n{targets}")

# Vérifier décalage
print("\n5. Vérification du décalage inputs -> targets:")
tokenizer = tiktoken.get_encoding("gpt2")
for i in range(3):
    input_text = tokenizer.decode(inputs[i].tolist())
    target_text = tokenizer.decode(targets[i].tolist())
    print(f"\nSéquence {i+1}:")
    print(f"  Input:  {inputs[i].tolist()} -> '{input_text}'")
    print(f"  Target: {targets[i].tolist()} -> '{target_text}'")

# Test avec différents strides
print("\n6. Impact du stride:")
for stride_val in [4, 2, 1]:
    dl = create_dataloader_v1(raw_text, batch_size=8, max_length=4, 
                              stride=stride_val, shuffle=False)
    print(f"Stride={stride_val}: {len(dl.dataset)} séquences, {len(dl)} batches")

# Visualisation overlap
print("\n7. Visualisation de l'overlap:")
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

for idx, (stride_val, ax) in enumerate(zip([4, 2, 1], axes)):
    dl = create_dataloader_v1(raw_text, batch_size=1, max_length=8, 
                              stride=stride_val, shuffle=False)
    
    sequences = []
    for i, (inp, _) in enumerate(dl):
        if i >= 10:
            break
        sequences.append(inp[0].tolist())
    
    # Matrice pour visualiser overlap
    max_token = max(max(seq) for seq in sequences)
    matrix = np.zeros((len(sequences), max(len(seq) for seq in sequences)))
    
    for i, seq in enumerate(sequences):
        for j, token_id in enumerate(seq):
            matrix[i, j] = token_id
    
    im = ax.imshow(matrix, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_xlabel('Position dans la séquence', fontsize=11)
    ax.set_ylabel('Index séquence', fontsize=11)
    ax.set_title(f'Stride = {stride_val} (overlap = {8-stride_val})', 
                 fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Token ID')

plt.tight_layout()
plt.savefig('outputs/tp1_stride_overlap.png', dpi=150, bbox_inches='tight')
print("✓ Visualisation sauvegardée: outputs/tp1_stride_overlap.png")
plt.close()

# Créer dataloader pour la suite
print("\n8. Création du dataloader principal (max_length=256, stride=128):")
main_dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=256, 
                                       stride=128, shuffle=True, drop_last=True)
print(f"Dataset size: {len(main_dataloader.dataset)}")
print(f"Number of batches: {len(main_dataloader)}")
print(f"Batch size: 8")
print("✓ Dataloader prêt pour l'entraînement")
