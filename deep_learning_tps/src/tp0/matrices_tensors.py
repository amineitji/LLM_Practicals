import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("="*60)
print("TP0 - PARTIE 1: MATRICES ET TENSORS")
print("="*60)

# 1. Version PyTorch
print(f"\n1. PyTorch version: {torch.__version__}")
assert torch.__version__ >= "2.1.2", "Version PyTorch trop ancienne"

# 2. Créer tenseurs de différentes dimensions
print("\n2. Création de tenseurs de différentes dimensions:")
scalar = torch.tensor(42)
vector = torch.tensor([1, 2, 3, 4])
matrix = torch.tensor([[1, 2], [3, 4]])
tensor_3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

print(f"Scalar (0D): {scalar}, shape: {scalar.shape}")
print(f"Vector (1D): {vector}, shape: {vector.shape}")
print(f"Matrix (2D):\n{matrix}, shape: {matrix.shape}")
print(f"Tensor 3D:\n{tensor_3d}, shape: {tensor_3d.shape}")

# 3. Numpy to Tensor
print("\n3. Conversion Numpy vers Tensor:")
np_array = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(f"Numpy array:\n{np_array}")

tensor_copy = torch.tensor(np_array)
tensor_shared = torch.from_numpy(np_array)

print(f"\ntorch.tensor() - copie: {tensor_copy}")
print(f"torch.from_numpy() - mémoire partagée: {tensor_shared}")

np_array[0, 0, 0] = 999
print(f"\nAprès modification numpy array:")
print(f"torch.tensor() (copie): {tensor_copy[0, 0, 0]}")
print(f"torch.from_numpy() (partagée): {tensor_shared[0, 0, 0]}")

# 4. Reshape vs View
print("\n4. Reshape vs View:")
matrix_2x3 = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f"Matrice originale 2x3:\n{matrix_2x3}")

reshaped = matrix_2x3.reshape(3, 2)
viewed = matrix_2x3.view(3, 2)

print(f"\nReshape 3x2:\n{reshaped}")
print(f"View 3x2:\n{viewed}")
print("\nDifférence: reshape peut copier, view nécessite mémoire contiguë")

# 5. Multiplication matricielle
print("\n5. Multiplication matricielle:")
result = torch.matmul(matrix_2x3, matrix_2x3.T)
print(f"2x3 @ 3x2 = shape {result.shape}")
print(f"Résultat:\n{result}")

# Visualisation
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

sns.heatmap(matrix_2x3.numpy(), annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
axes[0].set_title('Matrix 2x3')

sns.heatmap(matrix_2x3.T.numpy(), annot=True, fmt='d', cmap='Greens', ax=axes[1], cbar=False)
axes[1].set_title('Matrix 3x2 (Transposée)')

sns.heatmap(result.numpy(), annot=True, fmt='d', cmap='Reds', ax=axes[2], cbar=False)
axes[2].set_title('Résultat 2x2')

plt.tight_layout()
plt.savefig('outputs/tp0_matrices.png', dpi=150, bbox_inches='tight')
print("\n✓ Visualisation sauvegardée: outputs/tp0_matrices.png")
plt.close()