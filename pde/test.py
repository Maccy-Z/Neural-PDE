import torch

indices = torch.zeros(2, 1, dtype=torch.long)
indices[0, 0] = 0  # Row index (always 0 for a single row)
indices[1, 0] = 450  # Column index
print(f'{indices = }')
row_val = torch.sparse_coo_tensor(
    indices=indices,
    values=torch.ones(1),
    size=(1, 1744), device="cuda"
)
print(indices)
print(row_val)
