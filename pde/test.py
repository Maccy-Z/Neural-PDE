import torch

a = torch.tensor([[1, 0], [0, 1]])
b = torch.tensor([1, 2])

c = a.mv(b)
print(f'{c = }')