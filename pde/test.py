import torch
print(torch.cuda.is_available())
x = torch.tensor(4, device="cuda")

print(x)
