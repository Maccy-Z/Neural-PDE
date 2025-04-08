import torch

Tau = torch.tensor([[ 1.0167e-05, -4.3955e-05],
        [-4.3955e-05,  1.5055e-05]])
n = torch.tensor([-0.0122, -0.0209])

print((Tau * n.unsqueeze(-1)).sum(dim=-2))

