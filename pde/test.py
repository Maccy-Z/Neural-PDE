import torch

a = torch.tensor([[-22.4951,  30.0639,   6.2763],
        [-10.5029, -20.0643,  24.2934]])
b = torch.tensor([0.0205, 0.2064, 0.1157]) - 0.2280
grad = a @ b

print(grad)
a = torch.tensor([[ -0.3062,  26.8933, -27.2727],
        [-77.8432,  13.2859,  14.8987]])
b = torch.tensor([ 0.1, 0.2064,  0.0205]) - -0.1297
grad = a @ b
print(grad)