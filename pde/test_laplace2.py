import torch


A = torch.tensor([[1., 2], [3, 4]]).to_sparse_coo().requires_grad_(True)
b = torch.tensor([1., 2])

y = torch.matmul(A, b)

print(y.grad_fn.next_functions)

