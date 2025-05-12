import torch
torch.set_printoptions(profile="full")

jacobian = torch.load("jacobian.pt").coalesce()
jacobian = jacobian.to_dense().to_sparse_csr()
jac_T = jacobian.transpose(0, 1)

print(f'{jacobian.shape = }, {jac_T.shape = }')
print(jacobian._nnz())
print(jacobian)
