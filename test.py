import torch
from pde.utils import ARTEFACT_DIR
torch.set_printoptions(linewidth=120)


save_dict = torch.load(ARTEFACT_DIR/"jac_T.pt")
A = save_dict["J"].to_dense()
adj = save_dict["adj"]
b = save_dict["loss_u"]

base_error = (A @ adj - b).norm()
print(f'{base_error = }')


x = torch.linalg.solve(A, b)
error = (A @ x - b).norm()
print(f'{error = }')
print()

# Row norm
row_norms = A.norm(dim=1, keepdim=True).clamp(min=1e-4)
A2 = A / row_norms
b2 = b / row_norms.squeeze(-1)
x2 = torch.linalg.solve(A2, b2)

error_row = (A @ x2 - b).norm()
print(f'{error_row = }')


# Col norm
b3 = b
col_norms = A.norm(dim=0, keepdim=True).clamp_min(1e-3)
A3 = A / col_norms
print(f'{A3.to_sparse_coo().values().abs().mean() = }')
x3 = torch.linalg.solve(A3, b3)
x3 = x3 / col_norms.squeeze(0)

error_col = (A @ x3 - b).norm()
print(f'{error_col = }')


# Chatgpt suggested
row_norms = A.norm(dim=1, keepdim=True).clamp_min(1e-5)
A_row = A / row_norms                 # rows have unit 2-norm
b_row = b / row_norms.squeeze(-1)     # scale RHS the same way

col_norms = A_row.norm(dim=0, keepdim=True).clamp_min(1e-3)
A_eq = A_row / col_norms              # columns now unit 2-norm

x4 = torch.linalg.solve(A_eq, b_row)
x4 = x4 / col_norms.squeeze(0)        # undo column scaling

error3 = (A @ x4 - b).norm()
print(f'{error3 = }')


# Final
row_norms = A.norm(dim=1, keepdim=True).clamp_min(1e-4)
A_row = A / row_norms                 # rows have unit 2-norm
b_row = b / row_norms.squeeze(-1)     # scale RHS the same way

col_norms = A_row.norm(dim=0, keepdim=True).clamp_min(1e-4)
# col_norms = torch.ones_like(col_norms)
A_eq = A_row / col_norms              # columns now unit 2-norm
# NOTE: This is A_eq = D * A * diag(1/col_norms)

# ---- 3) Solve the equilibrated system ----
# (D * A * diag(1/col_norms)) * y = D * b
# Recover x from y using x = diag(1/col_norms) * y   (i.e., divide by col_norms)
x4 = torch.linalg.solve(A_eq, b_row)


x4 = (x4 / col_norms.squeeze(0))        # undo column scaling

# ---- 4) Residuals ----
abs_residual = (A @ x4 - b).norm()
print(f'{abs_residual = }')