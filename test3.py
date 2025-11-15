import cupy as cp
from cupyx.scipy.sparse import csr_matrix
import torch
import nvmath

from pde.utils import ARTEFACT_DIR

def torch_csr_to_cupy(A: torch.Tensor) -> csr_matrix:
    assert A.layout == torch.sparse_csr, "A must be CSR sparse"
    assert A.is_cuda, "A must be on CUDA"
    A = A.clone() # Ensure contiguous
    crow = A.crow_indices()
    col = A.col_indices()
    data = A.values()

    # Zero-copy convert via DLPack (stays on GPU)
    crow_cp = cp.from_dlpack(crow)
    col_cp  = cp.from_dlpack(col)
    data_cp = cp.from_dlpack(data)

    # CuPy CSR wants (data, indices, indptr)
    # and typically int32 indices; cast if needed
    crow_cp = crow_cp.astype(cp.int32)
    col_cp  = col_cp.astype(cp.int32)

    return csr_matrix((data_cp, col_cp, crow_cp), shape=A.shape)

def main():
    save_dict = torch.load(ARTEFACT_DIR / "A_cudss.pt")
    A, b = save_dict["A"], save_dict["b"]

    # A = A.to_dense()
    # col_norms = A.norm(dim=0, keepdim=True).clamp_min(1e-3)
    # A = A / col_norms
    # A = A.to_sparse_csr()

    # Row norms
    A = A.to_dense()
    row_norms = A.norm(dim=1, keepdim=True).clamp(min=1e-4)
    A = A / row_norms
    b = b / row_norms.squeeze(-1)
    A = A.to_sparse_csr()

    A_cp = torch_csr_to_cupy(A)
    b_cp = cp.from_dlpack(b)

    solver = nvmath.sparse.advanced.DirectSolver(A_cp, b_cp)
    # solver.solution_config.ir_num_steps = 1
    # solver.factorization_config.pivot_eps = 1e-5
    # solver.factorization_config.factorization_algorithm = 1
    solver.plan()
    solver.factorize()
    x_cp = solver.solve()

    x = torch.from_dlpack(x_cp)

    # Dense solver
    x_dense = torch.linalg.solve(A.to_dense(), b)  # For comparison

    #### Evaluation ####
    save_dict = torch.load(ARTEFACT_DIR / "A_cudss.pt")
    A, b = save_dict["A"], save_dict["b"]

    #x = x / col_norms.squeeze(0)
    err = torch.linalg.norm(A @ x - b)
    print(f'Solution residual norm: {err:.4g}')

    # For comparison, dense solve
    #x_dense = x_dense / col_norms.squeeze(0)
    err = torch.linalg.norm(A.to_dense() @ x_dense - b)
    print(f'Dense solution residual norm: {err:.4g}')

if __name__ == "__main__":
    main()
