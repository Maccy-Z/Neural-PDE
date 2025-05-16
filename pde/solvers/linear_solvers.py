import torch
import scipy.sparse.linalg as linalg
from cprint import c_print

import cupy as cp
import cupyx.scipy.sparse as sp
import cupyx.scipy.sparse.linalg as sp_linalg
from typing import Callable

from pde.solvers.gmres import gmres
from pde.solvers.pyamgx_holder import PyAMGXManager
from pde.config import LinMode


class LinearSolver:
    """ Solve Ax = b for x """
    solver: Callable
    preproc: Callable

    cfg: dict = None
    def __init__(self, mode: LinMode, device: str, cfg: dict=None):
        self.preproc = self.preproc_default
        if device == "cuda":
            if mode == LinMode.DENSE:
                self.solver = self.cuda_dense
            elif mode == LinMode.SPARSE:
                self.solver = self.cuda_sparse
                self.preproc = self.preproc_sparse
            elif mode == LinMode.ITERATIVE:
                self.solver = self.cuda_iterative
                self.cfg = cfg
                self.preproc = self.preproc_sparse
            elif mode == LinMode.AMGX:
                self.cfg = cfg
                self.amgx_solver = PyAMGXManager().create_solver(cfg)
                self.solver = self.cuda_amgx
                self.preproc = self.preproc_sparse

        elif device == "cpu":
            if mode == LinMode.DENSE:
                self.solver = self.cpu_dense
            elif mode == LinMode.SPARSE:
                self.solver = self.cpu_sparse

    def solve(self, A, b):
        return self.solver(A, b)

    def cuda_amgx(self, A_cp: cp.array, b: torch.Tensor):
        # Cupy to sparse is faster than torch to sparse
        self.amgx_solver.init_solver_cp(A_cp)
        x = torch.zeros_like(b)
        x, resid = self.amgx_solver.solve(b, x)
        return x, resid

    def cuda_iterative(self, A_cp: cp.array, b: torch.Tensor):
        b_cp = cp.from_dlpack(b)

        # Convert the dense matrix A_cupy to a sparse CSR matrix
        A_sparse_cupy = sp.csr_matrix(A_cp)

        # Solve the sparse linear system Ax = b using CuPy
        x, info = gmres(A_sparse_cupy, b_cp, **self.cfg)
        x = torch.from_dlpack(x)
        return x, 0

    def cuda_sparse(self, A_cp: cp.array, b: torch.Tensor):
        #A_cupy = cp.from_dlpack(A)
        b_cupy = cp.from_dlpack(b)

        A_cp = A_cp.astype(cp.float64)
        b_cupy = b_cupy.astype(cp.float64)

        # Solve the sparse linear system Ax = b using CuPy
        x = sp_linalg.spsolve(A_cp, b_cupy)

        x = torch.from_dlpack(x).float()
        values = torch.from_dlpack(A_cp.data)
        col_indices = torch.from_dlpack(A_cp.indices)
        crow_indices = torch.from_dlpack(A_cp.indptr)

        A = torch.sparse_csr_tensor(
                                    crow_indices,
                                    col_indices,
                                    values,
                                    size=A_cp.shape,
                                    device='cuda'
                                    ).float()
        residual = torch.linalg.norm(A @ x - b)
        return x, residual

    def cuda_dense(self, A: torch.Tensor, b: torch.Tensor):
        A = A.to_dense()
        # c_print(torch.linalg.matrix_rank(A), color="green")
        # c_print(A.shape, color="green")
        deltas = torch.linalg.solve(A, b)
        return deltas

    def cpu_sparse(self, A: torch.Tensor, b: torch.Tensor):
        A = A.numpy()
        b = b.numpy()
        deltas = linalg.spsolve(A, b, use_umfpack=True)
        deltas = torch.from_numpy(deltas)
        return deltas

    def cpu_dense(self, A: torch.Tensor, b: torch.Tensor):
        deltas = torch.linalg.solve(A, b)
        return deltas

    def preproc_tensor(self, A: torch.Tensor, b: torch.Tensor):
        """ Preprocess A matrix before solving, convert to sparse if needed so original can be deleted. """
        return self.preproc(A, b)

    def preproc_default(self, A: torch.Tensor, b: torch.Tensor):
        """ Default preprocessing, no conversion. """
        return A, b

    def preproc_sparse(self, A: torch.Tensor, b: torch.Tensor) -> sp.csr_matrix:
        """ Convert a torch tensor to a cupy sparse tensor """
        from scipy.sparse.csgraph import reverse_cuthill_mckee
        import scipy.sparse as spsp

        A_sp = A.to_sparse_coo().coalesce()

        # 2) pull out the three 1-D arrays
        row = A_sp.indices()[0].cpu().numpy().ravel()  # shape (nnz,)
        col = A_sp.indices()[1].cpu().numpy().ravel()  # shape (nnz,)
        data = A_sp.values().cpu().numpy().ravel()  # shape (nnz,)

        # sanity check
        assert row.ndim == col.ndim == data.ndim == 1, "Must be 1-D!"

        # 3) build SciPy COO and convert to CSR (or keep COO if you like)
        A_sp = spsp.coo_matrix((data, (row, col)), shape=A.shape).tocsr()
        perm = reverse_cuthill_mckee(A_sp, symmetric_mode=True)
        perm = torch.tensor(perm.copy(), dtype=torch.int64)

        if A.is_sparse_csr:
            A = A.to_dense()

            # A = A[perm][:, perm]
            r = A.norm(dim=1)
            A = A / r.unsqueeze(1)  * r.mean()
            b = b.squeeze() / r * r.mean()

            A = A.to_sparse_csr()

            values = A.values()
            indices = A.col_indices()
            indptr = A.crow_indices()

            values_cp = cp.from_dlpack(values)
            indices_cp = cp.from_dlpack(indices)
            indptr_cp = cp.from_dlpack(indptr)

            A_sparse_cp = sp.csr_matrix((values_cp, indices_cp, indptr_cp), shape=A.size())

            # self._est_cond_num(A, A_sparse_cp)
            # exit(7)


        else:
            A_cp = cp.from_dlpack(A)
            A_sparse_cp = sp.csr_matrix(A_cp)
        return A_sparse_cp, b

    def _est_cond_num(self, A, A_cp):
        x = torch.randn(A.shape[0], device=A.device)
        for _ in range(100):
            y = A.matmul(x)  # y = A x
            x = A.t().matmul(y)  # x = Aᵀ(A x)
            x = x / x.norm()
        # Rayleigh quotient gives σₘₐₓ² ≈ xᵀ (AᵀA) x
        sigma_max = (A.matmul(x)).norm().item()
        print(f'{sigma_max = }')

        v = torch.randn(A.shape[0], device=A.device)
        for _ in range(20):
            v_new, _ = self.cuda_sparse((A_cp.T @ A_cp), v)
            v = v_new
            v = v / v.norm()
        sigma_min = (A @ v).norm().item()
        print(f'{sigma_min = }')

        cond_num = sigma_max / sigma_min
        print(f'{cond_num = }')
        exit(7)




