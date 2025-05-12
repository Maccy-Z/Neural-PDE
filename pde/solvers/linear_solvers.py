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
        x = self.amgx_solver.solve(b, x)
        return x

    def cuda_iterative(self, A_cp: cp.array, b: torch.Tensor):
        b_cp = cp.from_dlpack(b)

        # Convert the dense matrix A_cupy to a sparse CSR matrix
        A_sparse_cupy = sp.csr_matrix(A_cp)

        # Solve the sparse linear system Ax = b using CuPy
        x, info = gmres(A_sparse_cupy, b_cp, **self.cfg)
        x = torch.from_dlpack(x)
        return x

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

    def preproc_tensor(self, A: torch.Tensor):
        """ Preprocess A matrix before solving, convert to sparse if needed so original can be deleted. """
        return self.preproc(A)

    def preproc_default(self, A: torch.Tensor):
        return A

    def preproc_sparse(self, A: torch.Tensor) -> sp.csr_matrix:
        """ Convert a torch tensor to a cupy sparse tensor """
        if A.is_sparse_csr:
            # A = A.to_dense().to_sparse_csr()
            values = A.values()
            indices = A.col_indices()
            indptr = A.crow_indices()

            values_cp = cp.from_dlpack(values)
            indices_cp = cp.from_dlpack(indices)
            indptr_cp = cp.from_dlpack(indptr)

            A_sparse_cp = sp.csr_matrix((values_cp, indices_cp, indptr_cp), shape=A.size())
        else:
            A_cp = cp.from_dlpack(A)
            A_sparse_cp = sp.csr_matrix(A_cp)
        return A_sparse_cp





