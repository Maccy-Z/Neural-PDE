import torch
import scipy.sparse.linalg as linalg
import time
import cupy as cp
import cupyx.scipy.sparse as sp
import cupyx.scipy.sparse.linalg as sp_linalg
from typing import Callable

from pde.solvers.gmres import gmres, gmres_cust
from pde.solvers.pyamgx_holder import PyAMGXManager
from pde.solvers.nvmath_cudss import CUDSSSolver
from pde.config import LinMode
from pde.utils_sparse import csr_compress, csr_torch_to_cupy

class LinearSolver:
    """ Solve Ax = b for x """
    solver: Callable
    preproc: Callable
    postproc: Callable
    cfg: dict = None
    col_norms: torch.Tensor = None

    def __init__(self, mode: LinMode, device: str, cfg: dict=None):

        self.preproc = self.preproc_default
        self.postproc = self.postproc_defualt

        if device == "cuda":
            if mode == LinMode.DENSE:
                self.solver = self.cuda_dense
            elif mode == LinMode.SPARSE:
                self.solver = self.cuda_sparse
                self.preproc = self.preproc_sparse
                self.postproc = self.postproc_sparse
            elif mode == LinMode.ITERATIVE:
                self.solver = self.cuda_iterative
                self.cfg = cfg
                self.preproc = self.preproc_sparse
            elif mode == LinMode.AMGX:
                self.cfg = cfg
                self.amgx_solver = PyAMGXManager().create_solver(cfg)
                self.solver = self.cuda_amgx
                self.preproc = self.preproc_sparse
                self.postproc = self.postproc_sparse
            elif mode == LinMode.CUDSS:
                self.cudss_solver = CUDSSSolver(cfg)
                self.solver = self.cudss


        elif device == "cpu":
            if mode == LinMode.DENSE:
                self.solver = self.cpu_dense
            elif mode == LinMode.SPARSE:
                self.solver = self.cpu_sparse

    def solve(self, A, b) -> torch.Tensor:
        A, b = self.preproc_tensor(A, b)
        x = self.solver(A, b)
        x = self.postproc(x)
        return x

    def cpu_sparse(self, A: torch.Tensor, b: torch.Tensor):
        A = A.numpy()
        b = b.numpy()
        deltas = linalg.spsolve(A, b, use_umfpack=True)
        deltas = torch.from_numpy(deltas)
        return deltas

    def cpu_dense(self, A: torch.Tensor, b: torch.Tensor):
        deltas = torch.linalg.solve(A, b)
        return deltas

    def cuda_sparse(self, A_cp: cp.ndarray, b: torch.Tensor):
        # st = time.time()
        #A_cupy = cp.from_dlpack(A)
        b_cupy = cp.from_dlpack(b)

        x = sp_linalg.spsolve(A_cp, b_cupy)
        x = torch.from_dlpack(x)
        # print(f'{time.time() - st :.4f}s cuda sparse solve')
        return x

    def cuda_dense(self, A: torch.Tensor, b: torch.Tensor):
        A = A.to_dense()
        # deltas = torch.linalg.solve(A, b)

        # Normalise columns to reduce numerical error
        col_norms = A.norm(dim=0, keepdim=True).clamp_min(1e-3)
        A = A / col_norms
        x3 = torch.linalg.solve(A, b)
        deltas = x3 / col_norms.squeeze(0)
        return deltas

    def cuda_amgx(self, A_cp: cp.ndarray, b: torch.Tensor):
        # Cupy to sparse is faster than torch to sparse
        self.amgx_solver.init_solver_cp(A_cp)
        x = torch.zeros_like(b)
        x, resid = self.amgx_solver.solve(b, x)
        return x

    def cuda_iterative(self, A_cp: cp.ndarray, b: torch.Tensor):
        b_cp = cp.from_dlpack(b)

        # Convert the dense matrix A_cupy to a sparse CSR matrix
        A_sparse_cupy = sp.csr_matrix(A_cp)

        # Solve the sparse linear system Ax = b using CuPy
        x, info = gmres(A_sparse_cupy, b_cp, **self.cfg)
        x = torch.from_dlpack(x)

        return x


    def cudss(self, A: torch.Tensor, b: torch.Tensor):
        x = self.cudss_solver.forward(A, b)
        #
        # A_cp = csr_torch_to_cupy(A)
        # b_cp = cp.from_dlpack(b)
        # x0 = cp.from_dlpack(x)
        # x_cp, info = gmres(A_cp, b_cp, x0=x0, **{"maxiter": 10, "restart": 10, "rtol": 1e-9})
        # x = torch.from_dlpack(x_cp)

        x, _ = gmres_cust(A, b, x0=x, **{"maxiter": 10, "restart": 10, "rtol": 1e-9})
        return x

    def preproc_tensor(self, A: torch.Tensor, b: torch.Tensor):
        """ Preprocess A matrix before solving. Do universal preprocessing, then solver specific preprocessing. """
        return self.preproc(A, b)

    def postproc_defualt(self, x: torch.Tensor):
        return x

    def preproc_default(self, A: torch.Tensor, b: torch.Tensor):
        """ Default preprocessing. """
        indptr, indices, values = csr_compress(A)
        A = torch.sparse_csr_tensor(crow_indices=indptr, col_indices=indices, values=values, size=A.size(), device=A.device)
        return A, b

    def postproc_sparse(self, x: torch.Tensor):
        if self.col_norms is not None:
            x = x / self.col_norms
        return x

    def preproc_sparse(self, A: torch.Tensor, b: torch.Tensor, precondition=False):
        """ Convert a torch tensor to a cupy sparse tensor """
        print(f'{A._nnz() = }')
        if A.is_sparse_csr:
            if precondition:
                A = A.to_dense() # Convert to dense for preprocessing
                """ Normalise rows and columns """
                # row_norms = A.norm(dim=1)
                # row_norms = row_norms + 1e-1 * torch.sign(row_norms)
                # row_norms = row_norms /row_norms.mean()
                # A = A / row_norms.unsqueeze(1)
                # b = b.squeeze() / row_norms

                col_norms = A.norm(dim=0, keepdim=True).clamp_min(1e-3)
                A = A / col_norms
                self.col_norms = col_norms.squeeze(0)
                A = A.to_sparse_csr()

            indptr, indices, values = csr_compress(A)
            # indptr, indices, values = A.crow_indices(), A.col_indices(), A.values()

            values_cp = cp.from_dlpack(values)
            indices_cp = cp.from_dlpack(indices)
            indptr_cp = cp.from_dlpack(indptr)

            A_sparse_cp = sp.csr_matrix((values_cp, indices_cp, indptr_cp), shape=A.size())
        else:
            A_cp = cp.from_dlpack(A)
            A_sparse_cp = sp.csr_matrix(A_cp)
        return A_sparse_cp, b




