import torch
import scipy.sparse.linalg as linalg
import time
import cupy as cp
import cupyx.scipy.sparse as sp
import cupyx.scipy.sparse.linalg as sp_linalg
from typing import Callable
from cprint import c_print

from pde.solvers.gmres import gmres, gmres_cust
from pde.solvers.pyamgx_holder import PyAMGXManager
from pde.solvers.nvmath_cudss import CUDSSSolver
from pde.config import LinMode, FwdConfig, AdjConfig
from pde.utils_sparse import csr_compress, csr_normalise

class LinearSolver:
    """ Solve Ax = b for x """
    solver: Callable
    preproc: Callable
    postproc: Callable
    solver_cfg: dict = None
    col_norms: torch.Tensor | None = None

    def __init__(self, mode: LinMode, device: str, cfg: FwdConfig|AdjConfig=None):
        self.cfg = cfg
        self.preproc = self.preproc_default
        self.postproc = self.postproc_defualt

        if device == "cuda":
            self.preproc = self.preproc_sparse
            self.postproc = self.postproc_sparse

            if mode == LinMode.DENSE:
                self.solver = self.cuda_dense
            elif mode == LinMode.SPARSE:
                self.solver = self.cuda_sparse
            elif mode == LinMode.ITERATIVE:
                self.solver = self.cuda_iterative
                self.solver_cfg = cfg.solver_cfg
            elif mode == LinMode.AMGX:
                self.solver_cfg = cfg.solver_cfg
                self.amgx_solver = PyAMGXManager().create_solver(self.solver_cfg)
                self.solver = self.cuda_amgx
            elif mode == LinMode.CUDSS:
                self.cudss_solver = CUDSSSolver(cfg.solver_cfg)
                self.solver = self.cudss
                self.gmres_cfg = cfg.gmres_cfg

        elif device == "cpu":
            if mode == LinMode.DENSE:
                self.solver = self.cpu_dense
            elif mode == LinMode.SPARSE:
                self.solver = self.cpu_sparse

    def solve(self, A, b) -> torch.Tensor:
        A_proc, b_proc = self.preproc_tensor(A, b)
        x = self.solver(A_proc, b_proc)
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
        deltas = torch.linalg.solve(A, b)
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
        x, info = gmres(A_sparse_cupy, b_cp, **self.solver_cfg)
        x = torch.from_dlpack(x)

        return x

    def cudss(self, A: torch.Tensor, b: torch.Tensor):
        """ Solve using CUDSS.
            Optional GMRES refinement, then dense solve if not converged.  """
        x = self.cudss_solver.forward(A, b)
        rel_err = torch.norm(A @ x - b) / (torch.norm(b) + 1e-7)
        if rel_err > 1e-3:
            x, info = gmres_cust(A, b, x0=x, **self.gmres_cfg)
            rel_err2 = torch.norm(A @ x - b) / (torch.norm(b) + 1e-7)
            # print(f'CUDSS rel err: {rel_err:.3g} -> GMRES rel err: {rel_err2:.3g}, {info['residual_norm']:2g}')
            if rel_err2 > 3e-3:
                # Use old solution as start. x = x2 + A^-1 (b - A x2)
                # c_print("Dense solve fallback", color="yellow")
                r2 = b - A @ x
                x = x + self.cuda_dense(A, r2)
                rel_err3 = torch.norm(A @ x - b) / (torch.norm(b) + 1e-7)
                c_print(f'Dense rel err: {rel_err3:.3g}', color="yellow")

                self.cudss_solver.n_uses = self.cudss_solver.max_n_uses + 1  # Force replan next solve.
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

    def preproc_sparse(self, A: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ Normalise and simplify sparse matrix A before solving. """
        # Compress CSR matrix
        if self.cfg.csr_compress:
            crow_indices, col_indices, values = csr_compress(A) # crow_indices, col_indices, values
            A = torch.sparse_csr_tensor(crow_indices=crow_indices, col_indices=col_indices, values=values, size=A.size(), device=A.device)
        # Normalize rows and columns
        A, b, self.col_norms = csr_normalise(A, b, self.cfg.norm_col, self.cfg.norm_row)
        return A, b




