import nvmath.sparse.advanced as nvmath_advanced
import torch
import cupyx.scipy.sparse as sp
import cupy as cp
import logging

from pde.utils_sparse import csr_compress
from pde.utils import ARTEFACT_DIR

class CUDSSSolver:
    crow_indices: torch.Tensor
    col_indices: torch.Tensor
    solver: nvmath_advanced.DirectSolver

    A_cp: sp.csr_matrix

    def __init__(self, cfg: dict):
        self.solver = None
        self.crow_indices = torch.tensor([], device='cuda', dtype=torch.int32)
        self.col_indices = torch.tensor([], device='cuda', dtype=torch.int32)

        self.options = nvmath_advanced.DirectSolverOptions(logger=logging.Logger("CUDSSSolver"),
                                                           multithreading_lib="/home/maccyz/miniforge3/envs/test/lib/python3.13/site-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so.0")
        self.ir_n_steps = cfg['ir_n_steps']
        self.norm_row = cfg['norm_mode'] == "row"
        self.norm_col = cfg['norm_mode'] == "col"

    def forward(self, A: torch.Tensor, b):
        """ Solve Ax = b using CUDSS
            A: Sparse csr matrix
            b: dense RHS
        """
        if self.norm_col:
            A = A.to_dense()
            col_norms = A.norm(dim=0, keepdim=True).clamp_min(1e-3)
            A = A / col_norms
            A = A.to_sparse_csr()
        if self.norm_row:
            A = A.to_dense()
            row_norms = A.norm(dim=1, keepdim=True).clamp(min=1e-3)
            A = A / row_norms
            b = b / row_norms.squeeze(-1)
            A = A.to_sparse_csr()

        crow_indices, col_indices, values = csr_compress(A)
        values = cp.from_dlpack(values)
        b_cp = cp.from_dlpack(b)

        # If A has the same sparsity, use previous solver.
        if torch.equal(crow_indices, self.crow_indices) and torch.equal(col_indices, self.col_indices):
            # Same sparsity pattern. Reuse solver and sparse matrix.
            self.A_cp.data[...] = values
            self.solver.reset_operands(b=b_cp)
        else:
            self.crow_indices = crow_indices
            self.col_indices = col_indices

            # Convert to cupy
            indices = cp.from_dlpack(col_indices)
            indptr = cp.from_dlpack(crow_indices)
            self.A_cp = sp.csr_matrix((values, indices, indptr), shape=A.size())

            # Initialize solver new solver
            self._init_solver(self.A_cp, b_cp)

        # Solve
        self.solver.factorize()
        x_cp = self.solver.solve()
        x = torch.from_dlpack(x_cp)

        if self.norm_col:
            return x / col_norms.squeeze(0)
        else:
            return x

    def _init_solver(self, A_cp: sp.csr_matrix, b_cp: cp.ndarray):
        """ Initialize the CUDSS solver with planning done once."""
        if self.solver is not None:
            self.solver.free()

        self.solver = nvmath_advanced.DirectSolver(A_cp, b_cp, options=self.options)
        self.solver.plan()

        solution_config = self.solver.solution_config
        solution_config.ir_num_steps = self.ir_n_steps
        self.solver.factorization_config.pivot_eps=1e-5
        self.solver.factorization_config.pivot_eps_algorithm = 0