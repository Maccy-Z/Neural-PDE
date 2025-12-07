import nvmath.sparse.advanced as nvmath_advanced
import torch
import cupyx.scipy.sparse as sp
import cupy as cp
import logging

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

        self.n_uses = 0 # Replan every so often. Matching step depends on values.
        self.max_n_uses = cfg['max_n_uses']

    def forward(self, A: torch.Tensor, b):
        """ Solve Ax = b using CUDSS
            A: Sparse csr matrix
            b: dense RHS
        """
        crow_indices, col_indices, values = A.crow_indices(), A.col_indices(), A.values()
        crow_indices, col_indices = crow_indices.to(dtype=torch.int32), col_indices.to(dtype=torch.int32)
        values = cp.from_dlpack(values)
        b_cp = cp.from_dlpack(b)

        # If A has the same sparsity, use previous solver.
        if (torch.equal(crow_indices, self.crow_indices) and torch.equal(col_indices, self.col_indices)
                and self.n_uses < self.max_n_uses):
            # Same sparsity pattern. Reuse solver and sparse matrix.
            self.A_cp.data[...] = values
            self.solver.reset_operands(b=b_cp)

            self.n_uses += 1
        else:
            self.crow_indices = crow_indices
            self.col_indices = col_indices

            # Convert to cupy
            indices = cp.from_dlpack(col_indices)
            indptr = cp.from_dlpack(crow_indices)
            self.A_cp = sp.csr_matrix((values, indices, indptr), shape=A.size())

            # Initialize solver new solver
            self._init_solver(self.A_cp, b_cp)

            self.n_uses = 0

        # Solve
        self.solver.factorize()
        x_cp = self.solver.solve()
        x = torch.from_dlpack(x_cp)

        return x

    def _init_solver(self, A_cp: sp.csr_matrix, b_cp: cp.ndarray):
        """ Initialize the CUDSS solver with planning done once."""
        if self.solver is not None:
            self.solver.free()

        self.solver = nvmath_advanced.DirectSolver(A_cp, b_cp, options=self.options)

        self.solver.plan_config.use_matching = 1
        self.solver.plan_config.matching_algorithm = 0
        plan = self.solver.plan()

        self.solver.factorization_config.pivot_eps = 1e-5
        self.solver.factorization_config.pivot_eps_algorithm = 0

        self.solver.solution_config.ir_num_steps = self.ir_n_steps
