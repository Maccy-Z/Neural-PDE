import torch
import scipy.sparse.linalg as linalg


import cupy as cp
import cupyx.scipy.sparse as sp
import cupyx.scipy.sparse.linalg as sp_linalg
from typing import Callable

from pde.solvers.gmres import gmres
from pde.solvers.pyamgx_holder import PyAMGXManager
from pde.config import LinMode
from pde.utils_sparse import csr_compress

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
        A, b = self.preproc_tensor(A, b)
        x, resid = self.solver(A, b)
        x = self.postproc_tensor(x)
        return x, resid


    def cuda_sparse(self, A_cp: cp.array, b: torch.Tensor):
        #A_cupy = cp.from_dlpack(A)
        b_cupy = cp.from_dlpack(b)

        # Convert the dense matrix A_cupy to a sparse CSR matrix

        # Solve the sparse linear system Ax = b using CuPy
        x = sp_linalg.spsolve(A_cp, b_cupy)

        x = torch.from_dlpack(x)
        return x, 0

    def cuda_dense(self, A: torch.Tensor, b: torch.Tensor):
        A = A.to_dense()
        # c_print(torch.linalg.matrix_rank(A), color="green")
        # c_print(A.shape, color="green")

        deltas = torch.linalg.solve(A, b)
        return deltas, 0

    def cpu_sparse(self, A: torch.Tensor, b: torch.Tensor):
        A = A.numpy()
        b = b.numpy()
        deltas = linalg.spsolve(A, b, use_umfpack=True)
        deltas = torch.from_numpy(deltas)
        return deltas

    def cpu_dense(self, A: torch.Tensor, b: torch.Tensor):
        deltas = torch.linalg.solve(A, b)
        return deltas

    def cuda_amgx(self, A_cp: cp.array, b: torch.Tensor):
        # Cupy to sparse is faster than torch to sparse
        self.amgx_solver.init_solver_cp(A_cp)
        x = torch.zeros_like(b)
        x, resid = self.amgx_solver.solve(b, x)

        # if self.inv_perm is not None:
        #     x = x[self.inv_perm]
        if self.col_norms is not None:
            x = x / self.col_norms
        # c_print(f'{b = }', color="bright_blue")
        # c_print(f'{x = }', color="bright_blue")


        return x, resid

    def cuda_iterative(self, A_cp: cp.array, b: torch.Tensor):
        b_cp = cp.from_dlpack(b)

        # Convert the dense matrix A_cupy to a sparse CSR matrix
        A_sparse_cupy = sp.csr_matrix(A_cp)

        # Solve the sparse linear system Ax = b using CuPy
        x, info = gmres(A_sparse_cupy, b_cp, **self.cfg)
        x = torch.from_dlpack(x)

        # if self.inv_perm is not None:
        #     x = x[self.inv_perm]
        if self.col_norms is not None:
            x = x / self.col_norms
        #
        # c_print(f'{b = }', color="bright_blue")
        # c_print(f'{x = }', color="bright_blue")
        return x, info["resid_norm"]

    def preproc_tensor(self, A: torch.Tensor, b: torch.Tensor):
        """ Preprocess A matrix before solving. Do universal preprocessing, then solver specific preprocessing. """
        indptr, indices, values = csr_compress(A)
        A = torch.sparse_csr_tensor(crow_indices=indptr, col_indices=indices, values=values, size=A.size(), device=A.device)
        return self.preproc(A, b)

    def postproc_tensor(self, x: torch.Tensor):
        return x

    def preproc_default(self, A: torch.Tensor, b: torch.Tensor):
        """ Default preprocessing. """

        return A, b

    def preproc_sparse(self, A: torch.Tensor, b: torch.Tensor) -> sp.csr_matrix:
        """ Convert a torch tensor to a cupy sparse tensor """
        from scipy.sparse.csgraph import reverse_cuthill_mckee
        import scipy.sparse as spsp
        from pde.utils_sparse import plot_sparsity
        from cupyx.scipy.sparse.linalg import spilu

        self.inv_perm = None
        self.col_norms = None

        if A.is_sparse_csr:
            # A = A.to_dense()
            # # Permuting
            # A_sp = A.to_sparse_coo().coalesce()
            #
            # # 2) pull out the three 1-D arrays
            # row = A_sp.indices()[0].cpu().numpy().ravel()  # shape (nnz,)
            # col = A_sp.indices()[1].cpu().numpy().ravel()  # shape (nnz,)
            # data = A_sp.values().cpu().numpy().ravel()  # shape (nnz,)
            #
            # # 3) build SciPy COO and convert to CSR (or keep COO if you like)
            # A_sp = spsp.coo_matrix((data, (row, col)), shape=A.shape).tocsr()
            # perm = reverse_cuthill_mckee(A_sp, symmetric_mode=True)
            # perm = torch.tensor(perm.copy(), dtype=torch.int64)


            #
            """ Normalise rows and columns """
            # row_norms = A.norm(dim=1)
            # row_norms = row_norms + 1e-1 * torch.sign(row_norms)
            # row_norms = row_norms /row_norms.mean()
            # A = A / row_norms.unsqueeze(1)
            # b = b.squeeze() / row_norms
            #
            # col_norms = A.norm(dim=0)
            # col_norms = col_norms + 1e-1 * torch.sign(col_norms)
            # col_norms = col_norms / col_norms.mean()
            # A = A / col_norms.unsqueeze(0)
            # self.col_norms = col_norms

            # print(f'{row_norms.abs().min() = }, {col_norms.abs().min() = }')
            """ Permutation """
            # N = A.shape[0] // 3
            # perm = torch.tensor(
            #     [j * N + i
            #      for i in range(N)  # node index
            #      for j in range(3)  # variable index: 0=Vx,1=Vy,2=P
            #      ],
            #     dtype=torch.long,
            #     device=A.device
            # )
            # A = A[:, perm][perm]
            # b = b[perm]
            # self.inv_perm = perm.argsort()

            """Fill in diagonals"""
            # diag = A.diagonal()
            # zero_mask = diag == 0
            # zero_idx = torch.where(zero_mask)[0]
            # print(f'{len(zero_idx) = }')
            # for i in zero_idx:
            #     A[i, i] = 1e-3
            #
            # plot_sparsity(A)
            #
            # exit(7)

            # A = A.to_sparse_csr()
            # values = A.values()
            # indices = A.col_indices()
            # indptr = A.crow_indices()
            # with Timer(text="Time to compress: : {:.4f}"):
            indptr, indices, values = csr_compress(A)


            values_cp = cp.from_dlpack(values)
            indices_cp = cp.from_dlpack(indices)
            indptr_cp = cp.from_dlpack(indptr)

            A_sparse_cp = sp.csr_matrix((values_cp, indices_cp, indptr_cp), shape=A.size())

            # print(A_sparse_cp[0])
            # self.A_lu = spilu(A_sparse_cp, fill_factor=1)
            # print(A_lu)
            # exit(7)
            # self._est_cond_num(A, A_sparse_cp)


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



