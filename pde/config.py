from dataclasses import dataclass, field
from enum import StrEnum

class JacMode(StrEnum):
    DENSE = "dense"
    SPLIT = "split"
    SPARSE = "sparse"
    GRAPH = "graph"

class LinMode(StrEnum):
    SPARSE = "sparse"
    DENSE = "dense"
    ITERATIVE = "iterative"
    AMGX = "amgx"
    CUDSS = "cudss"


amgx_cfg = {
            "config_version": 2,
            "determinism_flag": 0,
            "exception_handling": 1,

            "solver": "DENSE_LU_SOLVER",

            # "solver": {
            #     "obtain_timings": 0,
            #     # "print_solve_stats": 1,
            #     "solver": "FGMRES",  #"PBICGSTAB", #
            #     "norm": "L2",
            #     "max_iters": 120,
            #     "gmres_n_restart": 120,
            #     "gram_schmidt_options": "REORTHOGONALIZED",   # "NORMAL", "MODIFIED", "REORTHOGONALIZED"
            #     "gs_reorthog_repeat": 1,
            #     "gs_reorthog_end": 1000,
            #     "preconditioner": "NOSOLVER",
            #
            #     "preconditioner": {"solver": "JACOBI_L1", # "BLOCK_JACOBI", #"DENSE_LU_SOLVER",
            #                         "relaxation_factor": .5,
            #                         "max_iters": 3,
            #                      },
            #
            #     # "preconditioner": {
            #     #     # "print_grid_stats": 1,
            #     #     "solver": "AMG",
            #     #     "smoother": {
            #     #         "solver":  #"DENSE_LU_SOLVER",
            #     #                     "JACOBI_L1",
            #     #                     "relaxation_factor": 1.5,
            #     #                     #"max_iters": 5,
            #     #                  },
            #     #     "coarse_solver": "DENSE_LU_SOLVER",
            #     #     "algorithm": "AGGREGATION",
            #     #     "selector": "SIZE_2",
            #     #     "max_iters": 2,
            #     #     "presweeps": 3,
            #     #     "postsweeps": 3,
            #     #     "cycle": "V",
            #     #     "max_levels":2,
            #     # },
            # }
        }

@dataclass
class FwdConfig:
    # Newton Raphson PDE solver settings
    N_iter: int = 10
    solve_acc: float = 0.001

    # Forward linear solver settings
    lin_mode: LinMode = LinMode.CUDSS
    solver_cfg: dict = None
    gmres_cfg = {"maxiter": 60, "restart": 60, "rtol": 1e-3}

    norm_row: bool = True       # Normalize rows of A
    norm_col: bool = False      # Normalize columns of A
    csr_compress: bool = False   # Remove zero entries before solving

    def __post_init__(self):
        if self.lin_mode == LinMode.AMGX:
            self.solver_cfg = amgx_cfg
        elif self.lin_mode == LinMode.ITERATIVE:
            self.solver_cfg = {"maxiter": 3000, "restart": 3000, "rtol": 1e-9}
        elif self.lin_mode == LinMode.CUDSS:
            self.solver_cfg = {'ir_n_steps': 1, 'max_n_uses': 300}


@dataclass
class AdjConfig:
    # Linear solver settings
    lin_mode: LinMode = LinMode.CUDSS
    solver_cfg: dict = None
    gmres_cfg = {"maxiter": 60, "restart": 60, "rtol": 1e-3}

    norm_row: bool = False      # Normalize rows of A
    norm_col: bool = True       # Normalize columns of A
    csr_compress: bool = True   # Remove zero entries before solving

    def __post_init__(self):
        if self.lin_mode == LinMode.AMGX:
            self.solver_cfg = amgx_cfg
        elif self.lin_mode == LinMode.ITERATIVE:
            self.solver_cfg = {"maxiter": 500, "restart": 100, "rtol": 1e-4}
        elif self.lin_mode == LinMode.CUDSS:
            self.solver_cfg = {'ir_n_steps': 2, 'max_n_uses': 300}
            self.gmres_cfg = {"maxiter": 60, "restart": 60, "rtol": 1e-3}

@dataclass
class Config:
    device: str = "cuda"

    # Phyiscal Parameters
    mu = 1.
    rho = 1000

    # Grid settings
    xmin: float = 0
    xmax: float = 1
    N: tuple[int] = (100, 125)
    # Derivative settings
    max_degree: int = 2     # Max degree of derivative to compute
    grad_neigh: int = 16    # Number of neighbors to use for gradient computations

    # Forward PDE solver config
    fwd_cfg: FwdConfig = field(default_factory=FwdConfig)

    # Adjoint config
    adj_cfg: AdjConfig = field(default_factory=AdjConfig)

    # Training settings
    N_steps = 2001
    mup_lr = 0.02
    mup_betas = (0.9, 0.99)
    mup_wd = 1e-4
    scalar_lr = 0.001

    clip_norm = 0.5

    N_print = 50
    N_valid = 100