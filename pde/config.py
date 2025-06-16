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

@dataclass
class FwdConfig:
    # Newton Raphson PDE solver settings
    N_iter: int = 10
    # lr: float = 0.5
    solve_acc: float = 0.001
    dU_clamp: float = 0.5  # maximum change in U per iteration

    # Forward linear solver settings
    # lin_mode: LinMode = LinMode.SPARSE
    lin_mode: LinMode = LinMode.DENSE
    maxiter: int = 3000
    restart: int = 3000
    rtol: float = 1e-9
    lin_solve_cfg: dict = None
    def __post_init__(self):
        self.lin_solve_cfg = \
            {
            "config_version": 2,
            "determinism_flag": 0,
            "exception_handling": 1,

            "solver": "DENSE_LU_SOLVER",

            # "solver": {
            #     "obtain_timings": 0,
            #     # "print_solve_stats": 1,
            #     "solver": "FGMRES",  #"PBICGSTAB", #
            #     "norm": "L2",
            #     "max_iters": 100,
            #     "gmres_n_restart": 100,
            #     "gram_schmidt_options": "REORTHOGONALIZED",   # "NORMAL", "MODIFIED", "REORTHOGONALIZED"
            #     "gs_reorthog_repeat": 0,
            #     "gs_reorthog_end": 1000,
            #     "preconditioner": "NOSOLVER",
            #
            #     "preconditioner": {"solver":  #"DENSE_LU_SOLVER",
            #                         "BLOCK_JACOBI",
            #                         "relaxation_factor": .6,
            #                         "max_iters": 5,
            #                      },

                # "preconditioner": {
                #     "print_grid_stats": 1,
                #     "solver": "AMG",
                #     "smoother": {
                #         "solver":  #"DENSE_LU_SOLVER",
                #                     "JACOBI_L1",
                #                     "relaxation_factor": 1.5,
                #                     #"max_iters": 5,
                #                  },
                #     "coarse_solver": "DENSE_LU_SOLVER",
                #     "algorithm": "AGGREGATION",
                #     "selector": "SIZE_2",
                #     "max_iters": 2,
                #     "presweeps": 3,
                #     "postsweeps": 3,
                #     "cycle": "V",
                #     "max_levels":2,
                # },
            # }
        }

        if self.lin_mode == LinMode.ITERATIVE:
            self.lin_solve_cfg = {"maxiter": self.maxiter, "restart": self.restart, "rtol": self.rtol}

@dataclass
class AdjointConfig:
    # General settings
    # N_iter: int = 1

    # Linear solver settings
    lin_mode: LinMode = LinMode.AMGX
    maxiter: int = 500
    restart: int = 100
    rtol: float = 1e-4
    lin_solve_cfg: dict = None

    def __post_init__(self):
        self.lin_solve_cfg = {"maxiter": self.maxiter, "restart": self.restart, "rtol": self.rtol}
        self.lin_solve_cfg = {
            "config_version": 2,
            "determinism_flag": 0,
            "exception_handling": 1,

            "solver": "DENSE_LU_SOLVER",
        }

@dataclass
class Config:
    DEVICE: str = "cuda"

    # Phyiscal Parameters
    mu = 1.
    rho = 1000

    # Grid settings
    xmin: float = 0
    xmax: float = 1
    N: tuple[int] = (100, 125)

    # Forward PDE solver config
    fwd_cfg: FwdConfig = field(default_factory=FwdConfig)

    # Adjoint config
    adj_cfg: AdjointConfig = field(default_factory=AdjointConfig)
