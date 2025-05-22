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

    # Forward linear solver settings
    maxiter: int = 500
    restart: int = 500
    rtol: float = 1e-9
    lin_solve_cfg: dict = None

    # Jacobian mode
    jac_mode: JacMode = JacMode.GRAPH

    # Newton Raphson PDE solver settings
    # lin_mode: LinMode = LinMode.SPARSE
    lin_mode: LinMode = LinMode.AMGX
    N_iter: int = 0
    lr: float = 1.
    acc: float = 0.

    def __post_init__(self):
        self.lin_solve_cfg = \
            {
            "config_version": 2,
            "determinism_flag": 0,
            "exception_handling": 1,

            "solver": "DENSE_LU_SOLVER",

            "solver": {
                "obtain_timings": 0,
                # "print_solve_stats": 1,
                "solver": "FGMRES",  #"PBICGSTAB", #
                "norm": "L2",
                "max_iters": 1,
                "gmres_n_restart": 1,
                "gram_schmidt_options": "REORTHOGONALIZED",   # "NORMAL", "MODIFIED", "REORTHOGONALIZED"
                "gs_reorthog_repeat": 2,
                "preconditioner": "NOSOLVER",

                # "preconditioner": {"solver":  #"DENSE_LU_SOLVER",
                #                     "BLOCK_JACOBI",
                #                     "relaxation_factor": .6,
                #                     "max_iters": 30,
                #                  },

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
            }
        }

        if self.lin_mode == LinMode.ITERATIVE:
            self.lin_solve_cfg = {"maxiter": self.maxiter, "restart": self.restart, "rtol": self.rtol}



@dataclass
class AdjointConfig:
    # Jacobian mode
    num_blocks: int = 4
    jac_mode: JacMode = JacMode.GRAPH

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

            "solver": {
                "monitor_residual": 1,
                # "print_solve_stats": 1,
                "solver": "PBICGSTAB",
                "convergence": "RELATIVE_INI_CORE",
                "tolerance": 1e-4,
                "max_iters": 25,
                # "gmres_n_restart": 75,
                "preconditioner": {
                    # "solver": "NOSOLVER",
                    "solver": "AMG",
                    "algorithm": "AGGREGATION",
                    "selector": "SIZE_2",
                    "max_iters": 1,
                    "cycle": "V",
                    # "max_levels": 5,
                }
            }
        }

@dataclass
class Config:
    DEVICE: str = "cuda"

    # Phyiscal Parameters
    mu = 1.0
    rho = 1000

    # Grid settings
    xmin: float = 0
    xmax: float = 1
    N: tuple[int] = (100, 125)

    # Forward PDE solver config
    fwd_cfg: FwdConfig = field(default_factory=FwdConfig)

    # Adjoint config
    adj_cfg: AdjointConfig = field(default_factory=AdjointConfig)
