import torch
from pde.config import Config
from pde.cartesian_grid.X_grid import XGrid2D

class grid_setup:
    def __init__(self, cfg: Config):

        grid_N = cfg.N
        N = torch.tensor(cfg.N)
        Xs_grid = XGrid2D(cfg.xmin, cfg.xmax, N, device=cfg.device)

        # Dirichlet BC
        dirichlet_bc = torch.full(grid_N, float('nan'))
        dirichlet_bc[-1, :] = 0.
        # dirichlet_bc[0, :] = 0
        dirichlet_bc[:, 0] = 0.  # torch.arange(0, grid_N[0]) / (grid_N[0]-1)
        dirichlet_bc[:, -1] = 0.  # torch.arange(0, grid_N[0]) / (grid_N[0]-1)

        # Neuman BC
        neuman_bc = torch.full(grid_N, float('nan'))
        neuman_bc[0, 1:-1] = 0.

        self.Xs_grid = Xs_grid
        self.dirichlet_bc = dirichlet_bc
        self.neuman_bc = neuman_bc