import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from pde.config import Config
from pde.utils import show_grid


class PDEFunc(torch.nn.Module, ABC):
    def __init__(self, cfg: Config, device='cpu'):
        """ Given u and derivatives, return the PDE residual. """
        super().__init__()
        self.device = device

    def residuals(self, u_dus: torch.Tensor, Xs: torch.Tensor, aux_input=None) -> tuple[torch.Tensor, torch.Tensor]:
        """
                f(u, du/dX, d2u/dX2, X, thetas) = 0
        Args:
            u_dus: u and all gradients at point X. Shape = [BS, N_grads+1]. Sorted by (0, 0), (1, 0), (0, 1), (2, 0), (1, 1), ...
            Xs: Grid points. Shape = [BS, 2]
        Returns: PDE residual (=0 for exact solution), shape=[BS]
        """
        residuals = self(u_dus, Xs, aux_input)
        return residuals, residuals

    @abstractmethod
    def forward(self, u_dus: torch.Tensor, Xs: torch.Tensor, aux_input=None) -> torch.Tensor:
        """ us_dus.shape = (BS)[N_grads+1, N_vector]. Sorted by (0, 0), (1, 0), (0, 1), (2, 0), (1, 1), ...
            In vmap-able format, (bs) implicit.
            return.shape = (BS)[N_vector]
        """
        pass


class HeatLearned(PDEFunc):
    def __init__(self, cfg: Config, device='cpu'):
        super().__init__(cfg=cfg, device=device)
        self.to(device)
        self.a = nn.Parameter(torch.tensor(0., device=device), requires_grad=True)

    def forward(self, u_dus: torch.Tensor, Xs: torch.Tensor, aux_input=None):
        # print(f'{u_dus.shape = }')

        u = u_dus[0]
        dudx, dudy = u_dus[1], u_dus[2]
        d2udx2, d2udxdy, d2udy2 = u_dus[3], u_dus[4], u_dus[5]

        resid = d2udy2 + d2udx2 + self.a
        return resid


class Fluid(PDEFunc):
    def __init__(self, cfg: Config, device='cpu'):
        super().__init__(cfg=cfg, device=device)
        self.to(device)

        self.mu = cfg.mu
        self.rho = cfg.rho

    def forward(self, u_dus: torch.Tensor, Xs: torch.Tensor, aux_input=None):
        """ u_dus.shape = [n_grads, n_comp]
            Xs.shape = [2]

            return.shape = [n_comp]
        """

        u = u_dus[0]
        dudx, dudy = u_dus[1], u_dus[2]
        d2udx2, d2udy2 = u_dus[3], u_dus[5]

        # Momentum equations
        advect_x = self.rho * (u[0] * dudx[0] + u[1] * dudy[0])
        advect_y = self.rho * (u[0] * dudx[1] + u[1] * dudy[1])
        laplace_Vx = self.mu * (d2udx2[0] + d2udy2[0])
        laplace_Vy = self.mu * (d2udx2[1] + d2udy2[1])
        dpdx = dudx[2]
        dpdy = dudy[2]

        resid_x = -dpdx + laplace_Vx - advect_x
        resid_y = -dpdy + laplace_Vy - advect_y

        divergence = dudx[0] + dudy[1]
        # divergence = 1 - 1 / 2 * x - u[2]

        resid = torch.stack([resid_x, resid_y, divergence], dim=-1)

        return resid


class FluidLearned(PDEFunc):
    def __init__(self, cfg: Config, device='cpu'):
        super().__init__(cfg=cfg, device=device)
        self.to(device)

        self.mu = cfg.mu
        self.rho = cfg.rho
        self.a = nn.Parameter(torch.tensor(0., device=device), requires_grad=True)


    def forward(self, u_dus: torch.Tensor, Xs: torch.Tensor, aux_input=None):
        """ u_dus.shape = [n_grads, n_comp]
            Xs.shape = [2]

            return.shape = [n_comp]
        """

        u = u_dus[0]
        dudx, dudy = u_dus[1], u_dus[2]
        d2udx2, d2udy2 = u_dus[3], u_dus[5]

        # Momentum equations
        advect_x = self.rho * (u[0] * dudx[0] + u[1] * dudy[0])
        advect_y = self.rho * (u[0] * dudx[1] + u[1] * dudy[1])
        laplace_Vx = self.mu * (d2udx2[0] + d2udy2[0])
        laplace_Vy = self.mu * (d2udx2[1] + d2udy2[1])
        dpdx = dudx[2]
        dpdy = dudy[2]

        resid_x = -dpdx + laplace_Vx - advect_x + self.a
        resid_y = -dpdy + laplace_Vy - advect_y + self.a

        divergence = dudx[0] + dudy[1]
        # divergence = 1 - 1 / 2 * x - u[2]

        resid = torch.stack([resid_x, resid_y, divergence], dim=-1)

        return resid

class NNFunc(PDEFunc):
    def __init__(self, cfg, device='cuda'):
        super().__init__(cfg=cfg, device=device)

        self.lin1 = nn.Linear(5, 32)
        self.lin2 = nn.Linear(32, 1)

        nn.init.zeros_(self.lin2.bias)

        self.to(device)

    def forward(self, u_dus: tuple[torch.Tensor, ...], Xs: torch.Tensor):
        u, dudX, d2udX2 = u_dus

        in_state = torch.cat([u, dudX, d2udX2], dim=-1)
        f = self.lin1(in_state)
        f = self.lin2(f).squeeze()

        return f
