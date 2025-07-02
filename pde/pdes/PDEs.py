import torch
import torch.nn as nn
import torch.nn.functional as F

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
            In vmap-able format, (BS) implicit.
            return.shape = (BS)[N_vector]
        """
        pass

class Dummy(PDEFunc):
    def __init__(self, cfg: Config, device='cpu'):
        super().__init__(cfg=cfg, device=device)
        self.to(device)

        self.a = nn.Parameter(torch.tensor(3.1, device=device), requires_grad=True)


    def forward(self, u_dus: torch.Tensor, Xs: torch.Tensor, aux_input=None):
        """ Dummy PDE function for testing. """

        return u_dus[0] * (self.a + 1) + self.a  # Return a constant value of 10 for all components

class Heat(PDEFunc):
    def __init__(self, cfg: Config, device='cpu'):
        super().__init__(cfg=cfg, device=device)
        self.to(device)

    def forward(self, u_dus: torch.Tensor, Xs: torch.Tensor, aux_input=None):
        # print(f'{u_dus.shape = }')
        x, y = Xs
        u = u_dus[0]
        dudx, dudy = u_dus[1], u_dus[2]
        d2udx2, d2udxdy, d2udy2 = u_dus[3], u_dus[4], u_dus[5]

        resid = d2udy2 + d2udx2
        # resid = u[0] - x
        return resid


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
        x, y = Xs
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

        """ Testing """
        # resid_x = u[0] # -dpdx + laplace_Vx
        # resid_y = 2*u[1] # -dpdy + laplace_Vy
        # divergence =  1 - 1 / 2 * x - u[2]

        resid = torch.stack([resid_x, resid_y, divergence], dim=-1)

        return resid


class FluidLearned(PDEFunc):
    def __init__(self, cfg: Config, device='cpu'):
        super().__init__(cfg=cfg, device=device)
        self.to(device)

        self.mu = cfg.mu
        self.rho = cfg.rho
        self.a = nn.Parameter(torch.tensor([100., 100.], device=device), requires_grad=True)


    def forward(self, u_dus: torch.Tensor, Xs: torch.Tensor, aux_input=None):
        """ u_dus.shape = [n_grads, n_comp]
            Xs.shape = [2]

            return.shape = [n_comp]
        """

        x, y = Xs
        u = u_dus[0]
        dudx, dudy = u_dus[1], u_dus[2]
        d2udx2, d2udy2 = u_dus[3], u_dus[5]

        # Momentum equations
        advect_x = u[0] * dudx[0] + u[1] * dudy[0]
        advect_y = u[0] * dudx[1] + u[1] * dudy[1]
        laplace_Vx = d2udx2[0] + d2udy2[0]
        laplace_Vy = d2udx2[1] + d2udy2[1]
        dpdx = dudx[2]
        dpdy = dudy[2]
        resid_x = -dpdx + self.mu * laplace_Vx - self.a[0] * advect_x
        resid_y = -dpdy + self.mu * laplace_Vy - self.a[1] * advect_y
        divergence = dudx[0] + dudy[1]


        resid = torch.stack([resid_x, resid_y, divergence], dim=-1)

        # print(advect_x.mean())
        return resid


class NNFunc(PDEFunc):
    def __init__(self, cfg, device='cuda'):
        super().__init__(cfg=cfg, device=device)

        self.lin1 = nn.Linear(2, 3)
        self.lin2 = nn.Linear(32, 3)

        nn.init.zeros_(self.lin2.bias)
        nn.init.zeros_(self.lin2.weight)

        self.to(device)

    def forward(self, u_dus: torch.Tensor, Xs: torch.Tensor, aux_input=None):#
        """ us_dus.shape = (BS)[N_grads+1, N_vector] """
        u = u_dus[0]
        dudx, dudy = u_dus[1], u_dus[2]
        d2udx2, d2udy2 = u_dus[3], u_dus[5]

        # Base residual
        advect_x = 100 *(u[0] * dudx[0] + u[1] * dudy[0])
        advect_y = 100 * (u[0] * dudx[1] + u[1] * dudy[1])
        laplace_Vx = 1 * (d2udx2[0] + d2udy2[0])
        laplace_Vy = 1 * (d2udx2[1] + d2udy2[1])
        dpdx = dudx[2]
        dpdy = dudy[2]

        resid_x = -dpdx + laplace_Vx - advect_x
        resid_y = -dpdy + laplace_Vy - advect_y

        divergence = dudx[0] + dudy[1]

        resid = torch.stack([resid_x, resid_y, divergence], dim=-1)

        # Neural Network component
        in_state = torch.stack([u[0] * dudx[0]+ u[1] * dudy[0], u[0] * dudx[1]+ u[1] * dudy[1]], dim=0)
        f = self.lin1(in_state)
        # f = F.leaky_relu(f)
        f = self.lin2(f).squeeze()
        f[2] *= 0
        resid = resid + f
        return resid
