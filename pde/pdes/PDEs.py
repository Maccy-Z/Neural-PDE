import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod
from pde.config import Config
from .MLP_mup import MLP_mup, get_MLP_mup, MLP
from pde.utils import unwrap_vmap

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
        # u_dus = 1e5 * torch.tanh(u_dus / 1e5)

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


        # advect_x = 1e2 * torch.tanh(advect_x / 1e3)
        # advect_y = 1e2 * torch.tanh(advect_y / 1e3)

        resid_x = -dpdx + laplace_Vx - advect_x
        resid_y = -dpdy + laplace_Vy - advect_y

        divergence = dudx[0] + dudy[1]

        # print("gradient:", torch.func.debug_unwrap(u_dus).abs().max())
        # print("Advection:", torch.func.debug_unwrap(advect_x).abs().max())

        resid = torch.stack([resid_x, resid_y, divergence], dim=-1)

        return resid


class FluidLearned(PDEFunc):
    # scaling = [ 0.0125,  0.033, 0.033, 0.27, 0.27, 0.27]
    def __init__(self, cfg: Config, device='cpu'):
        super().__init__(cfg=cfg, device=device)
        self.to(device)

        # self.mu = cfg.mu
        # self.rho = cfg.rho
        self.rhos = nn.Parameter(torch.tensor([0., 0.], device=device), requires_grad=True)
        self.mu = nn.Parameter(torch.tensor(1., device=device), requires_grad=True)

        self.rescaling = torch.tensor([ [1.2500e-02, 1.2500e-02, 4.7587e-01],
                                        [3.3000e-02, 3.3000e-02, 6.8853e-01],
                                        [3.3000e-02, 3.3000e-02, 4.9480e-01],
                                        [2.7000e-01, 2.7000e-01, 6.0260e+01],
                                        [2.7000e-01, 2.7000e-01, 1.6940e+01],
                                        [2.7000e-01, 2.7000e-01, 6.0878e+01]], device=device)

    def forward(self, u_dus: torch.Tensor, Xs: torch.Tensor, aux_input=None):
        """ u_dus.shape = [n_grads, n_comp]
            Xs.shape = [2]

            return.shape = [n_comp]
        """

        x, y = Xs
        # Rescale input equations
        u_dus = u_dus / self.rescaling


        u = u_dus[0]
        dudx, dudy = u_dus[1], u_dus[2]
        d2udx2, d2udy2 = u_dus[3], u_dus[5]

        # u = u / self.scaling[0]
        # dudx, dudy = dudx / self.scaling[1], dudy / self.scaling[1]
        # d2udx2, d2udy2 = d2udx2 / self.scaling[3], d2udy2 / self.scaling[3]
        # mu = self.mu * self.rescaling[3, 0]
        # a = self.a * self.scaling[0] * self.scaling[1]

        # Momentum equations
        dpdx = dudx[2]
        dpdy = dudy[2]
        advect_x = u[0] * dudx[0] + u[1] * dudy[0]
        advect_y = u[0] * dudx[1] + u[1] * dudy[1]
        laplace_Vx = d2udx2[0] + d2udy2[0]
        laplace_Vy = d2udx2[1] + d2udy2[1]

        resid_x = -dpdx + self.mu * laplace_Vx - self.rhos[0] * advect_x
        resid_y = -dpdy + self.mu * laplace_Vy - self.rhos[1] * advect_y
        divergence = dudx[0] + dudy[1]

        resid = torch.stack([resid_x, resid_y, divergence], dim=-1)
        return resid


class NNFunc(PDEFunc):
    def __init__(self, cfg, norm_mean, norm_std, device='cuda'):
        super().__init__(cfg=cfg, device=device)

        self.mlp: MLP_mup = get_MLP_mup(in_dim=8, out_dim=3, width=1024, n_hidden=1, activation=F.relu
                                        , zero_out=True)
        # self.mlp = MLP(in_dim=8, out_dim=3, width=32, n_hidden=0, activation=F.leaky_relu)

        mu = torch.tensor(1, device=device, dtype=torch.float32)
        self.other_params = nn.ParameterDict({"mu": torch.nn.Parameter(mu)})


        self.to(device)
        # self.norm_std = torch.tensor([ [1.2500e-02, 1.2500e-02, 4.7587e-01],
        #                                 [3.3000e-02, 3.3000e-02, 6.8853e-01],
        #                                 [3.3000e-02, 3.3000e-02, 4.9480e-01],
        #                                 [2.7000e-01, 2.7000e-01, 6.0260e+01],
        #                                 [2.7000e-01, 2.7000e-01, 1.6940e+01],
        #                                 [2.7000e-01, 2.7000e-01, 6.0878e+01]], device=device)
        self.norm_mean = norm_mean
        self.norm_std = norm_std


    def forward(self, u_dus: torch.Tensor, Xs: torch.Tensor, aux_input=None):#
        """ us_dus.shape = (BS)[N_grads+1, N_vector] """
        mu = self.other_params['mu']

        # Rescale input equations
        u_dus = (u_dus - self.norm_mean) / self.norm_std

        u = u_dus[0]
        dudx, dudy = u_dus[1], u_dus[2]
        d2udx2, d2udy2 = u_dus[3], u_dus[5]

        # Base residual
        # advect_x = (u[0] * dudx[0] + u[1] * dudy[0])
        # advect_y = (u[0] * dudx[1] + u[1] * dudy[1])
        laplace_Vx = d2udx2[0] + d2udy2[0]
        laplace_Vy = d2udx2[1] + d2udy2[1]
        dpdx = dudx[2]
        dpdy = dudy[2]

        resid_x = -dpdx + mu * laplace_Vx #- advect_x
        resid_y = -dpdy + mu * laplace_Vy #- advect_y

        divergence = dudx[0] + dudy[1]

        resid = torch.stack([resid_x, resid_y, divergence], dim=-1)

        # Neural Network component
        # in_state = torch.stack([advect_x, advect_y], dim=0)
        in_state = torch.stack([u[0], u[1], dudx[0], dudx[1], d2udx2[0], d2udx2[1], d2udy2[0], d2udy2[1]], dim=0)

        f = self.mlp(in_state)  # [3]
        # f = self.lin1(in_state)
        # f = F.leaky_relu(f)
        # f = self.lin2(f).squeeze()
        # f[2] *= 0
        resid = resid + f
        return resid
