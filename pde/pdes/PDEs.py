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

    def residuals(self, u_dus: dict[tuple, torch.Tensor], Xs: torch.Tensor) -> torch.Tensor:
        """
                f(u, du/dX, d2u/dX2, X, thetas) = 0
        Args:
            u_dus: u and all gradients at point X. Shape = [BS, N_grads+1]. Sorted by (0, 0), (1, 0), (0, 1), (2, 0), (1, 1), ...
            Xs: Grid points. Shape = [BS, 2]
        Returns: PDE residual (=0 for exact solution), shape=[BS]
        """
        u_dus = list(u_dus.values())
        exit("hi")
        return self.forward(u_dus, Xs)

    @abstractmethod
    def forward(self, u_dus: torch.Tensor, Xs: torch.Tensor, aux_input=None) -> torch.Tensor:
        """ us_dus.shape = (BS)[N_grads+1, N_vector]. Sorted by (0, 0), (1, 0), (0, 1), (2, 0), (1, 1), ...
            In vmap-able format, (bs) implicit.
            return.shape = (BS)[N_vector]
        """
        pass


class Poisson(PDEFunc):
    def __init__(self, cfg: Config, device='cpu'):
        super().__init__(cfg=cfg, device=device)
        self.to(device)

    def forward(self, u_dus: torch.Tensor, Xs: torch.Tensor, aux_input=None):
        # print(f'{u_dus.shape = }')

        u = u_dus[0]
        dudx, dudy = u_dus[1], u_dus[2]
        d2udx2, d2udxdy, d2udy2 = u_dus[3], u_dus[4], u_dus[5]

        resid = d2udy2 + d2udx2 + 0 * dudx + 0 * dudy - 5 * u + 5
        return resid


class LearnedFunc(PDEFunc):
    def __init__(self, cfg, device='cpu'):
        super().__init__(cfg=cfg, device=device)

        self.test_param = torch.nn.Parameter(torch.tensor([1, 1., -5, 5], device=device, dtype=torch.float32))
        self.to(device)

    def forward(self, u_dus: list[torch.Tensor], Xs: torch.Tensor):
        u = u_dus[0]
        dudx, dudy = u_dus[1], u_dus[2]
        d2udx2, d2udxdy, d2udy2 = u_dus[3], u_dus[4], u_dus[5]

        p1, p2, p3, p4 = self.test_param
        resid = p1 * d2udx2  + p2 * d2udy2 + p3 * u + p4

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
