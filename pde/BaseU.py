from abc import abstractmethod

import torch
from torch import Tensor
import abc


class UBase(abc.ABC):
    device: torch.device | str
    N_dim: int
    N_us_real: int  # Number of real points.

    _Us: Tensor  # Value of u at all points.
    _Xs: Tensor  # Coordinates of all points. Shape = [N_total, 2]

    updt_mask: Tensor  # Which us have gradient. Shape = [N_u_grad]
    pde_mask: Tensor  # Which PDEs are used to fit us. Automatically disregard extra points. Shape = [N_pde, ...]
    u_mask: tuple[slice, ...]  # Real us points [N_u_real]

    N_us_grad: int        # Number of points that need fitting
    N_comp: int           # Number of vector components

    pde_true_idx: Tensor
    us_grad_idx: Tensor





    @abstractmethod
    def _cuda(self):
        pass