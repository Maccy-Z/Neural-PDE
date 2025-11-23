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


    def update_grid(self, deltas):
        """
        Update grid with changes, and fix boundary conditions with new grid.
        deltas.shape = [N*N_comp]
        us -> us - deltas
        """
        deltas = deltas.view(-1, self.N_comp)
        # self._Us -= deltas
        self.set_grid(self._Us - deltas)

    def get_test_update(self, deltas):
        """
        Get test update for grid with changes, without applying them.
        deltas.shape = [N*N_comp]
        us -> us - deltas
        """
        deltas = deltas.view(-1, self.N_comp)
        us_test = torch.clone(self._Us) - deltas
        return us_test

    def get_us_mask(self):
        """
        Return us, and mask of which elements are trainable. Used for masking Jacobian equations.
        """
        return self._Us, self.updt_mask, self.pde_mask


    def get_all_us_Xs(self):
        """ Return all grid points, including fake boundaries. """
        return self._Us, self._Xs


    @abstractmethod
    def _cuda(self):
        pass