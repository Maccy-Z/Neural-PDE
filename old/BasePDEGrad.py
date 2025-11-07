from abc import ABC, abstractmethod
from pde.BaseU import UBase
import torch

class PDEFwdBase(ABC):
    @abstractmethod
    def residuals(self, subgrid: UBase, us_grad: torch.Tensor=None):
        """  Returns residuals of equations that require gradients only. """
        pass

    def only_resid(self):
        """ Only returns residuals. Used for tracking solve progress."""
        pass

    def derivative(self, us):
        raise NotImplementedError
#