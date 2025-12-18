import torch
import torch.nn as nn
from abc import abstractmethod

from pde.graph_grid.U_graph import UValues


class Loss(nn.Module):
    Us_pred: torch.Tensor = None
    loss_out: torch.Tensor = None
    requires_grad: bool

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, us_pred, requires_grad=True):
        """
            us_pred: Predicted value
            requires_grad: If True, the loss will be differentiable with respect to us_pred.
            Returns: Scalar loss
        """
        pass

    def gradient(self):
        """
            Returns: Gradient of loss wrt us_pred. Shape = us_pred.shape
        """
        assert self.requires_grad
        return torch.autograd.grad(outputs=self.loss_out, inputs=self.Us_pred)[0]

    def save_for_backward(self, us_pred, requires_grad=True):
        """
            Save the predicted values for backward pass.
            us_pred: Predicted values, in flattened form
            requires_grad: If True, the loss will be differentiable with respect to us_pred.
        """
        self.Us_pred = us_pred
        self.requires_grad = requires_grad
        self.Us_pred.requires_grad_(requires_grad)


class MSELoss(Loss):
    def __init__(self, us_true):
        super().__init__()
        self.us_true = us_true

    def forward(self, us_pred):
        self.save_for_backward(us_pred, requires_grad=True)
        #print(f'{us_pred.shape = }, {self.us_true.shape = }')
        loss = torch.mean((self.us_pred - self.us_true)**2)
        self.loss_out = loss

        return loss

    def gradient(self):
        with torch.no_grad():
            grads =  2 * (self.us_pred - self.us_true) / self.us_true.numel()
        return grads


class MSELoss2(Loss):
    def __init__(self, Us_true):
        super().__init__()
        self.Us_true = Us_true

    def forward(self, Us_pred: torch.Tensor, requires_grad=True):
        self.save_for_backward(Us_pred, requires_grad=requires_grad)
        error = Us_pred.flatten() - self.Us_true.flatten()
        loss = (error ** 2).mean()
        self.loss_out = loss
        return loss


class MSELossNorm(Loss):
    def __init__(self, norm_std: torch.Tensor = None):
        """ Normalised MSE loss.
            norm_std: shape [n_comp], standard deviation for each component to normalise by.
                        If None, normalize each sample independently.
        """
        super().__init__()

        if norm_std is not None:
            self.norm_std = norm_std.unsqueeze(0)      # Shape [1, n_comp]
        else:
            self.norm_std = None

    def forward(self, Us_pred: UValues, Us_true: UValues, requires_grad=True):
        Us_pred, Us_true = Us_pred.Us, Us_true.Us
        self.save_for_backward(Us_pred, requires_grad=requires_grad)

        if self.norm_std is not None:
            stds = self.norm_std
        else:
            stds = Us_true.std(dim=0, keepdim=True) + 0.01

        error = (Us_pred - Us_true) / stds
        loss = (error ** 2).mean()
        self.loss_out = loss
        return loss

class DummyLoss(Loss):
    def __init__(self):
        super().__init__()

    def forward(self):
        return None


class MaskLoss(Loss):
    def __init__(self, mask):
        super().__init__()
        self.mask = mask.bool()

    def forward(self, Us_pred: torch.Tensor):
        self.save_for_backward(Us_pred, requires_grad=True)
        loss = torch.mean((self.Us_pred * self.mask)**2)
        self.loss_out = loss

        print(self.Us_pred * self.mask)
        return loss



def main():
    us_true = torch.tensor([1., 2., 3.])
    us_pred = torch.tensor([1., 3., 3.])
    loss_fn = MSELoss2(us_true)
    loss = loss_fn(us_pred)

    grads = loss_fn.gradient()
    print(grads)
    # loss.backward()
    # print(us_pred.grad)


if __name__ == "__main__":
    main()
