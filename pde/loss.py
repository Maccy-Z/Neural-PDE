import torch
import torch.nn as nn
from abc import abstractmethod

class Loss(nn.Module):
    us_pred: torch.Tensor = None
    loss_out: torch.Tensor = None

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, us_pred):
        """
            us_pred: Predicted values, in flattened form
            Returns: Scalar loss
        """
        pass

    def gradient(self):
        """
            Returns: Gradient of loss wrt us_pred. Shape = us_pred.shape
        """
        return torch.autograd.grad(outputs=self.loss_out, inputs=self.us_pred)[0]

class MSELoss(Loss):
    def __init__(self, us_true):
        super().__init__()
        self.us_true = us_true

    def forward(self, us_pred):
        self.us_pred = us_pred
        #print(f'{us_pred.shape = }, {self.us_true.shape = }')
        loss = torch.mean((self.us_pred - self.us_true)**2)
        self.loss_out = loss

        return loss

    def gradient(self):
        with torch.no_grad():
            grads =  2 * (self.us_pred - self.us_true) / self.us_true.numel()
        return grads


class MSELoss2(Loss):
    def __init__(self, us_true):
        super().__init__()
        self.us_true = us_true

    def forward(self, us_pred: torch.Tensor):
        us_pred.requires_grad_(True)
        self.us_pred = us_pred
        loss = torch.mean((self.us_pred - self.us_true)**2)
        self.loss_out = loss

        return loss

class DummyLoss(Loss):
    def __init__(self):
        super().__init__()

    def forward(self, us_pred):
        us_pred.requires_grad_(True)
        self.us_pred = us_pred
        loss = torch.sum(self.us_pred ** 2)
        self.loss_out = loss

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
