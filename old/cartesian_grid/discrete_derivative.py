import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from pde.cartesian_grid.X_grid import XGrid
import abc


class DerivativeCalc(abc.ABC):
    def __init__(self, x_grid: XGrid, order, device='cpu'):
        self.device = device
        self.dx = x_grid.dx
        if order == 2:
            self.du_kern_raw = [-0.5, 0, 0.5]
            self.d2u_ker_raw = [1, -2, 1]
        elif order == 4:
            self.du_kern_raw = [1 / 12, -2 / 3, 0, 2 / 3, -1 / 12]
            self.d2u_ker_raw = [-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12]
        else:
            raise ValueError(f"Given order {order} not supported. Must be 2, 4")

        # Boundary coefficients
        self.du_forward = [-1.5, 2, -0.5]
        self.du_backward = [0.5, -2, 1.5]
        self.d2u_forward = [2, -5, 4, -1]
        self.d2u_backward = [-1, 4, -5, 2]

    @abc.abstractmethod
    def derivative(self, u):
        pass


class DerivativeCalc1D(DerivativeCalc):
    def __init__(self, dx, order, device='cpu'):
        super().__init__(dx, order, device=device)

        # Central kernels
        self.du_kernel = torch.tensor([self.du_kern_raw], dtype=torch.float32).reshape(1, 1, -1).to(self.device) / dx
        self.d2u_kernel = torch.tensor([self.d2u_ker_raw], dtype=torch.float32).reshape(1, 1, -1).to(self.device) / (dx ** 2)

        # Edge forward/backward kernels
        self.du_forward_kernel = torch.tensor([self.du_forward], dtype=torch.float32).reshape(1, 1, -1).to(self.device) / dx
        self.du_backward_kernel = torch.tensor([self.du_backward], dtype=torch.float32).reshape(1, 1, -1).to(self.device) / dx
        self.d2u_forward_kernel = torch.tensor([self.d2u_forward], dtype=torch.float32).reshape(1, 1, -1).to(self.device) / (dx ** 2)
        self.d2u_backward_kernel = torch.tensor([self.d2u_backward], dtype=torch.float32).reshape(1, 1, -1).to(self.device) / (dx ** 2)

    def derivative(self, u):
        """ u.shape = [N + boundary]
            Return shape: [N]
        """
        u = u.view(1, *u.shape)
        dudx = F.conv1d(u, self.du_kernel, padding=0, stride=1)[0]  # [0, self.extra_points:-self.extra_points]
        d2udx2 = F.conv1d(u, self.d2u_kernel, padding=0, stride=1)[0]

        return dudx, d2udx2

    def derivative_boundary(self, u):
        """
        Derivative, including boundary points. Approxed using forward/backward difference.
            u.shape = [N]
            Return shape: [N]
        """
        dudx, d2udx2 = torch.empty_like(u), torch.empty_like(u)
        # Main points
        dudx_cent, d2udx2_cent = self.derivative(u)
        dudx[1:-1] = dudx_cent
        d2udx2[1:-1] = d2udx2_cent

        # Boundary points
        du_low, d2u_low = self._forward_diff(u)
        du_high, d2u_high = self._backward_diff(u)
        dudx[0] = du_low
        dudx[-1] = du_high
        d2udx2[0] = d2u_low
        d2udx2[-1] = d2u_high

        return dudx, d2udx2

    def _forward_diff(self, u):
        """Forward derivative for boundary points"""

        u = u.view(1, *u.shape)
        dudx_low = F.conv1d(u[:, :3], self.du_forward_kernel, padding=0, stride=1)[0]
        d2udx2_low = F.conv1d(u[:, :4], self.d2u_forward_kernel, padding=0, stride=1)[0]

        return dudx_low, d2udx2_low

    def _backward_diff(self, u):
        """Backward derivative for boundary points"""

        u = u.view(1, *u.shape)
        dudx_high = F.conv1d(u[:, -3:], self.du_backward_kernel, padding=0, stride=1)[0]
        d2udx2_high = F.conv1d(u[:, -4:], self.d2u_backward_kernel, padding=0, stride=1)[0]

        return dudx_high, d2udx2_high


class DerivativeCalc2D(DerivativeCalc):
    def __init__(self, x_grid: XGrid, order):
        super().__init__(x_grid, order, device=x_grid.device)

        # Kernel shape: [out_channels, in_channels, kern_x, kern_y]
        self.dudx_kern = torch.tensor([self.du_kern_raw], dtype=torch.float32, device=self.device).reshape(1, 1, -1, 1) / self.dx
        self.dudy_kern = torch.tensor([self.du_kern_raw], dtype=torch.float32, device=self.device).reshape(1, 1, 1, -1) / self.dx

        self.d2udx_kern = torch.tensor([self.d2u_ker_raw], dtype=torch.float32, device=self.device).reshape(1, 1, -1, 1) / (self.dx ** 2)
        self.d2udy_kern = torch.tensor([self.d2u_ker_raw], dtype=torch.float32, device=self.device).reshape(1, 1, 1, -1) / (self.dx ** 2)

        self.extra_points = 1

    def derivative(self, u) -> dict[tuple, torch.Tensor]:
        """
            Calculate central finite difference derivatives
            u.shape = [Nx + boundary, Ny + boundary]
            Return shape: [Nx, Ny, 2]
        """

        u = u.view(1, *u.shape)  # Shape: [1, Nx, Ny]

        # Convolutions for derivatives. Conv removes padding along main axis, need to slice to remove padding along other axis
        dudx = F.conv2d(u, self.dudx_kern, padding=0, stride=1)[0, :, self.extra_points:-self.extra_points]
        dudy = F.conv2d(u, self.dudy_kern, padding=0, stride=1)[0, self.extra_points:-self.extra_points, :]

        d2udx2 = F.conv2d(u, self.d2udx_kern, padding=0, stride=1)[0, :, self.extra_points:-self.extra_points]
        d2udy2 = F.conv2d(u, self.d2udy_kern, padding=0, stride=1)[0, self.extra_points:-self.extra_points, :]

        # dudX = torch.stack([dudx, dudy], dim=-1)
        # d2udX2 = torch.stack([d2udx2, d2udy2], dim=-1)
        # return dudX, d2udX2

        deriv_dict = {(1, 0): dudx, (0, 1): dudy, (2, 0): d2udx2, (1, 1): torch.empty_like(d2udx2), (0, 2): d2udy2}
        return deriv_dict

def main2D():
    def f(Xs):
        # Xs.shape = [Nx, Ny, 2]
        x = Xs[..., 0]
        y = Xs[..., 1]
        return torch.sin(x) * torch.cos(y)

    # Parameters
    L = 2 * torch.pi  # Length of the spatial domain
    N = 5  # Number of points
    dx = L / N  # Resolution of discretisation
    order = 2  # Order of derivative
    extra_points = order // 2

    derivative_calculator = DerivativeCalc2D(dx, order=order, extra_points=extra_points)

    # Tensor of points, with additional boundary points
    x_idx = torch.arange(N + extra_points * 2)
    y_idx = torch.arange(N + extra_points * 2)
    x, y = torch.meshgrid(x_idx, y_idx, indexing='xy')
    Xs = torch.stack([x, y], dim=-1)
    Xs = (Xs - extra_points) * L / N
    us = f(Xs).to(torch.float32)

    dudX, d2udX2 = derivative_calculator.derivative(us)

    # Remove extra points for plotting
    us = us[extra_points:-extra_points, extra_points:-extra_points].numpy().squeeze()
    dudX = dudX.numpy()
    d2udX2 = d2udX2.numpy()

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(us, origin='lower')
    plt.title('Function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.imshow(d2udX2[1], origin='lower', vmin=-1, vmax=1)
    # plt.imshow(d2udx2, label="y'' = -sin(x)")
    plt.title('Discrete Derivative')
    plt.xlabel('x')
    plt.ylabel("y")
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main1D()
