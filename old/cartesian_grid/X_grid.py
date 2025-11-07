import torch
from torch import Tensor
import abc
from cprint import c_print


class XGrid(abc.ABC):
    """ Grid of X points to be shared between classes """
    N_dim: int
    Xs: Tensor  # X points, xs.shape = [[N]* N_dim]
    dx: float  # Spacing between points
    N: Tensor  # Number of real points in grid

    def __init__(self, device='cpu'):
        """ Xmin: minimum of grid
            Xmax: maximum of grid
            N: number of points in grid
            device: device to store grid on
        """
        self.device = device

    def __repr__(self):
        return f"{self.N_dim}D X points (including extra boundary points): " + str(self.Xs)


class XGrid1D(XGrid):
    N_dim = 1

    def __init__(self, Xmin, Xmax, dx: float, device='cpu'):
        super().__init__(device)
        self.dx = dx
        self.L = Xmax - Xmin
        N = self.L / dx
        assert torch.all(torch.eq(N, torch.round(N))), f'{N = } is not an integer'

        self.Xs = torch.linspace(Xmin - self.dx, Xmax + self.dx, N + 2).to(self.device)  # Shape: [N + 2], [Xmin-dx, Xmin, ...,Xmin, Xmin+dx]


class XGrid2D(XGrid):
    N_dim = 2

    def __init__(self, Xmin: float, Xmax: float, N: Tensor, device='cpu'):
        """ Xmin and Xmax are for first dimension. Second dimension is calculated from N.  """
        super().__init__(device)
        self.N = N

        # Establish range of y values from N and dx
        Xmin, Xmax = torch.tensor(Xmin), torch.tensor(Xmax)
        Lx = Xmax - Xmin
        self.dx = Lx / (N[0] - 1)
        Ly = self.dx * (N[1] - 1)
        self.L = torch.tensor([Lx, Ly])

        self.Xmin = torch.stack([Xmin, Xmin]).to(self.device)
        self.Xmax = torch.stack([Xmax, Xmin + Ly]).to(self.device)

        x_values = torch.linspace(self.Xmin[0] - self.dx, self.Xmax[0] + self.dx, self.N[0] + 2, device=self.device)
        y_values = torch.linspace(self.Xmin[1] - self.dx, self.Xmax[1] + self.dx, self.N[1] + 2, device=self.device)

        # Create the grid of points
        n_grid, m_grid = torch.meshgrid(y_values, x_values, indexing='xy')

        # Combine the grids to form the final N x M grid of points
        self.Xs = torch.stack([m_grid, n_grid], dim=-1).to(self.device)     # shape = [N0, N1, 2]

        c_print(f'Grid Range: {self.Xmin.tolist()} to {self.Xmax.tolist()}.', 'green')
        c_print(f'Grid Spacing: {self.dx:.5g}, grid Shape: {self.N.tolist()}.', 'green')



def main():
    Xs = XGrid2D(0, 1, torch.tensor([26, 51]))

    print(Xs.Xs)


if __name__ == "__main__":
    main()
