import torch
from abc import ABC, abstractmethod
from codetiming import Timer
from matplotlib import pyplot as plt
import numpy as np

class FVMCells:
    state: torch.Tensor  # shape = (n_cells, N_component)
    def __init__(self, n_cells, n_component, init_val=None, device="cpu"):
        self.device = device
        if init_val is None:
            self.state = torch.zeros(n_cells, n_component, device=device)
        else:
            assert init_val.shape == (n_cells, n_component), f'Incorrect us init shape {init_val.shape = }'
            self.state = init_val.clone().to(device)

    def update_cells(self, state_new):
        """ Update cell values """
        self.state =  state_new

    def get_values(self):
        return self.convert_state_to_value(self.state)

    def convert_state_to_value(self, state):
        # TODO: TEMPORARY
        # momentum_x, momentum_y, density = state[:, 0], state[:, 1], state[:, 2]
        # u_x, u_y = momentum_x / density, momentum_y / density
        # primatives = torch.stack([u_x, u_y, density], dim=1)
        return state, state[:, :2]


class TSolver(ABC):
    """
    Time-stepping solver for PDEs. This class is abstract and should be subclassed
    to implement specific time-stepping schemes.
    """
    cells: FVMCells

    def __init__(self, cells: FVMCells, dt: float, n_steps: int, eq=None):
        """
        Initialize the time-stepping solver.

        Args:
            dt: The time step size.
        """
        from time_fvm import FVMEquation
        self.dt = dt
        self.n_steps = n_steps
        self.cells = cells
        self.eq: FVMEquation = eq

    def solve(self):
        E_props = self.eq.E_props

        plot_i = int(0.999 / self.dt)
        Eks, Eps, ts, TVs = [], [], [], []
        for i in range(self.n_steps):
            print()
            t = i * self.dt

            with Timer(text=f"{i=} Time: {{:.4g}}"):
                new_Us = self._step(i)

            primatives = self.cells.get_values()[0]


            # Track total variation
            grads = E_props.cell_grads  # shape = (n_cells, 2, 3)
            TV = grads.norm(dim=1).sum()
            TVs.append(TV.cpu())

            # Track total energy
            A = self.eq.mesh.areas.cuda()
            Ek = (primatives[:, 0] ** 2 + primatives[:, 1] ** 2) * A
            Ep = ((primatives[:, 2] - 0) ** 2) * A
            Eks.append(Ek.sum().cpu()), Eps.append(Ep.sum().cpu()), ts.append(t)

            if i % plot_i == 0:
                #self.eq.plot_cells(dEdt_pred[-1], title=f"Energy t={i * self.dt :.4g}", convert=False)
                primatives = self.cells.get_values()[0]
                dp = E_props.U_face[:, 0, 2] - E_props.U_face[:, 1, 2]
                dv = E_props.U_face[:, 0, 0] - E_props.U_face[:, 1, 0]

                self.eq.plot_flux(torch.stack([dv, dp], dim=1), title=f"Value at t={i * self.dt :.2g}", show_index=False)
                # self.eq.plot_flux(E_props.U_face[:, :, 0], title=f"Vx t={i * self.dt :.4g}", show_index=False)
                self.eq.plot_cells(primatives[:, [0, 2]], convert=False, title=f"Values at t={i * self.dt :.4g}")

            if t >= 4.5:
                Eks, Eps = torch.tensor(Eks), torch.tensor(Eps)
                E = Eks + Eps
                plt.plot(ts, Eks, label="Kinetic Energy")
                plt.plot(ts, Eps, label="Potential Energy")
                plt.plot(ts, E, label="Total Energy")
                print(f'{E.max() = }')
                plt.legend()
                plt.show()

                TVs = torch.stack(TVs, dim=0)
                plt.plot(ts, TVs, label="Total Variation")
                plt.legend()
                plt.show()

                exit(9)



            self.cells.update_cells(new_Us)

            # if t > -1:
            #     break

    @abstractmethod
    def _step(self, i: int):
        """
        Perform a single time step of the solver.

        Args:
            i: The index of the current time step.
        """
        pass


class Euler(TSolver):
    def __init__(self, cells: FVMCells, dt: float, n_steps: int, equation):
        super().__init__(cells, dt, n_steps, eq=equation)
        self.eq = equation

    def _step(self, i: int):
        dUdt = self.eq.forward(*self.cells.get_values(), i=i)

        U_i_1 = self.cells.state + self.dt * dUdt
        return U_i_1

class ExplMidpoint(TSolver):
    def __init__(self, cells: FVMCells, dt: float, n_steps: int, equation):
        super().__init__(cells, dt, n_steps, eq=equation)
        self.eq = equation

    def _step(self, i: int):
        state = self.cells.state
        primatives, _ = self.cells.get_values()

        dUdt_star = self.eq.forward(primatives, None, i=i)

        U_star = state + 0.5 * self.dt * dUdt_star        # U_{i+0.5}
        primatives_star, _ = self.cells.convert_state_to_value(U_star)

        dUdt = self.eq.forward(primatives_star, None, i=i)
        U_i_1 = state + self.dt * dUdt
        return U_i_1