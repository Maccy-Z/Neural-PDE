from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from codetiming import Timer
from matplotlib import pyplot as plt
import torch.profiler

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from time_fvm import FVMEquation

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
        #assert not torch.any(torch.isnan(state_new)), "Error in state_new"
        self.state =  state_new

    def get_values(self):
        return self.convert_state_to_value(self.state)

    def convert_state_to_value(self, state):
        # TODO: TEMPORARY
        momentum_x, momentum_y, density = state[:, 0], state[:, 1], state[:, 2]

        density = torch.clamp(density, 0.05, 1e6)
        u_x, u_y = momentum_x / density, momentum_y / density
        primatives = torch.stack([u_x, u_y, density], dim=1)

        return primatives, state[:, :2]


class TSolver(ABC):
    """
    Time-stepping solver for PDEs. This class is abstract and should be subclassed
    to implement specific time-stepping schemes.
    """
    cells: FVMCells
    eq: FVMEquation

    def __init__(self, cells: FVMCells, dt: float, n_steps: int, eq=None):
        """
        Initialize the time-stepping solver.

        Args:
            dt: The time step size.
        """
        # from time_fvm import FVMEquation
        self.dt = dt
        self.n_steps = n_steps
        self.cells = cells
        self.eq: FVMEquation = eq

    def solve(self):
        E_props = self.eq.E_props
        run = True
        if run:
            plot_i = int(0.5 / self.dt)
            Eks, Eps, ts, TVs = [], [], [], []

            for i in range(self.n_steps):
                t = i * self.dt
                with Timer(text=f"{i=} Time: {{:.4g}}"):
                    new_Us = self._step()

                primatives = self.cells.get_values()[0]



                # Track total energy
                A = self.eq.mesh.areas.cuda()
                Ek = (primatives[:, 0] ** 2 + primatives[:, 1] ** 2) * A
                Ep = torch.log(primatives[:, 2]) * A
                Eks.append(Ek.sum().cpu()), Eps.append(Ep.sum().cpu()), ts.append(t)

                if i % plot_i == 0 and i != 0:
                    # exit(f"DONE PLOTTING {t = }")
                    primatives = self.cells.get_values()[0]

                    self.eq.plot_cells(primatives[:], convert=False, title=f"Values at t={i * self.dt :.4g}", show_index=False)
                    # self.eq.plot_flux(E_props.U_face[:, 0, 0], title=f"Vx t={i * self.dt :.4g}", show_index=False)
                    #self.eq.plot_flux(dp, title=f"Vx t={i * self.dt :.4g}", show_index=True)

                    # exit("DONE PLOTTING")

                self.cells.update_cells(new_Us)

            Eks, Eps = torch.tensor(Eks), torch.tensor(Eps)
            E = Eks + Eps
            plt.plot(ts, Eks, label="Kinetic Energy")
            plt.plot(ts, Eps, label="Potential Energy")
            plt.plot(ts, E, label="Total Energy")
            print(f'{E.max() = }')
            plt.legend()
            plt.show()
            #
            # TVs = torch.stack(TVs, dim=0)
            # plt.plot(ts, TVs, label="Total Variation")
            # plt.legend()
            # plt.show()
            exit(9)

        else:

            for _ in range(5):
                new_Us = self._step()
                self.cells.update_cells(new_Us)

            with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    # schedule=torch.profiler.schedule(
                    #     warmup=1,  # Skip the first iteration (warm-up)
                    #     wait=1,  # Skip the first iteration (warm-up)
                    #     active=3  # Capture the next 3 iterations
                    # ),
                    #on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
                    record_shapes=True,  # Records tensor shapes for each op
                    with_stack=True,
            ) as prof:

                for i in range(10):
                    prof.step()
                    new_Us = self._step()
                    self.cells.update_cells(new_Us)

            print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
            prof.export_chrome_trace("trace.json")

    @abstractmethod
    def _step(self):
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

    def _step(self):
        dUdt = self.eq.forward(*self.cells.get_values())

        U_i_1 = self.cells.state + self.dt * dUdt
        return U_i_1

class ExplMidpoint(TSolver):
    def __init__(self, cells: FVMCells, dt: float, n_steps: int, equation):
        super().__init__(cells, dt, n_steps, eq=equation)
        self.eq: FVMEquation = equation

    def _step(self):
        state = self.cells.state
        primatives, _ = self.cells.get_values()

        dUdt_star = self.eq.forward(primatives, None)

        U_star = state + 0.5 * self.dt * dUdt_star        # U_{i+0.5}
        primatives_star, _ = self.cells.convert_state_to_value(U_star)

        dUdt = self.eq.forward(primatives_star, None)
        U_i_1 = state + self.dt * dUdt

        return U_i_1