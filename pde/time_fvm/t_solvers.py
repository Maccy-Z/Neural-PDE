from __future__ import annotations
from cprint import c_print
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
            self.state = init_val.to(device)

    def update_cells(self, state_new):
        """ Update cell values """
        #assert not torch.any(torch.isnan(state_new)), "Error in state_new"
        self.state =  state_new

    def get_values(self):
        return self.convert_state_to_value(self.state)

    # @torch.compile()
    def convert_state_to_value(self, state):
        # momentum_x, momentum_y, density = state[:, 0], state[:, 1], state[:, 2]
        # v_x, v_y = momentum_x / density, momentum_y / density
        #
        # primatives = torch.stack([v_x, v_y, density], dim=1)

        momentum, density = state[:, :2], state[:,2]
        density = density.unsqueeze(-1)
        V = momentum / density
        primatives = torch.cat([V, density], dim=-1)
        return primatives, state

    def save(self, name="state.pt"):
        torch.save(self.state, name)

    def load(self, name="state.pt"):
        self.state = torch.load(name, weights_only=True)


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

    def _solve(self):
        E_props = self.eq.E_props

        plot_i = int(2 / self.dt)
        # Eks, Eps, ts, TVs = [], [], [], []
        dts = []

        t = 0
        for i in range(self.n_steps):
            t += self.dt
            with Timer(text=f"{i=}, {t=:.5g} Time: {{:.4g}}"):
                new_Us= self._step(t)
                self.cells.update_cells(new_Us)

            dts.append(self.dt)

            print(f'{self.dt = :.3g}')
            # # Track total energy
            # primatives = self.cells.get_values()[0]
            # A = self.eq.mesh.areas.cuda()
            # Ek = (primatives[:, 0] ** 2 + primatives[:, 1] ** 2) * A
            # Ep = torch.log(primatives[:, 2]/0.1) * A
            # Eks.append(Ek.sum().cpu()), Eps.append(Ep.sum().cpu()), ts.append(t)

            # if t == 35:
            #     with open("save_state.pt", "wb") as f:
            #         torch.save(self.cells.state, f)
            #     exit(7)

            if i % plot_i == 0 and t>15.:
                c_print(f'{t = :.5g}', color="bright_yellow")

                primatives = self.cells.get_values()[0]
                Xlims = None # [[0.45, 0.52], [0.77, 0.84]]

                # self.eq.plot_interp(E_props.cell_grads[:, 1, 0], title=f"Grad t={i * self.dt :.4g}", Xlims=Xlims)
                # self.eq.plot_cells(div_KT[:, 0], title=f"Values at t={i * self.dt :.4g}", show_index=True, Xlims=Xlims)
                # self.eq.plot_flux(self.eq.div_V_face[:, 0], title=f"Vx t={i * self.dt :.4g}", show_index=False)
                self.eq.plot_interp(primatives[:], title=f"Values at t={t :.4g}", Xlims=Xlims, )
                # self.eq.plot_cells(primatives[:, 0], title=f"Values at t={i * self.dt :.4g}", show_index=True, Xlims=Xlims)

                if torch.any(torch.isnan(primatives)):
                    print("Nan in primatives")
                    exit(9)
                # if t>0.2:
                #     exit("DONE PLOTTING")

        dts = torch.stack(dts).cpu()
        kernel_size = 10
        kernel = torch.ones(1, 1, kernel_size) / kernel_size
        dts_smooth = torch.nn.functional.conv1d(dts.unsqueeze(0).unsqueeze(0), kernel, padding=kernel_size // 2)[0][0]
        print(f'{dts[200:].mean() = }')
        plt.plot(dts)
        plt.plot(dts_smooth)
        plt.show()
        # Eks, Eps = torch.tensor(Eks), torch.tensor(Eps)
        # E = Eks + Eps
        # plt.plot(ts, Eks, label="Kinetic Energy")
        # plt.plot(ts, Eps, label="Potential Energy")
        # plt.plot(ts, E, label="Total Energy")
        # plt.legend()
        # plt.show()



    @torch.inference_mode()
    def solve(self):
        run = True
        if run:
            self._solve()
        else:
            self._solve_profile()


    def _solve_profile(self):
        for _ in range(5):
            new_Us = self._step(0)
            self.cells.update_cells(new_Us)

        # import gc
        # gc.collect()
        # torch.cuda.empty_cache()
        # tot_el = 0
        # for name, value in vars(self.eq.t_solver).items():
        #
        #     if torch.is_tensor(value) and value.is_cuda:
        #         if value.is_sparse or value.is_sparse_csr:
        #             numel = value._nnz()
        #         else:
        #             numel = value.numel()
        #         print(f"Name: {name}, Size: {value.size()}, numel = {numel}")
        #         tot_el += numel
        #
        # c_print(f'{tot_el = }', color="magenta")

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
                # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
                record_shapes=True,  # Records tensor shapes for each op
                with_stack=True,
        ) as prof:

            for i in range(10):
                t = i * self.dt
                prof.step()
                new_Us = self._step(t)
                self.cells.update_cells(new_Us)

        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
        prof.export_chrome_trace("trace.json")


    @abstractmethod
    def _step(self, t):
        """
        Perform a single time step of the solver.

        Args:
            i: The index of the current time step.
        """
        pass

    def _euler_step(self, U, t):
        prim_a, _ = self.cells.convert_state_to_value(U)
        U_i_1 = U + self.dt * self.eq.forward(prim_a, t)
        return U_i_1

    def _forward_state(self, U, t):
        prim, _ = self.cells.convert_state_to_value(U)
        return self.eq.forward(prim, t)