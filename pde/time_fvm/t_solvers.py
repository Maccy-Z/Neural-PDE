from __future__ import annotations
from cprint import c_print
import torch
from abc import ABC, abstractmethod
from codetiming import Timer
from matplotlib import pyplot as plt
import torch.profiler
from collections import deque

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

    def convert_state_to_value(self, state):
        # TODO: TEMPORARY
        momentum_x, momentum_y, density = state[:, 0], state[:, 1], state[:, 2]

        density = torch.clamp(density, 0.01, 1e6)
        u_x, u_y = momentum_x / density, momentum_y / density
        #u_x, u_y = momentum_x , momentum_y

        primatives = torch.stack([u_x, u_y, density], dim=1)

        #primatives[:, 0] = 1.5

        return primatives, state # state[:, :2]

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

        plot_i =  int(10 / self.dt)
        Eks, Eps, ts, TVs = [], [], [], []

        for i in range(self.n_steps):
            t = i * self.dt
            with Timer(text=f"{i=}, {t=:.5g} Time: {{:.4g}}"):
                new_Us = self._step(t)
            #new_Us[:, 1] = new_Us[:, 1] *0.999

            # # Track total energy
            primatives = self.cells.get_values()[0]
            A = self.eq.mesh.areas.cuda()
            Ek = (primatives[:, 0] ** 2 + primatives[:, 1] ** 2) * A
            Ep = torch.log(primatives[:, 2]/0.1) * A
            Eks.append(Ek.sum().cpu()), Eps.append(Ep.sum().cpu()), ts.append(t)

            # if t == 12:
            #     with open("save_state.pt", "wb") as f:
            #         torch.save(self.cells.state, f)
            #     exit(7)

            if i % plot_i == 0 and t>1.2:
                c_print(f'{t = :.5g}', color="bright_yellow")
                # print(f'{E_props.phi_lim.mean() = }')

                primatives = self.cells.get_values()[0]
                Xlims = None # [[0.45, 0.52], [0.77, 0.84]]


                # print(f'Div:    {self.eq.div_all[7255].cpu()}, {self.eq.div_all[7147].cpu()} ')
                #print(f'Visc:   {self.eq.div_visc[7255].cpu()}, {self.eq.div_visc[7147].cpu()}')
                # print(f'Advect: {self.eq.div_advect[7255].cpu()}, {self.eq.div_advect[7147].cpu()}')
                # print(f'P:      {self.eq.div_P[7255].cpu()}, {self.eq.div_P[7147].cpu()}')
                # print(f'Grads:  {E_props.cell_grads[7255, :, 0].cpu()}, {E_props.cell_grads[7147, :, 0].cpu()}')

                # adv_flux = self.eq.adv_flux.view(-1, 3)[:, :2]
                # print(f'{adv_flux[[11010, 9993]] = }')
                #self.eq.plot_flux(adv_flux, title=f"Vx t={i * self.dt :.4g}", Xlims=Xlims, show_index=True )
                # print(f'{primatives[[7082, 7256, 7255]]}')
                # [7225, 7147]
                """ KT FLUX """
                # div_KT = self.eq.div_KT.view(-1, 3)
                # flux_KT = self.eq.KT_flux.view(-1, 3)
                # print(f'{div_KT[14580] = }')
                # print(f'{E_props.cell_grads[14580, :, 0] = }')
                # print(f'{E_props.phi_lim[14580, :, 0] = }')

                #print(f'{E_props.mesh.tri_to_edge[14580,] = }')     # [22358, 22368, 22359])
                # print(f'{flux_KT.abs().mean() = }')
                # print(f'{flux_KT[[22358, 22368, 22359]] = }')


                # self.eq.plot_interp(E_props.cell_grads[:, 1, 0], title=f"Grad t={i * self.dt :.4g}", Xlims=Xlims)
                # self.eq.plot_interp(div_KT[:, 0], title=f"KT t={i * self.dt :.4g}", Xlims=Xlims)
                # self.eq.plot_cells(div_KT[:, 0], title=f"Values at t={i * self.dt :.4g}", show_index=True, Xlims=Xlims)
                # self.eq.plot_flux(flux_KT[:, 0], title=f"Vx t={i * self.dt :.4g}", Xlims=Xlims, show_index=True)

                # self.eq.plot_interp(E_props.phi_lim.mean(dim=1), title=f"Values at t={i * self.dt :.4g}", Xlims=Xlims)

                self.eq.plot_interp(primatives[:], title=f"Values at t={i * self.dt :.4g}", Xlims=Xlims)
                # self.eq.plot_cells(primatives[:, 0], title=f"Values at t={i * self.dt :.4g}", show_index=True, Xlims=Xlims)


                # exit(34)
                if torch.any(torch.isnan(primatives)):
                    #print(f'{primatives = }')
                    exit(9)
                # exit("DONE PLOTTING")

            self.cells.update_cells(new_Us)

        Eks, Eps = torch.tensor(Eks), torch.tensor(Eps)
        E = Eks + Eps
        plt.plot(ts, Eks, label="Kinetic Energy")
        plt.plot(ts, Eps, label="Potential Energy")
        plt.plot(ts, E, label="Total Energy")
        plt.legend()
        plt.show()
        #
        # TVs = torch.stack(TVs, dim=0)
        # plt.plot(ts, TVs, label="Total Variation")
        # plt.legend()
        # plt.show()
        exit(9)

    @torch.inference_mode()
    def solve(self):
        run = True
        if run:
            self._solve()
        else:
            self._solve_profile()

    def _solve_profile(self):
        for _ in range(2):
            new_Us = self._step()
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
                prof.step()
                new_Us = self._step()
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
        U_i_1 = U + self.dt * self.eq.forward(prim_a, None, t+self.dt)
        return U_i_1

class Euler(TSolver):
    def __init__(self, cells: FVMCells, dt: float, n_steps: int, equation):
        super().__init__(cells, dt, n_steps, eq=equation)
        self.eq = equation

    def _step(self, t):
        """U^{i+1} = U^i + dt * f(U^i)"""

        dUdt = self.eq.forward(*self.cells.get_values(), t=t)

        # c_print(f'{self.dt * dUdt[[66, 122, 60], 2] = }', color='green')
        U_i_1 = self.cells.state + self.dt * dUdt
        # c_print(f'{self.cells.state[[66, 122, 60], 2] = }', color='green')

        return U_i_1


class ExplMidpoint(TSolver):
    def __init__(self, cells: FVMCells, dt: float, n_steps: int, equation):
        super().__init__(cells, dt, n_steps, eq=equation)
        self.eq: FVMEquation = equation

    def _step(self, t):
        """ U^{i+0.5} = U^i + dt/2 * f(U^i)
            U^{i+1} = U^i + dt * f(U^{i+0.5})
        """

        state = self.cells.state
        primatives, _ = self.cells.get_values()

        dUdt_star = self.eq.forward(primatives, None, t=t)
        U_star = state + 0.5 * self.dt * dUdt_star        # U_{i+0.5}

        primatives_star, _ = self.cells.convert_state_to_value(U_star)
        dUdt = self.eq.forward(primatives_star, None, t=t)
        U_i_1 = state + self.dt * dUdt

        return U_i_1


class Heuns(TSolver):
    def __init__(self, cells: FVMCells, dt: float, n_steps: int, equation):
        super().__init__(cells, dt, n_steps, eq=equation)
        self.eq: FVMEquation = equation

    def _step(self, t):
        """ U_s = U_i + dt * f(U_i)
            U_{i+1} = U_i + dt * [0.5 * f(U_s}) + 0.5 * f(U_i)]
        """
        # y_star = y_n + dt*f(t_n, y_n)
        dUdt_star = self.eq.forward(*self.cells.get_values(), t=t)
        U_star = self.cells.state + self.dt * dUdt_star

        # y_{n+1} = y_n + 0.5*dt*[f(t_n, y_n) + f(t_n+1, y_star)]
        primatives_star, _ = self.cells.convert_state_to_value(U_star)
        dUdt = self.eq.forward(primatives_star, None, t=t)

        U_i_1 = self.cells.state + 0.5 * self.dt * (dUdt_star + dUdt)
        return U_i_1


class RK3_SSP(TSolver):
    def __init__(self, cells: FVMCells, dt: float, n_steps: int, equation):
        super().__init__(cells, dt, n_steps, eq=equation)
        self.eq: FVMEquation = equation

    def _step(self, t):
        """ U_a = U_i + dt * f(U_i, t)
            U_b = 3/4 * U_i + 1/4 [U_a + dt * f(U_a)]
            U_{i+1} = 1/3 * U_i + 2/3 [U_b + dt * f(U_b)]
        """

        prim_i, U_i = self.cells.get_values()
        # U_a = U_i + dt * f(U_i)
        U_a = U_i + self.dt * self.eq.forward(prim_i, None, t)

        # U_b = 3/4 * U_i + 1/4 * dt * f(U_a)
        prim_a, U_a = self.cells.convert_state_to_value(U_a)
        U_b = 3/4 * U_i + 1/4 * (U_a + self.dt * self.eq.forward(prim_a, None, t+self.dt))

        # U_{i+1} = 1/3 * U_i + 2/3 [U_b + dt * f(U_b)]
        prim_b, U_b = self.cells.convert_state_to_value(U_b)
        U_i_1 = 1/3 * U_i + 2/3 * (U_b + self.dt * self.eq.forward(prim_b, None, t+self.dt/2))
        return U_i_1


class RK2_SSP(TSolver):
    def __init__(self, cells: FVMCells, dt: float, n_steps: int, equation):
        super().__init__(cells, dt, n_steps, eq=equation)
        self.eq: FVMEquation = equation

    def _step(self, t):
        """ U_a = U_i + dt * f(U_i, t)
            U_{i+1} = 0.5 * U_i + 0.5 [U_a + dt * f(U_a)]
        """
        prim_i, U_i = self.cells.get_values()
        # U_a = U_i + dt * f(U_i)
        U_a = U_i + self.dt * self.eq.forward(prim_i, None, t)

        # U_{i+1} = 0.5 * U_i + 0.5 * [U_a + dt * f(U_a)]
        prim_a, U_a = self.cells.convert_state_to_value(U_a)
        U_i_1 = 0.5 * U_i + 0.5 * (U_a + self.dt * self.eq.forward(prim_a, None, t+self.dt))
        return U_i_1

class RK2_SSP3(TSolver):
    def __init__(self, cells: FVMCells, dt: float, n_steps: int, equation):
        super().__init__(cells, dt, n_steps, eq=equation)
        self.eq: FVMEquation = equation

    def _step(self, t):
        """ U_a = 1/2 * U_i + 1/2 * [U_i + dt * f(U_i)]
            U_b = 1/2 * U_a + 1/2 * [U_a + dt * f(U_a)]
            U_{i+1} = 1/3 * U_i + 1/3 * U_b + 1/3 * [U_b + dt * f(U_b)]
        """
        _, U_i = self.cells.get_values()
        # U_a = 1/2 * U_i + 1/2 * [U_i + dt * f(U_i)]
        U_a = 1/2 * U_i + 1/2 * self._euler_step(U_i, t=t)

        # U_b = 1/2 * U_a + 1/2 * [U_a + dt * f(U_a)]
        U_b = 1/2 * U_a + 1/2 * self._euler_step(U_a, t=t+self.dt/2)

        # U_{i+1} = 1/3 * U_i + 1/3 * U_b + 1/3 * [U_b + dt * f(U_b)]
        U_i_1 = 1/3 * U_i + 1/3 * U_b + 1/3 * self._euler_step(U_b, t=t+self.dt)

        return U_i_1


class RK2_SSP4(TSolver):
    def __init__(self, cells: FVMCells, dt: float, n_steps: int, equation):
        super().__init__(cells, dt, n_steps, eq=equation)
        self.eq: FVMEquation = equation

    def _step(self, t):
        """ U_a = 2/3 * U_i + 1/3 * [U_i + dt * f(U_i)]
            U_b = 2/3 * U_a + 1/3 * [U_a + dt * f(U_a)]
            U_c = 2/3 * U_b + 1/3 * [U_b + dt * f(U_b)]
            U_{i+1} = 1/4 * U_i + 1/2 * U_c + 1/4 * [U_c + dt * f(U_c)]
        """
        _, U_i = self.cells.get_values()
        # U_a = 2/3 * U_i + 1/3 * [U_i + dt * f(U_i)]
        U_a = 2/3 * U_i + 1/3 * self._euler_step(U_i, t=t)

        # U_b = 2/3 * U_a + 1/3 * [U_a + dt * f(U_a)]
        U_b = 2/3 * U_a + 1/3 * self._euler_step(U_a, t=t+self.dt/3)

        # U_c = 2/3 * U_b + 1/3 * [U_b + dt * f(U_b)]
        U_c = 2 / 3 * U_b + 1/3 * self._euler_step(U_b, t=t + self.dt*2/3)

        # U_{i+1} = 1/4 * U_i + 1/2 * U_c + 1/4 * [U_c + dt * f(U_c)]
        U_i_1 = 1/4 * U_i + 1/2 * U_c + 1/4 * self._euler_step(U_c, t=t+self.dt)

        return U_i_1


class RK3_SSP4(TSolver):
    def __init__(self, cells: FVMCells, dt: float, n_steps: int, equation):
        super().__init__(cells, dt, n_steps, eq=equation)
        self.eq: FVMEquation = equation

    def _step(self, t):
        """ U_a = 1/2 * U_i + 1/2 * [U_i + dt * f(U_i)]
            U_b = 1/2 * U_a + 1/2 * [U_a + dt * f(U_a)]
            U_c = 2/3 * U_i + 1/6 * U_b + 1/6 * [U_b + dt * f(U_b)]
            U_{i+1} = 1/2 * U_c + 1/2 [U_c + dt * f(U_c)]
        """

        _, U_i = self.cells.get_values()
        # U_a = 1/2 * U_i + 1/2 * [U_i + dt * f(U_i)]
        U_a = 1/2 * U_i + 1/2 * self._euler_step(U_i, t=t)

        # U_b = 1/2 * U_a + 1/2 * [U_a + dt * f(U_a)]
        U_b = 1/2 * U_a + 1/2 * self._euler_step(U_a, t=t+self.dt/2) #(U_a + self.dt * self.eq.forward(prim_a, None, t+self.dt/2))

        # U_c = 2/3 * U_i + 1/6 * U_b + 1/6 * [U_b + dt * f(U_b)]
        U_c = 2/3 * U_i + 1/6 * U_b + 1/6 * self._euler_step(U_b, t=t+self.dt) #(U_b + self.dt * self.eq.forward(prim_b, None, t+self.dt))

        # U_{i+1} = 1/2 * U_c + 1/2 [U_c + dt * f(U_c)]
        U_i_1 = 1/2 * U_c + 1/2 * self._euler_step(U_c, t=t+self.dt/2) #(U_c + self.dt * self.eq.forward(prim_c, None, t+self.dt/2))

        return U_i_1


class Adams2(TSolver):
    """ Adams Bashforth 2 solver
        Non-Markov solver.
    """
    def __init__(self, cells: FVMCells, dt: float, n_steps: int, equation):
        super().__init__(cells, dt, n_steps, eq=equation)
        self.eq: FVMEquation = equation

        self.prev_dUdt = deque(maxlen=2)

    def _init_states(self, t):
        prim, _ = self.cells.get_values()
        dUdt_0 = self.eq.forward(prim, None, t)

        for _ in range(2):
            self.prev_dUdt.append(dUdt_0)

    def _step(self, t):
        """
        U_{t+1} = U_t + dt/2 * [3 * f(U_t) - f(U_{t-1})]
        :param t:
        :return:
        """
        if len(self.prev_dUdt) == 0:
            self._init_states(t)

        prim_t, U_t = self.cells.get_values()

        dUdt_t = self.eq.forward(prim_t, None, t)
        dUdt_tm1 = self.prev_dUdt[-1]
        # dUdt_tm2 = self.prev_dUdt[-2]

        # U_t_1 = U_t + self.dt/12 * (23 * dUdt_t - 16 * dUdt_tm1 + 5 * dUdt_tm2)
        U_t_1 = U_t + self.dt/2 * (3 * dUdt_t - dUdt_tm1)

        self.prev_dUdt.append(dUdt_t)

        return U_t_1

class Adams3PC(TSolver):
    """ Adams–Bashforth–Moulton predictor corrector 3 solver
        Non-Markov solver.
    """
    def __init__(self, cells: FVMCells, dt: float, n_steps: int, equation):
        super().__init__(cells, dt, n_steps, eq=equation)
        self.eq: FVMEquation = equation

        self.prev_dUdt = deque(maxlen=2)

    def _init_states(self, t):
        prim, _ = self.cells.get_values()
        dUdt_0 = self.eq.forward(prim, None, t)

        for _ in range(2):
            self.prev_dUdt.append(dUdt_0)

    #@torch.compile()
    def _step(self, t):
        """
        U_a = U_t + dt/2 * [3 * f(U_t) - f(U_{t-1})]
        U_{t+1} = U_t + dt/12 * [5 * f(U_a) + 8 * f(U_{t}) - 1 * f(U_{t-1})]
        :param t:
        :return:
        """
        if len(self.prev_dUdt) == 0:
            self._init_states(t)

        prim_t, U_t = self.cells.get_values()

        dUdt_t = self.eq.forward(prim_t, None, t)
        dUdt_tm1 = self.prev_dUdt[-1]

        # U_a = U_t + dt/2 * [3 * f(U_t) - f(U_{t-1})]
        U_a = U_t + self.dt/2 * (3 * dUdt_t - dUdt_tm1)

        # U_{t+1} = U_t + dt/12 * [5 * f(U_a) + 8 * f(U_{t}) - 1 * f(U_{t-1})]
        dUdt_a = self.eq.forward(*self.cells.convert_state_to_value(U_a), t)
        U_t_1 =  U_t + self.dt/12 * (5 * dUdt_a + 8 * dUdt_t - dUdt_tm1)

        self.prev_dUdt.append(dUdt_t)

        return U_t_1



