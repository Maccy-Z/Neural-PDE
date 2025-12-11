from collections import deque
import torch
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from time_fvm import FVMEquation
from time_fvm.config_fvm import ConfigFVM
from t_solvers import TSolver, FVMCells


class Adaptive:
    def _adapt_init(self, order: int, rtol, atol, mtol, alphas, dt_min=None, dt_max=None):
        self.order = order
        self.rtol = rtol
        self.atol = atol
        self.mtol = mtol
        self.alphas = torch.tensor(alphas, device="cuda")

        self.dt_min = dt_min
        self.dt_max = dt_max

    def update_stepsize(self, dU_high, dU_low, Us):
        """ Update the time step size based on the difference between two solutions.
            U.shape = [n_cells, n_comp]

        """
        # Compute the difference between the two solutions
        diff = dU_high - dU_low

        # Compute a new time step size based on the difference
        # For example, you could use a simple heuristic like:
        E = torch.norm(diff, dim=0) / (self.atol + self.rtol * torch.norm(dU_high, dim=0) + self.mtol * torch.norm(Us, dim=0))
        E = E.mean()

        # If E<1, increase the time step size, otherwise decrease step size
        factor = 0.9 * (1 / E) ** (1 / self.order)

        # if E>1:
        #     alpha = self.alphas[0]
        # else:
        #     alpha = self.alphas[1]
        alpha = torch.where(E > 1, self.alphas[0], self.alphas[1])

        self.dt = self.dt * (alpha  + (1-alpha) * factor)

        # if self.dt_min is not None:
        #     self.dt = torch.clamp(self.dt, min=self.dt_min, max=self.dt_max)

        # if self.dt == self.dt_min:
        #     print()
        #     print(f'Warning: dt is at minimum value {self.dt_min}.')
        #     print(E, torch.norm(diff, dim=0), torch.norm(dU_high, dim=0))
        #     E = torch.norm(diff, dim=0) / (self.atol + self.rtol * torch.norm(dU_high, dim=0))
        #     print(f'{E = }')
        #     Xlims = None # [[3.1, 3.4], [-1.8, -1.6]]
        #     eq.plot_interp(diff, Xlims=Xlims)
            # exit(7)


class RK3_SSP4(TSolver, Adaptive):
    def __init__(self, cells: FVMCells, dt: float, n_steps: int, equation, cfg: ConfigFVM):
        super().__init__(cells, dt, n_steps, eq=equation, cfg=cfg)
        self.eq: FVMEquation = equation

        self._adapt_init(order=4, atol=1e-1, rtol=1e-1, mtol=5e-7, alphas=(0.8, 0.995), dt_min=self.dt)

    def _step(self, t):
        """ U_a = 1/2 * U_i + 1/2 * [U_i + dt * f(U_i)]
            U_b = 1/2 * U_a + 1/2 * [U_a + dt * f(U_a)]
            U_c = 2/3 * U_i + 1/6 * U_b + 1/6 * [U_b + dt * f(U_b)]
            U_{i+1} = 1/2 * U_c + 1/2 [U_c + dt * f(U_c)]
        """

        U_0 = self.cells.state
        # U_a = 1/2 * U_i + 1/2 * [U_i + dt * f(U_i)]
        U_a = 1/2 * (U_0 + self._euler_step(U_0, t=t))

        # U_b = 1/2 * U_a + 1/2 * [U_a + dt * f(U_a)]
        U_b = 1/2 * (U_a + self._euler_step(U_a, t=t+self.dt/2))

        # U_c = 2/3 * U_i + 1/6 * U_b + 1/6 * [U_b + dt * f(U_b)]
        U_c = 2/3 * U_0 + 1/6 * (U_b + self._euler_step(U_b, t=t+self.dt))

        # U_{i+1} = 1/2 * U_c + 1/2 [U_c + dt * f(U_c)]
        U_i_1 = 1/2 * (U_c + self._euler_step(U_c, t=t+self.dt/2))

        self.update_stepsize((U_i_1 - U_0), (U_b - U_0), U_0)

        return U_i_1


class Adams3PC(TSolver, Adaptive):
    """ Adams–Bashforth–Moulton predictor corrector 3 solver
        Non-Markov solver.
    """
    def __init__(self, cells: FVMCells, dt: float, n_steps: int, equation, cfg: ConfigFVM):
        super().__init__(cells, dt, n_steps, eq=equation, cfg=cfg)
        self.eq: FVMEquation = equation

        self.prev_dUdt = deque(maxlen=2)
        self._adapt_init(order=4, atol=2e-3, rtol=2e-3, mtol=1e-7, alphas=(0.9, 0.99), dt_min=self.dt/2)
        self.need_init = True

    def _init_states(self, t):
        prim, _ = self.cells.get_values()
        dUdt_0 = self.eq.forward(prim, self.dt, t)

        self.dUdt_m1 = dUdt_0
        self.dUdt_m2 = dUdt_0


    def _step(self, t):
        """
        U_a = U_t + dt/2 * [3 * f(U_t) - f(U_{t-1})]
        U_{t+1} = U_t + dt/12 * [5 * f(U_a) + 8 * f(U_{t}) - 1 * f(U_{t-1})]
        :param t:
        :return:
        """
        if self.need_init:
            self._init_states(t)
            self.need_init = False

        prim_t, U_0 = self.cells.get_values()

        dUdt_0 = self.eq.forward(prim_t, self.dt, t)
        dUdt_m1 = self.dUdt_m1
        dUdt_m2 = self.dUdt_m2

        # U_a = U_t + dt/2 * [3 * f(U_t) - f(U_{t-1})]
        # U_ac = U_0 + self.dt * dUdt_0
        # U_a = U_0 + self.dt/2 * (3 * dUdt_0 - dUdt_m1)
        # U_a = U_0 + self.dt / 12 * (23 * dUdt_0 - 16 * dUdt_m1 + 6 * dUdt_m2)
        # U_a = 1/3 * U_a  + 1/3 * U_ab  + 1/3 * U_ac
        U_a = U_0 + self.dt / 36 * (53 * dUdt_0 - 22 * dUdt_m1 + 6 * dUdt_m2)

        # U_{t+1} = U_t + dt/12 * [5 * f(U_a) + 8 * f(U_{t}) - 1 * f(U_{t-1})]
        dUdt_a = self._forward_state(U_a, t)
        dU_high = self.dt/12 * (5 * dUdt_a + 8 * dUdt_0 - dUdt_m1)
        dU_low = self.dt/2 * (dUdt_a + dUdt_0)
        U_1_high =  U_0 + dU_high


        # Update buffer
        self.dUdt_m2 = dUdt_m1
        self.dUdt_m1 = dUdt_0

        self.update_stepsize(dU_high, dU_low)

        return U_1_high


class Adams4PC(TSolver, Adaptive):
    """ Adams–Bashforth–Moulton predictor corrector 4 solver
        Non-Markov solver.
    """
    def __init__(self, cells: FVMCells, dt: float, n_steps: int, equation, cfg: ConfigFVM):
        super().__init__(cells, dt, n_steps, eq=equation, cfg=cfg)
        self.eq: FVMEquation = equation

        self.prev_dUdt = deque(maxlen=2)
        self._adapt_init(order=4, atol=1e-3, rtol=1e-3, mtol=1e-6, alphas=(0.9, 0.99), dt_min=self.dt/2)
        self.need_init = True

    def _init_states(self, t):
        prim, _ = self.cells.get_values()
        dUdt_0 = self.eq.forward(prim, self.dt, t)

        self.dUdt_m1 = dUdt_0
        self.dUdt_m2 = dUdt_0

    @torch.compile()
    def _step(self, t):
        """
        U_a = U_t + dt/24 * [55 * f(U_t) - 59 * f(U_{t-1}) + 37 * f(U_{t-2}) - 9 * f(U_{t-3})]  (Or other order predcitor)
        U_{t+1} = U_t + dt/24 * [9 * f(U_a) + 19 * f(U_{t}) - 5 * f(U_{t-1}) + f(U_{t-2})]
        """
        if self.need_init:
            self._init_states(t)
            self.need_init = False

        prim_t, U_0 = self.cells.get_values()

        dUdt_0 = self.eq.forward(prim_t, self.dt, t)
        dUdt_m1 = self.dUdt_m1
        dUdt_m2 = self.dUdt_m2

        # Predictor step
        # U_ac = U_t + self.dt  * dUdt_t
        #U_a = U_0 + self.dt / 2 * (3 * dUdt_0 - dUdt_m1)
        # U_a = U_0 + self.dt/12 * (23 * dUdt_0 - 16 * dUdt_m1 + 6 * dUdt_m2)
        # U_a = 1/3 * U_a  + 1/3 * U_ab  + 1/3 * U_ac
        U_a = U_0 + self.dt / 36 * (53 * dUdt_0 - 22 * dUdt_m1 + 6 * dUdt_m2)

        # U_{t+1} = U_t + dt/24 * [9 * f(U_a) + 19 * f(U_{t}) - 5 * f(U_{t-1}) + f(U_{t-2})]
        dUdt_a = self._forward_state(U_a, t)

        dU_high = self.dt/24 * (9 * dUdt_a + 19 * dUdt_0 - 5 * dUdt_m1 + dUdt_m2)
        dU_low = self.dt / 12 * (5 * dUdt_a + 8 * dUdt_0 - dUdt_m1)
        U_1_high =  U_0 + dU_high

        # Update buffer
        self.dUdt_m2 = dUdt_m1
        self.dUdt_m1 = dUdt_0

        self.update_stepsize(dU_high, dU_low, U_0)
        return U_1_high


class Butcher_Tables:
    def __init__(self, name, device):
        if name == "RK4":
            A = torch.tensor([
                [0.0, 0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0, 0.0],
                [0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0]
            ], dtype=torch.float32)

            b = torch.tensor([1 / 6, 1 / 3, 1 / 3, 1 / 6], dtype=torch.float32)
            c = torch.tensor([0.0, 0.5, 0.5, 1.0], dtype=torch.float32)

        elif name == "RK3_SSP4":
            A = torch.tensor([
                [0.0, 0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0, 0.0],
                [0.5, 0.5, 0.0, 0.0],
                [1 / 6, 1 / 6, 1 / 6, 0.0]
            ], dtype=torch.float32)

            b = torch.tensor([1 / 6, 1 / 6, 1 / 6, 1 / 2], dtype=torch.float32)
            c = torch.tensor([0.0, 0.5, 1, 0.5], dtype=torch.float32)
            b2 = torch.tensor([1 / 4, 1 / 4, 1 / 4, 1 / 4], dtype=torch.float32)
            self.b2 = b2.reshape(-1, 1, 1)

        elif name == "RK3_SSP5":
            A = torch.tensor([
                [0.0, 0.0, 0.0, 0.0, 0],
                [0.37726891511710, 0.0, 0.0, 0.0, 0],
                [0.37726891511710, 0.37726891511710, 0.0, 0.0, 0],
                [0.16352294089771, 0.16352294089771, 0.16352294089771, 0.0, 0],
                [0.14904059394856, 0.14831273384724, 0.14831273384724, 0.34217696850008, 0],
            ], dtype=torch.float32)

            b = torch.tensor([0.19707596384481, 0.11780316509765, 0.11709725193772, 0.27015874934251, 0.29786487010104], dtype=torch.float32)
            c = torch.tensor([0, 0.37726891511710 , 0.75453783023419 , 0.49056882269314 , 0.78784303014311 ], dtype=torch.float32)
            b2 = torch.tensor([1/5, 1/5, 1/5, 1/5, 1/5], dtype=torch.float32)

        elif name == """RK3_SSP6""":
            A = torch.tensor([
                [0.0, 0.0, 0.0, 0.0, 0, 0],
                [0.28422, 0.0, 0.0, 0.0, 0, 0],
                [0.28422, 0.28422, 0.0, 0.0, 0, 0],
                [0.23071, 0.23071, 0.23071, 0.0, 0, 0],
                [0.13416, 0.13416, 0.13416, 0.16528, 0, 0],
                [0.13416, 0.13416, 0.13416, 0.16528, 0.28422, 0]
            ], dtype=torch.float32)

            b = torch.tensor([0.17016,  0.17016,  0.10198,  0.12563,  0.21604,  0.21604], dtype=torch.float32)
            c = torch.tensor([0, 0.28422, 0.56844 , 0.69213, 0.56776, 0.85198], dtype=torch.float32)
            b2 = torch.tensor([1/6, 1/6, 1/6, 1/6, 1/6, 1/6], dtype=torch.float32)

        elif name == """RK4_SSP5""":
            A = torch.tensor([
                [0.0, 0.0, 0.0, 0.0, 0],
                [0.39175222700392, 0.0, 0.0, 0.0, 0],
                [0.21766909633821, 0.36841059262959, 0.0, 0.0, 0],
                [0.08269208670950, 0.13995850206999,  0.25189177424738, 0.0, 0],
                [0.06796628370320, 0.11503469844438, 0.20703489864929, 0.54497475021237, 0],
            ], dtype=torch.float32)

            b = torch.tensor([0.14681187618661, 0.24848290924556, 0.10425883036650, 0.27443890091960, 0.22600748319395], dtype=torch.float32)
            c = torch.tensor([0., 0.39175222700392, 0.58607968896779 , 0.47454236302687, 0.93501063100924], dtype=torch.float32)


        elif name == "RK4_SSP10":
            A = torch.tensor([
                [0]*10,
                [1/6] + [0]*9,
                [1/6]*2 + [0]*8,
                [1/6]*3 + [0]*7,
                [1/6]*4 + [0]*6,
                [1/15]*5 + [0]*5,
                [1 / 15] * 5 + [1/6] + [0]*4,
                [1 / 15] * 5 + [1/6]*2 + [0]*3,
                [1 / 15] * 5 + [1/6]*3 + [0]*2,
                [1 / 15] * 5 + [1/6]*4 + [0]*1,

            ], dtype=torch.float32)
            b = torch.tensor([1/10]*10, dtype=torch.float32)
            c = A.sum(dim=1)

            b2 = torch.tensor([1/5, 0, 0, 3/10, 0, 0, 1/5, 0, 3/10, 0], dtype=torch.float32)
        else:
            raise NotImplementedError("Unknown Butcher tableau")

        self.A, self.b, self.c = A.to(device), b.reshape(-1, 1, 1).to(device), c.to(device)
        self.b2 = b2.reshape(-1, 1, 1).to(device) if 'b2' in locals() else None


class Butcher_adapt(TSolver, Adaptive):
    def __init__(self, cells: FVMCells, dt: float, n_steps: int, equation, name, cfg: ConfigFVM):
        super().__init__(cells, dt, n_steps, eq=equation, cfg=cfg)
        """
        Initializes the solver with a Butcher tableau.
        """

        tables = Butcher_Tables(name, cells.device)

        self.A = tables.A.unsqueeze(-1).unsqueeze(-1)
        self.b = tables.b
        self.b2 = tables.b2
        self.c = tables.b
        self.stages = self.b.shape[0]

        self._adapt_init(order=4, atol=2e-3, rtol=2e-3, mtol=1e-6, alphas=(0.8, 0.995), dt_min=self.dt*0.5)
        self.k = torch.zeros((self.stages, *self.cells.state.shape), device=self.A.device)

    def _step(self, t) -> torch.Tensor:
        """
        Take one step of the ODE solver.

        Returns:
            torch.Tensor: Updated state after one step.
        """

        state_0 = self.cells.state

          # shape = [stages, n_cells, n_comp]
        for i in range(self.stages):
            if i == 0:
                increment = 0
            else:
                # Compute the increment for y using previous stages
                increment = (self.A[i, :i] * self.k[:i]).sum(dim=0)
            # Evaluate the derivative at the stage time and state
            k_i =  self.dt * self._forward_state(state_0 + increment, t + self.c[i] * self.dt)
            self.k[i] = k_i

        # Combine stages to compute next state
        dU_high = torch.sum(self.b * self.k, dim=0)
        dU_low = torch.sum(self.b2 * self.k, dim=0)
        U_next_high = state_0 + dU_high

        self.update_stepsize(dU_high, dU_low, state_0)
        return U_next_high


class Euler(TSolver):
    def __init__(self, cells: FVMCells, dt: float, n_steps: int, equation, cfg: ConfigFVM):
        super().__init__(cells, dt, n_steps, eq=equation, cfg=cfg)
        self.eq = equation

    def _step(self, t):
        """U^{i+1} = U^i + dt * f(U^i)"""

        dUdt = self.eq.forward(self.cells.get_values()[0], self.dt, t=t)
        U_i_1 = self.cells.state + self.dt * dUdt

        return U_i_1


class ExplMidpoint(TSolver):
    def __init__(self, cells: FVMCells, dt: float, n_steps: int, equation, cfg: ConfigFVM):
        super().__init__(cells, dt, n_steps, eq=equation, cfg=cfg)
        self.eq: FVMEquation = equation

    def _step(self, t):
        """ U^{i+0.5} = U^i + dt/2 * f(U^i)
            U^{i+1} = U^i + dt * f(U^{i+0.5})
        """

        state = self.cells.state
        primatives, _ = self.cells.get_values()

        dUdt_star = self.eq.forward(primatives, t=t)
        U_star = state + 0.5 * self.dt * dUdt_star        # U_{i+0.5}

        primatives_star, _ = self.cells.convert_state_to_value(U_star)
        dUdt = self.eq.forward(primatives_star, t=t)
        U_i_1 = state + self.dt * dUdt

        return U_i_1


class Heuns(TSolver):
    def __init__(self, cells: FVMCells, dt: float, n_steps: int, equation, cfg: ConfigFVM):
        super().__init__(cells, dt, n_steps, eq=equation, cfg=cfg)
        self.eq: FVMEquation = equation

    def _step(self, t):
        """ U_s = U_i + dt * f(U_i)
            U_{i+1} = U_i + dt * [0.5 * f(U_s}) + 0.5 * f(U_i)]
        """
        U_0 = self.cells.state
        # y_star = y_n + dt*f(t_n, y_n)
        dUdt_star = self._forward_state(U_0, t)
        U_star = U_0 + self.dt * dUdt_star

        # y_{n+1} = y_n + 0.5*dt*[f(t_n, y_n) + f(t_n+1, y_star)]
        dUdt = self._forward_state(U_star, t)
        U_i_1 = self.cells.state + 0.5 * self.dt * (dUdt_star + dUdt)

        # # y_{n+1} = y_n + 0.5*dt*[f(t_n, y_n) + f(t_n+1, y_star)]
        # dUdt = self._forward_state(U_i_1, t)
        # U_i_1 = self.cells.state + 0.5 * self.dt * (dUdt_star + dUdt)

        return U_i_1


class RK3_SSP(TSolver):
    def __init__(self, cells: FVMCells, dt: float, n_steps: int, equation, cfg: ConfigFVM):
        super().__init__(cells, dt, n_steps, eq=equation, cfg=cfg)
        self.eq: FVMEquation = equation

    def _step(self, t):
        """ U_a = U_i + dt * f(U_i, t)
            U_b = 3/4 * U_i + 1/4 [U_a + dt * f(U_a)]
            U_{i+1} = 1/3 * U_i + 2/3 [U_b + dt * f(U_b)]
        """

        prim_i, U_i = self.cells.get_values()
        # U_a = U_i + dt * f(U_i)
        U_a = U_i + self.dt * self.eq.forward(prim_i, t)

        # U_b = 3/4 * U_i + 1/4 * dt * f(U_a)
        prim_a, U_a = self.cells.convert_state_to_value(U_a)
        U_b = 3/4 * U_i + 1/4 * (U_a + self.dt * self.eq.forward(prim_a, t+self.dt))

        # U_{i+1} = 1/3 * U_i + 2/3 [U_b + dt * f(U_b)]
        prim_b, U_b = self.cells.convert_state_to_value(U_b)
        U_i_1 = 1/3 * U_i + 2/3 * (U_b + self.dt * self.eq.forward(prim_b, t+self.dt/2))
        return U_i_1


class RK2_SSP(TSolver):
    def __init__(self, cells: FVMCells, dt: float, n_steps: int, equation, cfg: ConfigFVM):
        super().__init__(cells, dt, n_steps, eq=equation, cfg=cfg)
        self.eq: FVMEquation = equation

    def _step(self, t):
        """ U_a = U_i + dt * f(U_i, t)
            U_{i+1} = 0.5 * U_i + 0.5 [U_a + dt * f(U_a)]
        """
        prim_i, U_i = self.cells.get_values()
        # U_a = U_i + dt * f(U_i)
        U_a = U_i + self.dt * self.eq.forward(prim_i, t)

        # U_{i+1} = 0.5 * U_i + 0.5 * [U_a + dt * f(U_a)]
        prim_a, U_a = self.cells.convert_state_to_value(U_a)
        U_i_1 = 0.5 * U_i + 0.5 * (U_a + self.dt * self.eq.forward(prim_a, t+self.dt))
        return U_i_1


class RK2_SSP3(TSolver):
    def __init__(self, cells: FVMCells, dt: float, n_steps: int, equation, cfg: ConfigFVM):
        super().__init__(cells, dt, n_steps, eq=equation, cfg=cfg)
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
    def __init__(self, cells: FVMCells, dt: float, n_steps: int, equation, cfg: ConfigFVM):
        super().__init__(cells, dt, n_steps, eq=equation, cfg=cfg)
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


class Leapfrog2(TSolver):
    """ Leapfrog 2 solver
        Non-Markov solver.
    """
    def __init__(self, cells: FVMCells, dt: float, n_steps: int, equation, cfg: ConfigFVM):
        super().__init__(cells, dt, n_steps, eq=equation, cfg=cfg)
        self.eq: FVMEquation = equation

        self.U_tm1 = None

    def _step(self, t):
        """
        U_{t+1} = U_t-1 + dt * f(U_t, t)
        """
        prim_t, U_t = self.cells.get_values()
        dUdt_t = self.eq.forward(prim_t, t)

        # First iteration, use Euler step
        if self.U_tm1 is None:
            self.U_tm1 = U_t + self.dt * dUdt_t
            return self.U_tm1

        # U_{t+1} = U_t-1 + dt * f(U_t, t)
        U_t_1 = self.U_tm1 + 2 *self.dt * dUdt_t

        self.U_tm1 = U_t

        return U_t_1


class LeapfrogAss(TSolver):
    """ Asselin leapfrog 1 solver
        Non-Markov solver.
    """
    def __init__(self, cells: FVMCells, dt: float, n_steps: int, equation, cfg: ConfigFVM):
        super().__init__(cells, dt, n_steps, eq=equation, cfg=cfg)
        self.eq: FVMEquation = equation

        self.U_hat_tm1 = None

        self.even=True
    def _step(self, t):
        """
        U_{t+1} = U_t-1 + dt * f(U_t, t)
        """
        prim_t, U_t = self.cells.get_values()
        dUdt_t = self.eq.forward(prim_t, t)

        # First iteration, use Euler step
        if self.U_hat_tm1 is None:
            self.U_hat_tm1 =  U_t
            U_t_1 = U_t + self.dt * dUdt_t
            return U_t_1

        # U_{t+1} = U_t-1 + dt * f(U_t, t)
        U_t_1 = self.U_hat_tm1 + 2* self.dt * dUdt_t

        self.U_hat_tm1 = U_t + 0.6 * (self.U_hat_tm1 - 2 * U_t + U_t_1)

        return U_t_1


class Magazenkov(TSolver):
    """ Leapfrog + Adams-Bashforth 2 solver
        Non-Markov solver.
        Note: Takes two half-steps
    """
    def __init__(self, cells: FVMCells, dt: float, n_steps: int, equation, cfg: ConfigFVM):
        super().__init__(cells, dt, n_steps, eq=equation, cfg=cfg)
        self.eq: FVMEquation = equation
        self.dt = self.dt / 2 # Half step

        # self.prev_dUdt = deque(maxlen=2)

        self.U_tm1 = None
    #
    # def _init_states(self, t):
    #     prim, _ = self.cells.get_values()
    #     dUdt_0 = self.eq.forward(prim, t)
    #
    #     for _ in range(1):
    #         self.prev_dUdt.append(dUdt_0)

    def _step(self, t):
        """
        U_{t+1} = U_{t-1} + 2 * dt * f(U_t, t)
        U_{t+2} = U_t + dt/2 * [3 * f(U_{t+1}) - f(U_{t})]
        """
        prim_t, U_t = self.cells.get_values()

        # First iteration, use Euler step
        if self.U_tm1 is None:
            self.U_tm1 = U_t # + self.dt * dUdt_t

        dUdt_t = self.eq.forward(prim_t, t)
        # U_{t+1} = U_{t-1} + 2 * dt * f(U_t, t)
        U_t_1 = self.U_tm1 + 2 * self.dt * dUdt_t

        dUdt_t1 = self.eq.forward(prim_t, t + self.dt)
        # U_{t+2} = U_t + dt/2 * [3 * f(U_{t+1}) - f(U_{t})]
        U_t_2 = U_t_1 + self.dt/ 2 * (3 * dUdt_t1 - dUdt_t)

        self.U_tm1 = U_t_1
        return U_t_2


class Adams2(TSolver, Adaptive):
    """ Adams Bashforth 2 solver
        Non-Markov solver.
    """
    def __init__(self, cells: FVMCells, dt: float, n_steps: int, equation, cfg: ConfigFVM):
        super().__init__(cells, dt, n_steps, eq=equation, cfg=cfg)
        self.eq: FVMEquation = equation

        self.prev_dUdt = deque(maxlen=2)
        self._adapt_init(order=4, atol=3e-6, rtol=3e-6, alphas=(0.9, 0.99))

    def _init_states(self, t):
        prim, _ = self.cells.get_values()
        dUdt_0 = self.eq.forward(prim, t)

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

        dUdt_t = self.eq.forward(prim_t, t)
        dUdt_tm1 = self.prev_dUdt[-1]
        dUdt_tm2 = self.prev_dUdt[-2]

        U_1_high = U_t + self.dt/12 * (23 * dUdt_t - 16 * dUdt_tm1 + 5 * dUdt_tm2)
        U_1_low = U_t + self.dt/2 * (3 * dUdt_t - dUdt_tm1)
        # U_1_low = U_t + self.dt * dUdt_t
        self.prev_dUdt.append(dUdt_t)
        self.update_stepsize(U_1_high, U_1_low)

        return U_1_high


class Butcher(TSolver):
    def __init__(self, name, cells: FVMCells, dt: float, n_steps: int, equation, cfg: ConfigFVM):
        super().__init__(cells, dt, n_steps, eq=equation, cfg=cfg)
        """
        Initializes the solver with a Butcher tableau.

        Args:
            A (torch.Tensor): 2D tensor of stage coefficients with shape (s, s),
                              where s is the number of stages.
            b (torch.Tensor): 1D tensor of weights for combining stages.
            c (torch.Tensor): 1D tensor of time coefficients for each stage.
        """

        tables = Butcher_Tables(name, cells.device)

        self.A = tables.A
        self.b = tables.b
        self.c = tables.b
        self.stages = self.b.shape[0]

    def _step(self, t) -> torch.Tensor:
        """
        Take one step of the ODE solver.

        Returns:
            torch.Tensor: Updated state after one step.
        """

        state_0 = self.cells.state
        primatives, _ = self.cells.get_values()

        k = torch.zeros((self.stages, *state_0.shape), dtype=state_0.dtype, device=state_0.device)  # shape = [stages, n_cells, n_comp]
        for i in range(self.stages):
            if i == 0:
                increment = 0
            else:
                # Compute the increment for y using previous stages
                increment = (self.A[i, :i].unsqueeze(-1).unsqueeze(-1) * k[:i]).sum(dim=0)
            # Evaluate the derivative at the stage time and state
            k_i = self.dt * self._forward_state(state_0 + increment, t + self.c[i] * self.dt)
            k[i] = k_i

        # Combine stages to compute next state
        U_next_high = state_0 + torch.sum(self.b * k, dim=0)
        return U_next_high

# class IMPRKCSolver(TSolver):
#     # def __init__(self, f, s, shat, eta=2 / 13):
#     def __init__(self, cells: FVMCells, dt: float, n_steps: int, equation):#, A, b, c):
#         super().__init__(cells, dt, n_steps, eq=equation)
#         """
#         Initialize the improved RKC solver.
#
#         Parameters:
#         -----------
#         f : callable
#             Function f(t, y) defining the ODE y' = f(t,y). y must be a NumPy array.
#         s : int
#             Classical stage number.
#         shat : int
#             Extra stage number for improvement (typically 1 or a small integer).
#         eta : float, optional
#             Damping parameter (default 2/13 for second–order method).
#         """
#         self.s = 3
#         self.shat = 1
#         self.N = self.s + self.shat  # total number of stages for the method
#         self.eta = 2 / 13
#         self.theta = 1.0 / (self.shat + 1)
#         # Compute the second-order RKC coefficients and stage nodes.
#         self._compute_coefficients()
#
#     @staticmethod
#     def _acosh(x):
#         return np.log(x + np.sqrt(x * x - 1))
#
#     def _chebT(self, j, x):
#         # Chebyshev polynomial of first kind: T_j(x) = cosh(j*acosh(x)) for x>=1.
#         return np.cosh(j * self._acosh(x))
#
#     def _chebTprime(self, j, x):
#         # T'_j(x)= j*sinh(j*acosh(x))/sqrt(x^2-1)
#         return j * np.sinh(j * self._acosh(x)) / np.sqrt(x * x - 1)
#
#     def _chebTdoubleprime(self, j, x):
#         # T''_j(x)= j^2*cosh(j*acosh(x))/(x^2-1) - j*x*sinh(j*acosh(x))/( (x^2-1)**(3/2) )
#         return (j ** 2 * np.cosh(j * self._acosh(x)) / (x * x - 1)
#                 - j * x * np.sinh(j * self._acosh(x)) / ((x * x - 1) ** 1.5))
#
#     def _compute_coefficients(self):
#         """
#         Compute the coefficients for the second–order RKC method.
#         We compute arrays b, u, v, ũ (ut), γ̃ (gt) for j = 0,..., N.
#         (The index 0 is unused.)
#
#         The formulas (for 2 ≤ j ≤ s, extended here to j=1,...,N) are:
#
#             ω₀ = 1 + η/s²,
#             ω₁ = T'_s(ω₀) / T''_s(ω₀),
#
#             Choose b₀ = b₁ = b₂ = 1.
#             For j ≥ 3, set
#               b[j] = 1 / (T''_j(ω₀) * (T'_j(ω₀))²).
#
#             For j = 1:
#               ũ₁ = b₁·ω₁.
#             For j ≥ 2:
#               u[j] = 2 ω₀ (b[j]/b[j-1]),
#               v[j] = - (b[j]/b[j-2]),
#               ũ[j] = 2 ω₁ (b[j]/b[j-1]),
#               γ̃[j] = - (1 - b[j-1]*T_{j-1}(ω₀)) * ũ[j].
#
#         In addition, we compute the stage node values c[j] by the recurrence
#             c₀ = 0,  c₁ = ũ₁,
#             c[j] = u[j]*c[j-1] + v[j]*c[j-2] + ũ[j] + γ̃[j],   for j ≥ 2.
#         """
#         N = self.N
#         s = self.s
#         eta = self.eta
#
#         # ω₀ and ω₁ (note: ω₀ is based on the classical stage number s)
#         self.omega0 = 1 + eta / (s ** 2)
#         self.omega1 = self._chebTprime(s, self.omega0) / self._chebTdoubleprime(s, self.omega0)
#
#         # Allocate arrays (indices 0 .. N); index 0 is unused.
#         self.b = torch.zeros(N + 1)
#         self.u = torch.zeros(N + 1)
#         self.v = torch.zeros(N + 1)
#         self.ut = torch.zeros(N + 1)
#         self.gt = torch.zeros(N + 1)
#         self.c = torch.zeros(N + 1)  # stage nodes for evaluating f
#
#         # Set b[0], b[1], b[2]
#         self.b[0] = 1.0
#         self.b[1] = 1.0
#         self.b[2] = 1.0
#
#         # For j >= 3, compute b[j] via the Chebyshev formulas.
#         for j in range(3, N + 1):
#             Tprime = self._chebTprime(j, self.omega0)
#             Tdd = self._chebTdoubleprime(j, self.omega0)
#             self.b[j] = 1.0 / (Tdd * (Tprime ** 2))
#
#         # For j = 1, set ũ₁ = b₁·ω₁.
#         self.ut[1] = self.b[1] * self.omega1
#         # For j >= 2, compute u[j], v[j], ũ[j], and γ̃[j].
#         for j in range(2, N + 1):
#             self.u[j] = 2 * self.omega0 * (self.b[j] / self.b[j - 1])
#             self.v[j] = - (self.b[j] / self.b[j - 2])
#             self.ut[j] = 2 * self.omega1 * (self.b[j] / self.b[j - 1])
#             self.gt[j] = - (1 - self.b[j - 1] * self._chebT(j - 1, self.omega0)) * self.ut[j]
#
#         # Compute the stage nodes c[j]:
#         self.c[0] = 0.0
#         self.c[1] = self.ut[1]
#         for j in range(2, N + 1):
#             self.c[j] = (self.u[j] * self.c[j - 1] +
#                          self.v[j] * self.c[j - 2] +
#                          self.ut[j] + self.gt[j])
#
#     def _compute_cd_sequences(self):
#         """
#         Compute sequences c_j and d_j used for determining the parameters x₁ and x₂.
#         Here, we use the recurrences:
#             C₀ = 0,  C₁ = ũ₁,
#             C[j] = u[j]*C[j-1] + v[j]*C[j-2] + ũ[j] + γ̃[j],   for j ≥ 2,
#
#             D₀ = 0,  D₁ = 0,
#             D[j] = u[j]*D[j-1] + v[j]*D[j-2] + ũ[j]*C[j-1],   for j ≥ 2.
#         Returns:
#             C, D : NumPy arrays of length N+1.
#         """
#         N = self.N
#         C = torch.zeros(N + 1)
#         D = torch.zeros(N + 1)
#         C[0] = 0.0
#         C[1] = self.ut[1]
#         D[0] = 0.0
#         D[1] = 0.0
#         for j in range(2, N + 1):
#             C[j] = self.u[j] * C[j - 1] + self.v[j] * C[j - 2] + self.ut[j] + self.gt[j]
#             D[j] = self.u[j] * D[j - 1] + self.v[j] * D[j - 2] + self.ut[j] * C[j - 1]
#         return C, D
#
#     def _compute_hat(self, seq, j):
#         """
#         Compute the weighted (hat) value for index j from sequence seq:
#             hat_seq = sum_{l=0}^{j-1} theta*(1-theta)^l * seq[j-l]
#         """
#         hat_val = 0.0
#         for l in range(j):
#             hat_val += self.theta * (1 - self.theta) ** l * seq[j - l]
#         return hat_val
#
#     def _compute_x1_x2(self):
#         """
#         Compute the parameters x₁ and x₂ ensuring second order accuracy.
#         Using:
#             x₁ = (0.5·hat_C(N) - hat_D(N)) / (hat_C(N)*hat_D(N-1) - hat_C(N-1)*hat_D(N)),
#             x₂ = (1 - x₁·hat_C(N-1)) / hat_C(N),
#         where N = s+shat.
#         """
#         C, D = self._compute_cd_sequences()
#         N = self.N
#         hatC_Nm1 = self._compute_hat(C, N - 1)
#         hatC_N = self._compute_hat(C, N)
#         hatD_Nm1 = self._compute_hat(D, N - 1)
#         hatD_N = self._compute_hat(D, N)
#         numerator = 0.5 * hatC_N - hatD_N
#         denominator = hatC_N * hatD_Nm1 - hatC_Nm1 * hatD_N
#         x1 = numerator / denominator
#         x2 = (1 - x1 * hatC_Nm1) / hatC_N
#         return x1, x2
#
#     def forward(self, U, t):
#         return self.eq.forward(self.cells.convert_state_to_value(U)[0], t)
#
#     def _step(self, t):
#         """
#         Take one time step from (t, y) with step size h using the second order IMPRKC method.
#
#         The method computes stage values:
#           K₀ = y,      ˆK₀ = y,
#           K₁ = y + ũ₁·h·F₀,  ˆK₁ = α K₁ + (1-α)ˆK₀,
#           for j = 2,..., N:
#              Kⱼ = uⱼ Kⱼ₋₁ + vⱼ Kⱼ₋₂ + (1 - uⱼ - vⱼ) y + ũⱼ·h·Fⱼ₋₁ + γ̃ⱼ·h·F₀,
#              ˆKⱼ = α Kⱼ + (1-α)ˆKⱼ₋₁,
#           and then
#              yₙ₊₁ = (1 - x₁ - x₂) y + x₁ˆK_{N-1} + x₂ˆK_N.
#
#         Here, F₀ = f(t, y) and Fⱼ = f(t + cⱼ·h, Kⱼ) for j>=1.
#         """
#         N = self.N
#         shat = self.shat
#         alpha = 1.0 / (shat + 1)
#         beta = 1 - alpha
#
#         # Stage storage: K[j] and ˆK[j]
#         K = [None for _ in range(N + 1)]
#         Khat = [None for _ in range(N + 1)]
#
#         U_i = self.cells.state
#
#         # Stage 0.
#         K[0] = U_i
#         Khat[0] = U_i
#         F0 = self.forward(U_i, t)
#
#         # Stage 1.
#         K[1] = U_i + self.ut[1] * self.dt * F0
#         Khat[1] = alpha * K[1] + beta * Khat[0]
#
#         # For stages j = 2,..., N.
#         for j in range(2, N + 1):
#             # Evaluate f at stage: use t + c[j-1]*h and K[j-1]
#             t_stage = t + self.c[j - 1] * self.dt
#             Fjm1 = self.forward(K[j - 1], t_stage)
#             K[j] = (self.u[j] * K[j - 1] + self.v[j] * K[j - 2] +
#                     (1 - self.u[j] - self.v[j]) * U_i +
#                     self.ut[j] * self.dt * Fjm1 +
#                     self.gt[j] * self.dt * F0)
#             Khat[j] = alpha * K[j] + beta * Khat[j - 1]
#
#         # Compute parameters x₁ and x₂.
#         x1, x2 = self._compute_x1_x2()
#
#         # Final update.
#         y_next = (1 - x1 - x2) * U_i + x1 * Khat[N - 1] + x2 * Khat[N]
#         return y_next
