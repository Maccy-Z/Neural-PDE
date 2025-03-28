from __future__ import annotations
from cprint import c_print
import torch
from abc import ABC, abstractmethod
from codetiming import Timer
from matplotlib import pyplot as plt
import torch.profiler
from collections import deque
import math

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

        # density = torch.clamp(density, 0.01, 1e6)
        v_x, v_y = momentum_x / density, momentum_y / density

        primatives = torch.stack([v_x, v_y, density], dim=1)

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

        plot_i = int(0.2 / self.dt)
        Eks, Eps, ts, TVs = [], [], [], []

        for i in range(self.n_steps):
            t = i * self.dt
            with Timer(text=f"{i=}, {t=:.5g} Time: {{:.4g}}"):
                new_Us = self._step(t)
                self.cells.update_cells(new_Us)

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

            if i % plot_i == 0 and t>0.:
                c_print(f'{t = :.5g}', color="bright_yellow")

                primatives = self.cells.get_values()[0]
                Xlims = None # [[0.45, 0.52], [0.77, 0.84]]

                # xs = self.eq.mesh.centroids[:, 0]
                # mask = xs>1.5
                # prims = primatives[mask, 2]
                # print(prims)
                # print((prims-1).abs().mean())

                # dUdts = torch.stack(E_props.farfield_calc.dUdts).cpu()
                # plt.plot(dUdts)
                # plt.show()

                # self.eq.plot_interp(E_props.cell_grads[:, 1, 0], title=f"Grad t={i * self.dt :.4g}", Xlims=Xlims)
                # self.eq.plot_cells(div_KT[:, 0], title=f"Values at t={i * self.dt :.4g}", show_index=True, Xlims=Xlims)
                # self.eq.plot_flux(self.eq.div_V_face[:, 0], title=f"Vx t={i * self.dt :.4g}", show_index=False)
                self.eq.plot_interp(primatives[:], title=f"Values at t={i * self.dt :.4g}", Xlims=Xlims, )
                # self.eq.plot_cells(primatives[:, 0], title=f"Values at t={i * self.dt :.4g}", show_index=True, Xlims=Xlims)

                if torch.any(torch.isnan(primatives)):
                    print("Nan in primatives")
                    exit(9)
                # if t>0.2:
                #     exit("DONE PLOTTING")

        Eks, Eps = torch.tensor(Eks), torch.tensor(Eps)
        E = Eks + Eps
        plt.plot(ts, Eks, label="Kinetic Energy")
        plt.plot(ts, Eps, label="Potential Energy")
        plt.plot(ts, E, label="Total Energy")
        plt.legend()
        plt.show()



    #@torch.inference_mode()
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
        U_i_1 = U + self.dt * self.eq.forward(prim_a, t+self.dt)
        return U_i_1


# class Euler(TSolver):
#     def __init__(self, cells: FVMCells, dt: float, n_steps: int, equation):
#         super().__init__(cells, dt, n_steps, eq=equation)
#         self.eq = equation
#
#     def _step(self, t):
#         """U^{i+1} = U^i + dt * f(U^i)"""
#
#         dUdt, _ = self.eq.forward(self.cells.get_values()[0], t=t)
#         U_i_1 = self.cells.state + self.dt * dUdt
#
#         return U_i_1

class Euler(TSolver):
    def __init__(self, cells: FVMCells, dt: float, n_steps: int, equation):
        super().__init__(cells, dt, n_steps, eq=equation)
        self.eq = equation

    def _step(self, t):
        """
            f(U) = J U + N(U)
            U^{i+1} = U^i + dt * [J U^{i+1} + N(U^i)]
            [I - dt * J] U^{i+1} = U^i + dt * N(U^i)
        """
        prims, U_i = self.cells.get_values()
        dUdt, (J, diag_mask) = self.eq.forward(prims, t=t)

        n = J.shape[0]

        U_i_f = U_i.flatten()
        L_U_i_f = J @ U_i_f
        N_i = dUdt.flatten() - L_U_i_f

        # A = I - self.dt * J
        J_val = - self.dt * J.values()
        J_val[diag_mask] += 1
        A = torch.sparse_csr_tensor(J.crow_indices(), J.col_indices(), J_val, (n, n))

        b = U_i_f + self.dt * N_i
        #
        U_i_1 = self._spsolve(A, b)
        # U_i_1 = U_i_f + self.dt * (L_U_i_f + N_i)
        # U_i_1 = U_i + self.dt * dUdt
        return U_i_1.view(-1, 3)

    def _spsolve(self, A, b):
        import cupy as cp
        import cupyx.scipy.sparse as cusparse
        from pde.solvers.gmres import gmres
        import cupyx.scipy.sparse.linalg as cuslinalg

        cp_crow_indices = cp.from_dlpack(A.crow_indices())
        cp_col_indices = cp.from_dlpack(A.col_indices())
        cp_values = cp.from_dlpack(A.values())

        # Build a CuPy CSR matrix
        A = cusparse.csr_matrix((cp_values, cp_col_indices, cp_crow_indices), shape=A.shape)
        b = cp.from_dlpack(b)

        # x = cuslinalg.spsolve(A, b)
        x, info = gmres(A, b, maxiter=4, restart=4)

        # print(info)
        x = torch.from_dlpack(x)

        assert not torch.isnan(x).any()
        return x


class IMEX(TSolver):
    def __init__(self, cells: FVMCells, dt: float, n_steps: int, equation):
        super().__init__(cells, dt, n_steps, eq=equation)
        self.eq = equation

        # Explicit terms
        self.A_hat = torch.tensor([
            [0.0, 0,],
            [1, 0.0, ]
        ], dtype=torch.float32)

        self.b_hat = torch.tensor([1/2, 1/2], dtype=torch.float32)
        self.c_hat = torch.tensor([0.0, 1], dtype=torch.float32)
        # Implicit terms
        gamma = 1 - 1 / math.sqrt(2)
        self.A = torch.tensor([
            [gamma, 0,],
            [1-2*gamma, gamma,]
        ], dtype=torch.float32)

        self.b = torch.tensor([gamma, 1-gamma], dtype=torch.float32)
        self.c = torch.tensor([1/2, 1/2], dtype=torch.float32)

        # """  IMEX-SSP3(3,3,2) """
        # # Explicit terms
        # self.A_hat = torch.tensor([
        #     [0.0, 0, 0],
        #     [1., 0, 0],
        #     [1/4, 1/4, 0],
        # ], dtype=torch.float32)
        #
        # self.b_hat = torch.tensor([1/6, 1/6, 2/3], dtype=torch.float32)
        # self.c_hat = torch.tensor([0.0, 1, 1/2], dtype=torch.float32)
        # # Implicit terms
        # gamma = 1 - 1 / math.sqrt(2)
        # self.A = torch.tensor([
        #     [gamma, 0, 0.],
        #     [1 - 2 * gamma, gamma, 0],
        #     [1/2 - gamma, 0, gamma],
        # ], dtype=torch.float32)
        #
        # self.b = torch.tensor([1/6, 1/6, 2/3], dtype=torch.float32)
        # self.c = torch.tensor([gamma, 1-gamma, 1/2], dtype=torch.float32)

        """ IMEX-SSP2(3,2,2) """
        # # Explicit terms
        # self.A_hat = torch.tensor([
        #     [0.0, 0, 0],
        #     [0, 0, 0],
        #     [0, 1, 0],
        # ], dtype=torch.float32)
        #
        # self.b_hat = torch.tensor([0, 1/2, 1/2], dtype=torch.float32)
        # self.c_hat = torch.tensor([0.0, 0, 1], dtype=torch.float32)
        # # Implicit terms
        # self.A = torch.tensor([
        #     [1/2, 0, 0.],
        #     [-1/2, 1/2, 0],
        #     [0, 1/2, 1/2],
        # ], dtype=torch.float32)
        #
        # self.b = torch.tensor([0, 1/2, 1/2], dtype=torch.float32)
        # self.c = torch.tensor([1/2, 0, 1], dtype=torch.float32)

    def _get_parts(self, U, t):
        """ Forward equation, and get N and J """
        dUdt, (J, diag_mask) = self._forward(U, t)

        G_U_f = J @ U.flatten()
        N = dUdt.flatten() - G_U_f

        return N, G_U_f, J, diag_mask

    def _get_jacobian(self, U, t):
        """ Forward equation, and get N and J """
        J, diag_mask = self.eq.get_jacobian(self.cells.convert_state_to_value(U)[0], t)

        return J, diag_mask

    def _step(self, t):
        """
            f(U) = J U + N(U)
            U_i = U_0 + dt * [sum_j hat(a)_ij N(u_j) + sum_j a_ij (J@u_j)]
        """
        n = self.eq.E_props.n_cells * self.eq.E_props.n_component
        stages = self.A.shape[0]

        U_0 = self.cells.state

        Ns = torch.zeros([stages, n], device=self.eq.device)
        Gs = torch.zeros([stages, n], device=self.eq.device)

        J, diag_mask = self._get_jacobian(U_0, t)
        U0_f = U_0.flatten()

        for i in range(stages):

            # Explicit right side
            RHS = U0_f
            for j in range(i):
                RHS = RHS + self.dt * (self.A_hat[i, j] * Ns[j] + self.A[i, j] * Gs[j])
            # Implicit left contribution: I - dt * J
            I_m_dtJ = - self.dt * self.A[i, i] * J.values()
            # I_m_dtJ = torch.zeros_like(J.values())
            I_m_dtJ[diag_mask] += 1
            I_m_dtJ = torch.sparse_csr_tensor(J.crow_indices(), J.col_indices(), I_m_dtJ, (n, n))
            # Solve (I - dt * J) U_{i+1} = RHS(U_i)
            # print(f'{i}, {torch.any(torch.isnan(RHS))}')
            U_i_1 = self._spsolve(I_m_dtJ, RHS)
            # Update buffers with new values
            N_i, G_i, J, diag_mask = self._get_parts(U_i_1.view(-1, 3), t + self.c[i]*self.dt)
            Ns[i] = N_i
            Gs[i] = G_i

        # b = U + self.dt * N

        # U_i_1 = U_i_f + self.dt * (L_U_i_f + N_i)
        # U_i_1 = U_i + self.dt * dUdt
        U_next = U0_f
        for i in range(stages):
            U_next = U_next + self.dt * (self.b_hat[i] * Ns[i] + self.b[i] * Gs[i])

        return U_next.view(-1, 3)

    def _spsolve(self, A, b):
        import cupy as cp
        import cupyx.scipy.sparse as cusparse
        from pde.solvers.gmres import gmres
        import cupyx.scipy.sparse.linalg as cuslinalg

        cp_crow_indices = cp.from_dlpack(A.crow_indices())
        cp_col_indices = cp.from_dlpack(A.col_indices())
        cp_values = cp.from_dlpack(A.values())

        # Build a CuPy CSR matrix
        A = cusparse.csr_matrix((cp_values, cp_col_indices, cp_crow_indices), shape=A.shape)
        b = cp.from_dlpack(b)

        # x = cuslinalg.spsolve(A, b)
        x, info = gmres(A, b, maxiter=10, restart=10)

        # print(info)
        x = torch.from_dlpack(x)

        assert not torch.isnan(x).any()
        return x

    def _forward(self, U, t):
        return self.eq.forward(self.cells.convert_state_to_value(U)[0], t)


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

        dUdt_star = self.eq.forward(primatives, t=t)
        U_star = state + 0.5 * self.dt * dUdt_star        # U_{i+0.5}

        primatives_star, _ = self.cells.convert_state_to_value(U_star)
        dUdt = self.eq.forward(primatives_star, t=t)
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
        dUdt_star = self.eq.forward(self.cells.get_values()[0], t=t)
        U_star = self.cells.state + self.dt * dUdt_star

        # y_{n+1} = y_n + 0.5*dt*[f(t_n, y_n) + f(t_n+1, y_star)]
        primatives_star, _ = self.cells.convert_state_to_value(U_star)
        dUdt = self.eq.forward(primatives_star, t=t)

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
        U_a = U_i + self.dt * self.eq.forward(prim_i, t)

        # U_b = 3/4 * U_i + 1/4 * dt * f(U_a)
        prim_a, U_a = self.cells.convert_state_to_value(U_a)
        U_b = 3/4 * U_i + 1/4 * (U_a + self.dt * self.eq.forward(prim_a, t+self.dt))

        # U_{i+1} = 1/3 * U_i + 2/3 [U_b + dt * f(U_b)]
        prim_b, U_b = self.cells.convert_state_to_value(U_b)
        U_i_1 = 1/3 * U_i + 2/3 * (U_b + self.dt * self.eq.forward(prim_b, t+self.dt/2))
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
        U_a = U_i + self.dt * self.eq.forward(prim_i, t)

        # U_{i+1} = 0.5 * U_i + 0.5 * [U_a + dt * f(U_a)]
        prim_a, U_a = self.cells.convert_state_to_value(U_a)
        U_i_1 = 0.5 * U_i + 0.5 * (U_a + self.dt * self.eq.forward(prim_a, t+self.dt))
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

        U_i = self.cells.state
        # U_a = 1/2 * U_i + 1/2 * [U_i + dt * f(U_i)]
        U_a = 1/2 * U_i + 1/2 * self._euler_step(U_i, t=t)

        # U_b = 1/2 * U_a + 1/2 * [U_a + dt * f(U_a)]
        U_b = 1/2 * U_a + 1/2 * self._euler_step(U_a, t=t+self.dt/2)

        # U_c = 2/3 * U_i + 1/6 * U_b + 1/6 * [U_b + dt * f(U_b)]
        U_c = 2/3 * U_i + 1/6 * U_b + 1/6 * self._euler_step(U_b, t=t+self.dt)

        # U_{i+1} = 1/2 * U_c + 1/2 [U_c + dt * f(U_c)]
        U_i_1 = 1/2 * U_c + 1/2 * self._euler_step(U_c, t=t+self.dt/2)

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

        prim_t, U_t = self.cells.get_prims()

        dUdt_t = self.eq.forward(prim_t, t)
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

        self.prev_dUdt = deque(maxlen=1)

    def _init_states(self, t):
        prim, _ = self.cells.get_values()
        dUdt_0 = self.eq.forward(prim, t)

        for _ in range(1):
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

        dUdt_t = self.eq.forward(prim_t, t)
        dUdt_tm1 = self.prev_dUdt[-1]
        # dUdt_tm2 = self.prev_dUdt[-2]

        # U_a = U_t + dt/2 * [3 * f(U_t) - f(U_{t-1})]
        U_a = U_t + self.dt/2 * (3 * dUdt_t - dUdt_tm1)

        # U_{t+1} = U_t + dt/12 * [5 * f(U_a) + 8 * f(U_{t}) - 1 * f(U_{t-1})]
        prim_a = self.cells.convert_state_to_value(U_a)[0]
        dUdt_a = self.eq.forward(prim_a, t)
        U_t_1 =  U_t + self.dt/12 * (5 * dUdt_a + 8 * dUdt_t - dUdt_tm1)

        self.prev_dUdt.append(dUdt_t)
        return U_t_1


class Adams4PC(TSolver):
    """ Adams–Bashforth–Moulton predictor corrector 4 solver
        Non-Markov solver.
    """
    def __init__(self, cells: FVMCells, dt: float, n_steps: int, equation):
        super().__init__(cells, dt, n_steps, eq=equation)
        self.eq: FVMEquation = equation

        self.prev_dUdt = deque(maxlen=2)

    def __euler_step(self, U, t):
        prim, _ = self.cells.convert_state_to_value(U)
        dUdt = self.eq.forward(prim, t)
        U_i_1 = U + self.dt * dUdt

        return U_i_1, dUdt

    def _init_states(self, t):
        U_i = self.cells.state
        dUdt = self.eq.forward(U_i, t)

        for _ in range(2):
            self.prev_dUdt.append(dUdt)

    def _step(self, t):
        """
        U_a = U_t + dt/24 * [55 * f(U_t) - 59 * f(U_{t-1}) + 37 * f(U_{t-2}) - 9 * f(U_{t-3})]  ( Or other order predcitor)
        U_{t+1} = U_t + dt/24 * [9 * f(U_a) + 19 * f(U_{t}) - 5 * f(U_{t-1}) + f(U_{t-2})]
        """
        if len(self.prev_dUdt) == 0:
            self._init_states(t)

        prim_t, U_t = self.cells.get_values()

        dUdt_t = self.eq.forward(prim_t, t)
        dUdt_tm1 = self.prev_dUdt[-1]
        dUdt_tm2 = self.prev_dUdt[-2]
        # dUdt_tm3 = self.prev_dUdt[-3]

        # U_a = U_t + dt/24 * [55 * f(U_t) - 59 * f(U_{t-1}) + 37 * f(U_{t-2}) - 9 * f(U_{t-3})]
        # U_a = U_t + self.dt  * dUdt_t
        U_a = U_t + self.dt / 2 * (3 * dUdt_t - dUdt_tm1)
        # U_a = U_t + self.dt/12 * (23 * dUdt_t - 16 * dUdt_tm1 + 6 * dUdt_tm2)
        # U_a = U_t + self.dt/24 * (55 * dUdt_t - 59 * dUdt_tm1 + 37 * dUdt_tm2 - 9 * dUdt_tm3)

        # U_{t+1} = U_t + dt/24 * [9 * f(U_a) + 19 * f(U_{t}) - 5 * f(U_{t-1}) + f(U_{t-2})]
        prim_a = self.cells.convert_state_to_value(U_a)[0]
        dUdt_a = self.eq.forward(prim_a, t)
        U_t_1 =  U_t + self.dt/24 * (9 * dUdt_a + 19 * dUdt_t - 5 * dUdt_tm1 + dUdt_tm2)

        self.prev_dUdt.append(dUdt_t)

        return U_t_1


class Butcher(TSolver):
    def __init__(self, cells: FVMCells, dt: float, n_steps: int, equation):#, A, b, c):
        super().__init__(cells, dt, n_steps, eq=equation)
    #def __init__(self, A: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
        """
        Initializes the solver with a Butcher tableau.

        Args:
            A (torch.Tensor): 2D tensor of stage coefficients with shape (s, s),
                              where s is the number of stages.
            b (torch.Tensor): 1D tensor of weights for combining stages.
            c (torch.Tensor): 1D tensor of time coefficients for each stage.
        """
        """ RK4 """
        # A = torch.tensor([
        #     [0.0, 0.0, 0.0, 0.0],
        #     [0.5, 0.0, 0.0, 0.0],
        #     [0.0, 0.5, 0.0, 0.0],
        #     [0.0, 0.0, 1.0, 0.0]
        # ], dtype=torch.float32)
        #
        # b = torch.tensor([1 / 6, 1 / 3, 1 / 3, 1 / 6], dtype=torch.float32)
        # c = torch.tensor([0.0, 0.5, 0.5, 1.0], dtype=torch.float32)

        """ RK3 SSP4 """
        # A = torch.tensor([
        #     [0.0, 0.0, 0.0, 0.0],
        #     [0.5, 0.0, 0.0, 0.0],
        #     [0.5, 0.5, 0.0, 0.0],
        #     [1/6, 1/6, 1/6, 0.0]
        # ], dtype=torch.float32).cuda()
        #
        # b = torch.tensor([1 / 6, 1 / 6, 1 / 6, 1 / 2], dtype=torch.float32).cuda()
        # c = torch.tensor([0.0, 0.5, 1, 0.5], dtype=torch.float32)

        """ RK3 SSP5"""
        A = torch.tensor([
            [0.0, 0.0, 0.0, 0.0, 0],
            [0.37726891511710, 0.0, 0.0, 0.0, 0],
            [0.37726891511710, 0.37726891511710, 0.0, 0.0, 0],
            [0.16352294089771, 0.16352294089771, 0.16352294089771, 0.0, 0],
            [0.14904059394856, 0.14831273384724, 0.14831273384724, 0.34217696850008, 0],
        ], dtype=torch.float32).cuda()

        b = torch.tensor([0.19707596384481, 0.11780316509765, 0.11709725193772, 0.27015874934251, 0.29786487010104], dtype=torch.float32).cuda()
        c = torch.tensor([0, 0.37726891511710 , 0.75453783023419 , 0.49056882269314 , 0.78784303014311 ], dtype=torch.float32)

        """ RK4 SSP5 """
        # A = torch.tensor([
        #     [0.0, 0.0, 0.0, 0.0, 0],
        #     [0.39175222700392, 0.0, 0.0, 0.0, 0],
        #     [0.21766909633821, 0.36841059262959, 0.0, 0.0, 0],
        #     [0.08269208670950, 0.13995850206999,  0.25189177424738, 0.0, 0],
        #     [0.06796628370320, 0.11503469844438, 0.20703489864929, 0.54497475021237, 0],
        # ], dtype=torch.float32).cuda()
        #
        # b = torch.tensor([0.14681187618661, 0.24848290924556, 0.10425883036650, 0.27443890091960, 0.22600748319395], dtype=torch.float32).cuda()
        # c = torch.tensor([0., 0.39175222700392, 0.58607968896779 , 0.47454236302687, 0.93501063100924], dtype=torch.float32)


        self.A = A
        self.b = b.reshape(-1, 1, 1)
        self.c = c
        self.stages = b.shape[0]

    def _step(self, t) -> torch.Tensor:
        """
        Take one step of the ODE solver.

        Returns:
            torch.Tensor: Updated state after one step.
        """

        state_0 = self.cells.state
        primatives, _ = self.cells.get_values()


        k = torch.zeros((self.stages, *state_0.shape), dtype=state_0.dtype, device=state_0.device)
        for i in range(self.stages):
            if i == 0:
                increment = 0
            else:
                # Compute the increment for y using previous stages
                increment = (self.A[i, :i].unsqueeze(-1) * k[:i].view(i, -1)).sum(dim=0)
                increment = increment.view(state_0.shape)
            # Evaluate the derivative at the stage time and state
            k_i =  self.dt * self.eq.forward(self.cells.convert_state_to_value(state_0 + increment)[0], t + self.c[i] * self.dt)
            k[i] = k_i

        # k = torch.stack(k)
        # Combine stages to compute next state
        state_next = state_0 + torch.sum(self.b * k, dim=0)
        return state_next


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
