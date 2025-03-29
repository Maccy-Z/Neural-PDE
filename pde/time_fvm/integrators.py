from collections import deque
import math
import torch
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from time_fvm import FVMEquation


from pde.time_fvm.t_solvers import TSolver, FVMCells

class Euler(TSolver):
    def __init__(self, cells: FVMCells, dt: float, n_steps: int, equation):
        super().__init__(cells, dt, n_steps, eq=equation)
        self.eq = equation

    def _step(self, t):
        """U^{i+1} = U^i + dt * f(U^i)"""

        dUdt, _ = self.eq.forward(self.cells.get_values()[0], t=t)
        U_i_1 = self.cells.state + self.dt * dUdt

        return U_i_1


# class IMEX_Euler(TSolver):
#     def __init__(self, cells: FVMCells, dt: float, n_steps: int, equation):
#         super().__init__(cells, dt, n_steps, eq=equation)
#         self.eq = equation
#
#     def _step(self, t):
#         """
#             f(U) = J U + N(U)
#             U^{i+1} = U^i + dt * [J U^{i+1} + N(U^i)]
#             [I - dt * J] U^{i+1} = U^i + dt * N(U^i)
#         """
#         prims, U_i = self.cells.get_values()
#         dUdt = self.eq.forward(prims, t=t)
#         (J, diag_mask) = self.eq.get_jacobian()
#
#         n = J.shape[0]
#
#         U_i_f = U_i.flatten()
#         L_U_i_f = J @ U_i_f
#         N_i = dUdt.flatten() - L_U_i_f
#
#         # A = I - self.dt * J
#         # J_val = - self.dt * J.values()
#         # J_val[diag_mask] += 1
#         # A = torch.sparse_csr_tensor(J.crow_indices(), J.col_indices(), J_val, (n, n))
#
#         indices = torch.arange(n).unsqueeze(0).repeat(2, 1)  # shape: [2, N]
#         values = torch.ones(n)
#         I = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce().cuda()
#         A = I - self.dt * J
#         A = A.to_sparse_csr()
#
#         b = U_i_f + self.dt * N_i
#         #
#         U_i_1 = self._spsolve(A, b)
#         # U_i_1 = U_i_f + self.dt * (L_U_i_f + N_i)
#         # U_i_1 = U_i + self.dt * dUdt
#         return U_i_1.view(-1, 3)
#
#     def _spsolve(self, A, b):
#         import cupy as cp
#         import cupyx.scipy.sparse as cusparse
#         from pde.solvers.gmres import gmres
#
#         cp_crow_indices = cp.from_dlpack(A.crow_indices())
#         cp_col_indices = cp.from_dlpack(A.col_indices())
#         cp_values = cp.from_dlpack(A.values())
#
#         # Build a CuPy CSR matrix
#         A = cusparse.csr_matrix((cp_values, cp_col_indices, cp_crow_indices), shape=A.shape)
#         b = cp.from_dlpack(b)
#
#         # x = cuslinalg.spsolve(A, b)
#         x, info = gmres(A, b, maxiter=4, restart=4)
#
#         # print(info)
#         x = torch.from_dlpack(x)
#
#         assert not torch.isnan(x).any()
#         return x
#
#
# class IMEX(TSolver):
#     def __init__(self, cells: FVMCells, dt: float, n_steps: int, equation):
#         super().__init__(cells, dt, n_steps, eq=equation)
#         self.eq = equation
#
#         # """ IMEX-(2, 2, 2)"""
#         # # Explicit terms
#         # self.A_hat = torch.tensor([
#         #     [0.0, 0,],
#         #     [1, 0.0, ]
#         # ], dtype=torch.float32)
#         #
#         # self.b_hat = torch.tensor([1/2, 1/2], dtype=torch.float32)
#         # self.c_hat = torch.tensor([0.0, 1], dtype=torch.float32)
#         # # Implicit terms
#         # gamma = 1 - 1 / math.sqrt(2)
#         # self.A = torch.tensor([
#         #     [gamma, 0,],
#         #     [1-2*gamma, gamma,]
#         # ], dtype=torch.float32)
#         #
#         # self.b = torch.tensor([gamma, 1-gamma], dtype=torch.float32)
#         # self.c = torch.tensor([1/2, 1/2], dtype=torch.float32)
#
#         # """  IMEX-SSP3(3,3,2) """
#         # # Explicit terms
#         # self.A_hat = torch.tensor([
#         #     [0.0, 0, 0],
#         #     [1., 0, 0],
#         #     [1/4, 1/4, 0],
#         # ], dtype=torch.float32)
#         #
#         # self.b_hat = torch.tensor([1/6, 1/6, 2/3], dtype=torch.float32)
#         # self.c_hat = torch.tensor([0.0, 1, 1/2], dtype=torch.float32)
#         # # Implicit terms
#         # gamma = 1 - 1 / math.sqrt(2)
#         # self.A = torch.tensor([
#         #     [gamma, 0, 0.],
#         #     [1 - 2 * gamma, gamma, 0],
#         #     [1/2 - gamma, 0, gamma],
#         # ], dtype=torch.float32)
#         #
#         # self.b = torch.tensor([1/6, 1/6, 2/3], dtype=torch.float32)
#         # self.c = torch.tensor([gamma, 1-gamma, 1/2], dtype=torch.float32)
#
#         # """ IMEX-SSP2(3,2,2) """
#         # # Explicit terms
#         # self.A_hat = torch.tensor([
#         #     [0.0, 0, 0],
#         #     [0, 0, 0],
#         #     [0, 1, 0],
#         # ], dtype=torch.float32)
#         #
#         # self.b_hat = torch.tensor([0, 1/2, 1/2], dtype=torch.float32)
#         # self.c_hat = torch.tensor([0.0, 0, 1], dtype=torch.float32)
#         # # Implicit terms
#         # self.A = torch.tensor([
#         #     [1/2, 0, 0.],
#         #     [-1/2, 1/2, 0],
#         #     [0, 1/2, 1/2],
#         # ], dtype=torch.float32)
#         #
#         # self.b = torch.tensor([0, 1/2, 1/2], dtype=torch.float32)
#         # self.c = torch.tensor([1/2, 0.01, 1], dtype=torch.float32)
#
#         """ IMEX-SSP2(2,3,2) """
#         # Explicit terms
#         self.A_hat = torch.tensor([
#             [0.0, 0, 0],
#             [0.711664700366941, 0, 0],
#             [0.077338168947683, 0.917273367886007, 0],
#         ], dtype=torch.float32)
#
#         self.b_hat = torch.tensor([	0.398930808264688, 0.345755244189623, 0.255313947545689], dtype=torch.float32)
#         self.c_hat = torch.tensor([0.0, 0, 1], dtype=torch.float32)
#         # Implicit terms
#         self.A = torch.tensor([
#             [0, 0, 0.],
#             [0.353842865099275, 0.353842865099275, 0],
#             [0.398930808264689,	0.345755244189622,	0.255313947545689],
#         ], dtype=torch.float32)
#
#         self.b = torch.tensor([	0.398930808264688,	0.345755244189623,	0.255313947545689], dtype=torch.float32)
#         self.c = torch.tensor([0, 0.707685730198550, 1], dtype=torch.float32)
#
#     def _get_parts(self, U, t):
#         """ Forward equation, and get N and J """
#         dUdt = self._forward(U, t)
#         J, diag_mask = self._get_jacobian(U=U, t=t)
#
#         G_U_f = J @ U.flatten()
#         N = dUdt.flatten() - G_U_f
#
#         return N, G_U_f, J, diag_mask
#
#     def _get_jacobian(self, U=None, t=None):
#         """ Get """
#         if t == 0:
#             prims = self.cells.convert_state_to_value(U)[0]
#         else:
#             prims=None
#
#         J, diag_mask = self.eq.get_jacobian(prims=prims, t=t)
#
#         return J, diag_mask
#
#     def _step(self, t):
#         """
#             f(U) = J U + N(U)
#             U_i = U_0 + dt * [sum_j hat(a)_ij N(u_j) + sum_j a_ij (J@u_j)]
#         """
#         print()
#
#         n = self.eq.E_props.n_cells * self.eq.E_props.n_component
#         stages = self.A.shape[0]
#
#         U_0 = self.cells.state
#
#         Ns = torch.zeros([stages, n], device=self.eq.device)
#         Gs = torch.zeros([stages, n], device=self.eq.device)
#
#         J, diag_mask = self._get_jacobian(U_0, t)
#         U0_f = U_0.flatten()
#
#         for i in range(stages):
#
#             # Explicit right side
#             RHS = U0_f
#             for j in range(i):
#                 RHS = RHS + self.dt * (self.A_hat[i, j] * Ns[j] + self.A[i, j] * Gs[j])
#             # Implicit left contribution: I - dt * J
#
#
#             # # Solve (I - dt * A J) U_{i+1} = RHS(U_i)
#             # indices = torch.arange(n, device="cuda").unsqueeze(0).repeat(2, 1)  # shape: [2, N]
#             # values = torch.ones(n, device="cuda")
#             # I = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()
#             # I_m_dtJ = I - self.dt * self.A[i, i]  * J
#             # I_m_dtJ = I_m_dtJ.to_sparse_csr()
#             # U_i_1 = self._spsolve(I_m_dtJ, RHS)
#
#             # Diagonal solve (I - dt * A J) U_{i+1} = RHS(U_i)
#             J_diag = J.values()
#             I_m_dtJ = 1 - self.dt * self.A[i, i] * J_diag
#             U_i_1 = RHS / I_m_dtJ
#
#             # print(test.abs().max())
#             # print(RHS.abs().max() )
#             # I_m_dtJ = - self.dt * self.A[i, i] * J.values()
#             # # I_m_dtJ = torch.zeros_like(J.values())
#             # I_m_dtJ[diag_mask] += 1
#             # I_m_dtJ = torch.sparse_csr_tensor(J.crow_indices(), J.col_indices(), I_m_dtJ, (n, n))
#
#             # print(f'{i}, {torch.any(torch.isnan(RHS))}')
#             # Update buffers with new values
#             N_i, G_i, J, diag_mask = self._get_parts(U_i_1.view(-1, 3), t + self.c[i]*self.dt)
#             Ns[i] = N_i
#             Gs[i] = G_i
#
#         # b = U + self.dt * N
#
#         # U_i_1 = U_i_f + self.dt * (L_U_i_f + N_i)
#         # U_i_1 = U_i + self.dt * dUdt
#         U_next = U0_f
#         for i in range(stages):
#             U_next = U_next + self.dt * (self.b_hat[i] * Ns[i] + self.b[i] * Gs[i])
#
#         return U_next.view(-1, 3)
#
#     def _spsolve(self, A, b):
#         import cupy as cp
#         import cupyx.scipy.sparse as cusparse
#         from pde.solvers.gmres import gmres
#
#         cp_crow_indices = cp.from_dlpack(A.crow_indices())
#         cp_col_indices = cp.from_dlpack(A.col_indices())
#         cp_values = cp.from_dlpack(A.values())
#
#         # Build a CuPy CSR matrix
#         A = cusparse.csr_matrix((cp_values, cp_col_indices, cp_crow_indices), shape=A.shape)
#         b = cp.from_dlpack(b)
#
#         # x = cuslinalg.spsolve(A, b)
#         x, info = gmres(A, b, maxiter=10, restart=10)
#
#         # print(info)
#         x = torch.from_dlpack(x)
#
#         assert not torch.isnan(x).any()
#         return x
#
#     def _forward(self, U, t):
#         return self.eq.forward(self.cells.convert_state_to_value(U)[0], t)


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
        U_a = 1/2 * (U_i + self._euler_step(U_i, t=t))

        # U_b = 1/2 * U_a + 1/2 * [U_a + dt * f(U_a)]
        U_b = 1/2 * (U_a +  self._euler_step(U_a, t=t+self.dt/2))

        # U_c = 2/3 * U_i + 1/6 * U_b + 1/6 * [U_b + dt * f(U_b)]
        U_c = 2/3 * U_i + 1/6 * (U_b + self._euler_step(U_b, t=t+self.dt))

        # U_{i+1} = 1/2 * U_c + 1/2 [U_c + dt * f(U_c)]
        U_i_1 = 1/2 * (U_c + self._euler_step(U_c, t=t+self.dt/2))

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
        A = torch.tensor([
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
            [1/6, 1/6, 1/6, 0.0]
        ], dtype=torch.float32).cuda()

        b = torch.tensor([1 / 6, 1 / 6, 1 / 6, 1 / 2], dtype=torch.float32).cuda()
        c = torch.tensor([0.0, 0.5, 1, 0.5], dtype=torch.float32)

        # """ RK3 SSP5"""
        # A = torch.tensor([
        #     [0.0, 0.0, 0.0, 0.0, 0],
        #     [0.37726891511710, 0.0, 0.0, 0.0, 0],
        #     [0.37726891511710, 0.37726891511710, 0.0, 0.0, 0],
        #     [0.16352294089771, 0.16352294089771, 0.16352294089771, 0.0, 0],
        #     [0.14904059394856, 0.14831273384724, 0.14831273384724, 0.34217696850008, 0],
        # ], dtype=torch.float32).cuda()
        #
        # b = torch.tensor([0.19707596384481, 0.11780316509765, 0.11709725193772, 0.27015874934251, 0.29786487010104], dtype=torch.float32).cuda()
        # c = torch.tensor([0, 0.37726891511710 , 0.75453783023419 , 0.49056882269314 , 0.78784303014311 ], dtype=torch.float32)

        # """ RK3 SSP6"""
        # A = torch.tensor([
        #     [0.0, 0.0, 0.0, 0.0, 0, 0],
        #     [0.28422, 0.0, 0.0, 0.0, 0, 0],
        #     [0.28422, 0.28422, 0.0, 0.0, 0, 0],
        #     [0.23071, 0.23071, 0.23071, 0.0, 0, 0],
        #     [0.13416, 0.13416, 0.13416, 0.16528, 0, 0],
        #     [0.13416, 0.13416, 0.13416, 0.16528, 0.28422, 0]
        # ], dtype=torch.float32).cuda()
        #
        # b = torch.tensor([0.17016,  0.17016,  0.10198,  0.12563,  0.21604,  0.21604], dtype=torch.float32).cuda()
        # c = torch.tensor([0, 0.28422, 0.56844 , 0.69213, 0.56776, 0.85198], dtype=torch.float32)

        # """ RK4 SSP5 """
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

