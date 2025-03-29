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

        plot_i = int(1 / self.dt)
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
        U_i_1 = U + self.dt * self.eq.forward(prim_a, t+self.dt)
        return U_i_1



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
