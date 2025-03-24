import numpy as np


class IMPRKCSolver:
        def __init__(self, f, s, shat, eta=2 / 13):
                """
                Initialize the improved RKC solver.

                Parameters:
                -----------
                f : callable
                    Function f(t, y) defining the ODE y' = f(t,y). y must be a NumPy array.
                s : int
                    Classical stage number.
                shat : int
                    Extra stage number for improvement (typically 1 or a small integer).
                eta : float, optional
                    Damping parameter (default 2/13 for second–order method).
                """
                self.f = f
                self.s = s
                self.shat = shat
                self.N = s + shat  # total number of stages for the method
                self.eta = eta
                self.theta = 1.0 / (shat + 1)
                # Compute the second-order RKC coefficients and stage nodes.
                self._compute_coefficients()

        @staticmethod
        def _acosh(x):
                return np.log(x + np.sqrt(x * x - 1))

        def _chebT(self, j, x):
                # Chebyshev polynomial of first kind: T_j(x) = cosh(j*acosh(x)) for x>=1.
                return np.cosh(j * self._acosh(x))

        def _chebTprime(self, j, x):
                # T'_j(x)= j*sinh(j*acosh(x))/sqrt(x^2-1)
                return j * np.sinh(j * self._acosh(x)) / np.sqrt(x * x - 1)

        def _chebTdoubleprime(self, j, x):
                # T''_j(x)= j^2*cosh(j*acosh(x))/(x^2-1) - j*x*sinh(j*acosh(x))/( (x^2-1)**(3/2) )
                return (j ** 2 * np.cosh(j * self._acosh(x)) / (x * x - 1)
                        - j * x * np.sinh(j * self._acosh(x)) / ((x * x - 1) ** 1.5))

        def _compute_coefficients(self):
                """
                Compute the coefficients for the second–order RKC method.
                We compute arrays b, u, v, ũ (ut), γ̃ (gt) for j = 0,..., N.
                (The index 0 is unused.)

                The formulas (for 2 ≤ j ≤ s, extended here to j=1,...,N) are:

                    ω₀ = 1 + η/s²,
                    ω₁ = T'_s(ω₀) / T''_s(ω₀),

                    Choose b₀ = b₁ = b₂ = 1.
                    For j ≥ 3, set
                      b[j] = 1 / (T''_j(ω₀) * (T'_j(ω₀))²).

                    For j = 1:
                      ũ₁ = b₁·ω₁.
                    For j ≥ 2:
                      u[j] = 2 ω₀ (b[j]/b[j-1]),
                      v[j] = - (b[j]/b[j-2]),
                      ũ[j] = 2 ω₁ (b[j]/b[j-1]),
                      γ̃[j] = - (1 - b[j-1]*T_{j-1}(ω₀)) * ũ[j].

                In addition, we compute the stage node values c[j] by the recurrence
                    c₀ = 0,  c₁ = ũ₁,
                    c[j] = u[j]*c[j-1] + v[j]*c[j-2] + ũ[j] + γ̃[j],   for j ≥ 2.
                """
                N = self.N
                s = self.s
                eta = self.eta

                # ω₀ and ω₁ (note: ω₀ is based on the classical stage number s)
                self.omega0 = 1 + eta / (s ** 2)
                self.omega1 = self._chebTprime(s, self.omega0) / self._chebTdoubleprime(s, self.omega0)

                # Allocate arrays (indices 0 .. N); index 0 is unused.
                self.b = np.zeros(N + 1)
                self.u = np.zeros(N + 1)
                self.v = np.zeros(N + 1)
                self.ut = np.zeros(N + 1)
                self.gt = np.zeros(N + 1)
                self.c = np.zeros(N + 1)  # stage nodes for evaluating f

                # Set b[0], b[1], b[2]
                self.b[0] = 1.0
                self.b[1] = 1.0
                self.b[2] = 1.0

                # For j >= 3, compute b[j] via the Chebyshev formulas.
                for j in range(3, N + 1):
                        Tprime = self._chebTprime(j, self.omega0)
                        Tdd = self._chebTdoubleprime(j, self.omega0)
                        self.b[j] = 1.0 / (Tdd * (Tprime ** 2))

                # For j = 1, set ũ₁ = b₁·ω₁.
                self.ut[1] = self.b[1] * self.omega1
                # For j >= 2, compute u[j], v[j], ũ[j], and γ̃[j].
                for j in range(2, N + 1):
                        self.u[j] = 2 * self.omega0 * (self.b[j] / self.b[j - 1])
                        self.v[j] = - (self.b[j] / self.b[j - 2])
                        self.ut[j] = 2 * self.omega1 * (self.b[j] / self.b[j - 1])
                        self.gt[j] = - (1 - self.b[j - 1] * self._chebT(j - 1, self.omega0)) * self.ut[j]

                # Compute the stage nodes c[j]:
                self.c[0] = 0.0
                self.c[1] = self.ut[1]
                for j in range(2, N + 1):
                        self.c[j] = (self.u[j] * self.c[j - 1] +
                                     self.v[j] * self.c[j - 2] +
                                     self.ut[j] + self.gt[j])

        def _compute_cd_sequences(self):
                """
                Compute sequences c_j and d_j used for determining the parameters x₁ and x₂.
                Here, we use the recurrences:
                    C₀ = 0,  C₁ = ũ₁,
                    C[j] = u[j]*C[j-1] + v[j]*C[j-2] + ũ[j] + γ̃[j],   for j ≥ 2,

                    D₀ = 0,  D₁ = 0,
                    D[j] = u[j]*D[j-1] + v[j]*D[j-2] + ũ[j]*C[j-1],   for j ≥ 2.
                Returns:
                    C, D : NumPy arrays of length N+1.
                """
                N = self.N
                C = np.zeros(N + 1)
                D = np.zeros(N + 1)
                C[0] = 0.0
                C[1] = self.ut[1]
                D[0] = 0.0
                D[1] = 0.0
                for j in range(2, N + 1):
                        C[j] = self.u[j] * C[j - 1] + self.v[j] * C[j - 2] + self.ut[j] + self.gt[j]
                        D[j] = self.u[j] * D[j - 1] + self.v[j] * D[j - 2] + self.ut[j] * C[j - 1]
                return C, D

        def _compute_hat(self, seq, j):
                """
                Compute the weighted (hat) value for index j from sequence seq:
                    hat_seq = sum_{l=0}^{j-1} theta*(1-theta)^l * seq[j-l]
                """
                hat_val = 0.0
                for l in range(j):
                        hat_val += self.theta * (1 - self.theta) ** l * seq[j - l]
                return hat_val

        def _compute_x1_x2(self):
                """
                Compute the parameters x₁ and x₂ ensuring second order accuracy.
                Using:
                    x₁ = (0.5·hat_C(N) - hat_D(N)) / (hat_C(N)*hat_D(N-1) - hat_C(N-1)*hat_D(N)),
                    x₂ = (1 - x₁·hat_C(N-1)) / hat_C(N),
                where N = s+shat.
                """
                C, D = self._compute_cd_sequences()
                # N = self.N
                # hatC_Nm1 = self._compute_hat(C, N - 1)
                # hatC_N = self._compute_hat(C, N)
                # hatD_Nm1 = self._compute_hat(D, N - 1)
                # hatD_N = self._compute_hat(D, N)
                # numerator = 0.5 * hatC_N - hatD_N
                # denominator = hatC_N * hatD_Nm1 - hatC_Nm1 * hatD_N
                # x1 = numerator / denominator
                # x2 = (1 - x1 * hatC_Nm1) / hatC_N

                N = self.N
                hatC_Nm1 = self._compute_hat(C, N - 1)
                hatC_N = self._compute_hat(C, N)
                numerator = (1-hatC_N)
                denominator = hatC_Nm1 - hatC_N
                x1 = numerator / denominator
                x2 = 1-x1

                return x1, x2

        def step(self, t, y, h):
                """
                Take one time step from (t, y) with step size h using the second order IMPRKC method.

                The method computes stage values:
                  K₀ = y,      ˆK₀ = y,
                  K₁ = y + ũ₁·h·F₀,  ˆK₁ = α K₁ + (1-α)ˆK₀,
                  for j = 2,..., N:
                     Kⱼ = uⱼ Kⱼ₋₁ + vⱼ Kⱼ₋₂ + (1 - uⱼ - vⱼ) y + ũⱼ·h·Fⱼ₋₁ + γ̃ⱼ·h·F₀,
                     ˆKⱼ = α Kⱼ + (1-α)ˆKⱼ₋₁,
                  and then
                     yₙ₊₁ = (1 - x₁ - x₂) y + x₁ˆK_{N-1} + x₂ˆK_N.

                Here, F₀ = f(t, y) and Fⱼ = f(t + cⱼ·h, Kⱼ) for j>=1.
                """
                N = self.N
                shat = self.shat
                alpha = 1.0 / (shat + 1)
                beta = 1 - alpha

                # Stage storage: K[j] and ˆK[j]
                K = [None] * (N + 1)
                Khat = [None] * (N + 1)

                # Stage 0.
                K[0] = y.copy()
                Khat[0] = y.copy()
                F0 = self.f(t, y)

                # Stage 1.
                K[1] = y + self.ut[1] * h * F0
                Khat[1] = alpha * K[1] + beta * Khat[0]

                # For stages j = 2,..., N.
                for j in range(2, N + 1):
                        # Evaluate f at stage: use t + c[j-1]*h and K[j-1]
                        t_stage = t + self.c[j - 1] * h
                        Fjm1 = self.f(t_stage, K[j - 1])
                        K[j] = (self.u[j] * K[j - 1] + self.v[j] * K[j - 2] +
                                (1 - self.u[j] - self.v[j]) * y +
                                self.ut[j] * h * Fjm1 +
                                self.gt[j] * h * F0)
                        Khat[j] = alpha * K[j] + beta * Khat[j - 1]

                # Compute parameters x₁ and x₂.
                x1, x2 = self._compute_x1_x2()
                print(f'{x1 = }, {x2 = }')
                # Final update.
                y_next = (1 - x1 - x2) * y + x1 * Khat[N - 1] + x2 * Khat[N]
                t_next = t + h
                return t_next, y_next

        def solve(self, t0, T, y0, h):
                """
                Solve the ODE from time t0 to T with initial value y0 using fixed step size h.
                Returns:
                    t_vals : 1D NumPy array of time points.
                    y_vals : 2D NumPy array; each row is the solution at a time point.
                """
                t = t0
                y = y0.copy()
                t_vals = [t]
                y_vals = [y.copy()]
                while t < T:
                        t, y = self.step(t, y, h)
                        t_vals.append(t)
                        y_vals.append(y.copy())
                return np.array(t_vals), np.array(y_vals)


# ---------------------
# Simple Test Case
# ---------------------
if __name__ == "__main__":
        from matplotlib import pyplot as plt

        # Test ODE: y' = -λ y, with λ = 1, so the exact solution is y(t)=y0*exp(-t)
        coef = -50
        def f(t, y):
                return coef * y


        # Set initial condition and time interval.
        y0 = np.array([1.0])
        t0 = 0.0
        T = 1
        h = 0.01

        # Choose classical stage number s and shat (extra stage)
        s = 9
        shat = 1

        # Create the solver instance.
        solver = IMPRKCSolver(f, s, shat, eta=2 / 13)

        # Solve the ODE.
        t_vals, y_vals = solver.solve(t0, T, y0, h)

        # Print final value and compare to exact solution.
        y_exact = y0 * np.exp(coef * t_vals)

        print("Final t =", t_vals[-1])
        print("Computed y =", y_vals[-1])
        print("Exact y    =", y_exact[-1])

        plt.plot(t_vals, y_vals)
        plt.plot(t_vals, y_exact.squeeze())
        plt.show()
