import torch


def rkc_coefficients(s: int, epsilon: float = 0.05, device=None, dtype=torch.float32):
        """
        Generate coefficients for a second-order Runge-Kutta-Chebyshev (RKC) method.

        Parameters:
            s (int): Number of stages (must be >= 2).
            epsilon (float): Damping parameter (typically a small positive number, e.g. 0.05).
            device: torch.device (defaults to CPU).
            dtype: torch data type (default: torch.float32).

        Returns:
            dict: Dictionary with the following keys:
                - "a": Tensor of a_j coefficients, j = 0,..., s.
                - "mu": Tensor of mu_j coefficients (for j>=2; mu[0] and mu[1] are not used).
                - "nu": Tensor of nu_j coefficients (for j>=2; nu[0] and nu[1] are not used).
                - "kappa": Tensor of kappa_j coefficients (for j>=1; kappa[0] is not used).

        The function uses the recurrences for Chebyshev polynomials:
            T_0(x) = 1,   T_1(x) = x,   T_j(x) = 2*x*T_{j-1}(x) - T_{j-2}(x),
        and for their derivatives:
            T'_0(x) = 0,  T'_1(x) = 1,  T'_j(x) = 2*T_{j-1}(x) + 2*x*T'_{j-1}(x) - T'_{j-2}(x),
        and the second derivative:
            T''_0(x) = 0, T''_1(x) = 0, T''_j(x) = 4*T'_{j-1}(x) + 2*x*T''_{j-1}(x) - T''_{j-2}(x).

        The coefficients are then computed as follows:

            b[j] = T''_j(w0) / (T'_j(w0))^2,  for j >= 2  (with b[0] and b[1] set equal to b[2])
            a[j] = 1 - b[j]*T_j(w0)      for j = 0,...,s

            For j = 2:
                mu[2] = 2*w0,  nu[2] = -1.
            For j >= 3:
                mu[j] = 2*b[j]*w0 / b[j-1],
                nu[j] = -b[j] / b[j-2].

            Let w1 = T'_s(w0)/T''_s(w0) (assumed nonzero), then
                kappa[2] = 2*w1,
                for j >= 3: kappa[j] = 2*b[j]*w1 / b[j-1].
            For the first stage, a common choice is kappa[1] = 1.

        (This is one possible formulation; in practice, minor variations exist.)
        """
        if s < 2:
                raise ValueError("Number of stages s must be at least 2.")

        device = device or torch.device('cpu')

        # Set w0 = 1 + epsilon / s^2
        w0 = 1.0 + epsilon / (s ** 2)
        # Allocate tensors for Chebyshev polynomials T, Tprime, Tpp for j = 0,..., s.
        T = torch.zeros(s + 1, dtype=dtype, device=device)
        Tprime = torch.zeros(s + 1, dtype=dtype, device=device)
        Tpp = torch.zeros(s + 1, dtype=dtype, device=device)

        # Initialization
        T[0] = 1.0
        T[1] = w0
        Tprime[0] = 0.0
        Tprime[1] = 1.0
        Tpp[0] = 0.0
        Tpp[1] = 0.0

        # Compute Chebyshev polynomials and derivatives for j = 2,..., s
        for j in range(2, s + 1):
                T[j] = 2.0 * w0 * T[j - 1] - T[j - 2]
                Tprime[j] = 2.0 * T[j - 1] + 2.0 * w0 * Tprime[j - 1] - Tprime[j - 2]
                Tpp[j] = 4.0 * Tprime[j - 1] + 2.0 * w0 * Tpp[j - 1] - Tpp[j - 2]

        # Allocate coefficient tensors (indices 0..s; not all entries are used)
        b = torch.zeros(s + 1, dtype=dtype, device=device)
        a = torch.zeros(s + 1, dtype=dtype, device=device)
        c = torch.zeros(s + 1, dtype=dtype, device=device)
        mu = torch.zeros(s + 1, dtype=dtype, device=device)
        nu = torch.zeros(s + 1, dtype=dtype, device=device)
        kappa = torch.zeros(s + 1, dtype=dtype, device=device)

        # Compute b_j for j >= 2: b[j] = Tpp[j] / (Tprime[j]^2)
        for j in range(2, s + 1):
                b[j] = Tpp[j] / (Tprime[j] ** 2)

        # Set b[0] and b[1] equal to b[2] to avoid division by zero issues later.
        b[0] = b[2]
        b[1] = b[2]

        # Compute a_j = 1 - b[j]*T[j] for j = 0,..., s
        for j in range(0, s + 1):
                a[j] = 1.0 - b[j] * T[j]

        # Compute mu and nu coefficients.
        # For j = 2:
        mu[2] = 2.0 * w0
        nu[2] = -1.0
        # For j = 3,..., s:
        for j in range(3, s + 1):
                mu[j] = 2.0 * b[j] * w0 / b[j - 1]
                nu[j] = - b[j] / b[j - 2]

        # Compute w1 = Tprime[s] / Tpp[s] (assume Tpp[s] is nonzero)
        if Tpp[s] == 0:
                raise ValueError("Tpp[s] is zero; cannot compute w1.")
        w1 = Tprime[s] / Tpp[s]

        # Compute kappa coefficients.
        # A common choice is to set kappa[1] = 1.
        kappa[1] = b[1] * w1
        # For j = 2:
        kappa[2] = 2.0 * w1
        # For j = 3,..., s:
        for j in range(3, s + 1):
                kappa[j] = 2.0 * b[j] * w1 / b[j - 1]

        c[0] = 0.0
        c[1] = b[1] * w1
        for j in range(2, s + 1):
                c[j] = (T[j] - 1) / (T[-1] - 1)
        print(c)
        # Return the computed coefficients.
        return {
                "a": a,  # a_j, j = 0,..., s
                "mu": mu,  # mu_j for j>=2 (mu[0] and mu[1] are not used)
                "nu": nu,  # nu_j for j>=2 (nu[0] and nu[1] are not used)
                "kappa": kappa,  # kappa_j for j>=1 (kappa[0] is not used)
                "c": c,
                # Also returning these for debugging/inspection if needed:
                "_": None,
                "T": T,
                "Tprime": Tprime,
                "Tpp": Tpp,
                "w0": w0,
                "b": b,
                "w1": w1
        }




def rkc_step(f, t, U, h, s, coeffs, device=None, dtype=torch.float32):
        """
        Perform one integration step using the second-order Runge-Kutta-Chebyshev (RKC) method.

        Parameters:
            f (callable): Function f(t, U) returning dU/dt.
            t (float or torch scalar): Current time.
            U (torch.Tensor): Current solution (can be vector or tensor).
            h (float): Time step size.
            s (int): Number of stages (>= 2).
            device: torch.device (defaults to CPU if None).
            dtype: torch data type (default: torch.float32).

        Returns:
            torch.Tensor: The solution at time t+h (i.e. U_{n+1}).

        The recurrence used is:

            U_0 = U.
            f_0 = f(t, U_0).
            U_1 = U_0 + kappa[1]*h*f_0.
            For j = 2,..., s:
                U_j = U_0 + mu[j]*(U_{j-1} - U_0)
                            + nu[j]*(U_{j-2} - U_0)
                            + kappa[j]*h*( f(t + c[j-1]*h, U_{j-1}) - a[j-1]*f_0 ).

        Here, the coefficients (a, mu, nu, kappa) are computed from Chebyshev recurrences,
        and the nodes c (with c[0]=0 and c[s]=1) are taken as
            c[j] = 0.5*(1 - cos(pi*j/s)),
        which are the Chebyshev nodes scaled to [0,1].
        """
        device = device or torch.device("cpu")

        # Get RKC coefficients from our helper function.
        a = coeffs["a"]  # shape: (s+1,)
        mu = coeffs["mu"]  # shape: (s+1,), mu[0] and mu[1] not used
        nu = coeffs["nu"]  # shape: (s+1,), nu[0] and nu[1] not used
        kappa = coeffs["kappa"]  # shape: (s+1,), kappa[0] is not used

        # Compute Chebyshev nodes on [0,1] for stage time shifts.
        # Here we use: c[j] = 0.5*(1 - cos(pi * j/s)), for j = 0,..., s.
        j_arr = torch.arange(0, s + 1, dtype=dtype, device=device)
        # c = 0.5 * (1 - torch.cos(torch.pi * j_arr / s))
        # print(f'{c = }')
        c = coeffs["c"]

        # Stage 0: starting value.
        U0 = U
        f0 = f(t, U0)

        # Stage 1: an Euler-like step.
        U1 = U0 + kappa[1] * h * f0
        # Evaluate f at t + c[1]*h, U1.

        # Initialize previous stage values.
        U_prev2 = U0  # corresponds to U_{j-2}
        U_prev = U1  # corresponds to U_{j-1}

        # Loop over stages j = 2,..., s.
        for j in range(2, s + 1):
                # For the recurrence, evaluate f at shifted time for the previous stage.
                t_eval = t + c[j - 1] * h
                f_eval = f(t_eval, U_prev)
                # RKC recurrence:
                Uj = U0 + mu[j] * (U_prev - U0) + nu[j] * (U_prev2 - U0) \
                     + kappa[j] * h * (f_eval - a[j - 1] * f0)

                # Update previous stage values for next iteration.
                U_prev2 = U_prev
                U_prev = Uj
        y_next = Uj

        #h = 0.1
        # Assume y and f are defined

        # # Precomputed coefficients for s=5 (would be computed as above in real code)
        # kappa1 = 0.0313
        # mu2 = 2.004
        # nu2 = -1.0
        # kappa2 = 0.2519
        # a1 = 0.00784#0.7505
        # mu3 = 2.364
        # nu3 = -1.1797
        # kappa3 = 0.2972
        # a2 = 0.7490
        # mu4 = 2.0999
        # nu4 = -1.2361
        # kappa4 = 0.2639
        # a3 = 0.70095
        # mu5 = 2.0351
        # nu5 = -1.0641
        # kappa5 = 0.2558
        # a4 = 0.68230
        # # Stage 0
        # Y0 = U
        # F0 = f(t, Y0)
        # # Stage 1
        # Y1 = Y0 + kappa1 * h * F0
        # F1 = f(t + 0 * h, Y1)  # c1 would be kappa1 in this formulation
        # # Stage 2
        # Y2 = Y0 + mu2 * (Y1 - Y0) + nu2 * (Y0 - Y0) + kappa2 * h * (F1 - a1 * F0)
        # F2 = f(t + 0 * h, Y2)
        # # Stage 3
        # Y3 = Y0 + mu3 * (Y2 - Y0) + nu3 * (Y1 - Y0) + kappa3 * h * (F2 - a2 * F0)
        # F3 = f(t + 0 * h, Y3)
        # # Stage 4
        # Y4 = Y0 + mu4 * (Y3 - Y0) + nu4 * (Y2 - Y0) + kappa4 * h * (F3 - a3 * F0)
        # F4 = f(t + 0 * h, Y4)
        # # Stage 5
        # Y5 = Y0 + mu5 * (Y4 - Y0) + nu5 * (Y3 - Y0) + kappa5 * h * (F4 - a4 * F0)
        # # Y5 is the result after one step of size h
        # y_next = Y5

        # At the end of the loop, U_prev is U_s, the solution at t+h.
        return y_next


# Example usage:
if __name__ == "__main__":
        from matplotlib import pyplot as plt

        # Define a sample ODE: dU/dt = -lambda * U, for a stiff decay.
        lambda_val = -100.0


        def f(t, U):
                return lambda_val * U


        # Initial condition, time, and step size.
        U0 = torch.tensor([1.0], dtype=torch.float32)
        t0 = 0.0
        h = 0.015  # step size; with RKC, h can be larger than the explicit Euler limit for stiff problems
        s = 5  # number of stages


        coeffs = rkc_coefficients(s, epsilon=2/13)

        # Take one RKC step.
        Us = []
        for i in range(100):
                t = t0 + i * h
                Us.append(U0)
                U0 = rkc_step(f, t, U0, h, s, coeffs=coeffs)

                print(t)

        plt.plot(Us)

        U0 = torch.tensor([1.0], dtype=torch.float32)
        ts = torch.range(0, 100) * h
        true_Us = torch.exp(lambda_val * ts)
        plt.plot(true_Us, 'r--')

        plt.show()
