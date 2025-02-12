import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Parameters and domain
# -----------------------
L = 1.0  # domain length
N = 101  # number of grid points
dx = L / (N - 1)
x = np.linspace(0, L, N)
print(f'{dx = }')
T_max = 0.5  # final time
CFL = 0.1  # CFL number for stability (wave speed is 1)
dt = CFL * dx  # time step size
nsteps = int(T_max / dt)

# -----------------------------------
# Initial conditions
# -----------------------------------
# Here we choose a nontrivial initial condition.
# We set u = 0 everywhere, and p is chosen to be a step function:
# p = 0 for x < 0.5 and p = 1 for x > 0.5.
u = np.zeros(N)
p = np.zeros(N)
p = np.random.randn(N) * 0.01 + 1

# Enforce boundary conditions on the initial data:
u[0] = 0.0  # left: u = 0
p[0] = p[1]  # left: p_x = 0  (approximate by p[0] = p[1])
p[-1] = 1.0  # right: p = 1
u[-1] = u[-2]  # right: u_x = 0  (approximate by u[-1] = u[-2])

# -----------------------------------
# Time stepping loop (Euler explicit)
# -----------------------------------
plt.figure()
for n in range(nsteps):
    # Make copies to update simultaneously
    u_new = u.copy()
    p_new = p.copy()

    # Update interior points using central differences
    u_new[1:-1] = u[1:-1] - dt / (2 * dx) * (p[2:] - p[:-2])
    p_new[1:-1] = p[1:-1] - dt / (2 * dx) * (u[2:] - u[:-2])

    # Apply boundary conditions:
    # Left boundary (x=0):
    u_new[0] = 0.0  # u = 0
    p_new[0] = p_new[1]  # p_x = 0 --> p[0] = p[1]

    # Right boundary (x=L):
    p_new[-1] = 1.0  # p = 1
    u_new[-1] = u_new[-2]  # u_x = 0 --> u[-1] = u[-2]

    # Update the solution arrays
    u, p = u_new, p_new

    # Optionally, plot the solution every few time steps:
    if n % 10 == 0:
        plt.clf()
        plt.plot(x, u, label='u')
        plt.plot(x, p, label='p')
        plt.xlabel('x')
        plt.title(f't = {n * dt:.3f}')
        plt.legend()
        plt.pause(0.01)

plt.show()
