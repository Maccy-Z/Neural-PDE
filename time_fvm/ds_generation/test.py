import numpy as np
import matplotlib.pyplot as plt

# ---- dependencies (KDTree) ----
try:
    from scipy.spatial import cKDTree
except Exception as e:
    cKDTree = None
    print("SciPy not available; falling back to brute-force distances (slower).", e)

def _nn_distance(points):
    """Nearest-neighbor distance for each point."""
    points = np.asarray(points, float)
    n = len(points)
    if n < 2:
        return np.array([])
    if cKDTree is not None:
        tree = cKDTree(points)
        d, _ = tree.query(points, k=2)  # first is self (0), second is NN
        return d[:, 1]
    # brute force fallback
    dists = np.sqrt(((points[None, :, :] - points[:, None, :]) ** 2).sum(axis=-1))
    np.fill_diagonal(dists, np.inf)
    return dists.min(axis=1)

# ---- sampler under test ----
def weighted_poisson_points(P, w, n_samples, r0, seed=None, max_trials=300000):
    """
    Variable-radius weighted Poisson-disk sampling in 2D.

    P: (N,2) point coordinates
    w: (N,) nonnegative weights (higher => denser sampling allowed)
    n_samples: number of points to select
    r0: base radius (scaled by weights)
    r(i) = r0 / sqrt(w_i / mean(w))  -> higher weight => smaller exclusion radius
    """
    rng = np.random.default_rng(seed)

    P = np.asarray(P, float)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError("P must have shape (N, 2).")

    w = np.asarray(w, float)
    w = np.clip(w, 1e-12, None)  # avoid divide-by-zero / sqrt issues
    p = w / w.sum()
    w_mean = w.mean()

    selected = []
    selected_xy = []
    tree = None

    for _ in range(max_trials):
        if len(selected) >= n_samples:
            break

        i = rng.choice(len(P), p=p)
        x = P[i]

        ri = r0 / np.sqrt(w[i] / w_mean)

        if tree is None:
            selected.append(i)
            selected_xy.append(x)
            tree = cKDTree(np.array(selected_xy))
            continue

        d, _ = tree.query(x, k=1)
        if d >= ri:
            selected.append(i)
            selected_xy.append(x)
            tree = cKDTree(np.array(selected_xy))

    return np.array(selected, dtype=int)



# ---- baseline for comparison: plain weighted random (with replacement avoided) ----
def weighted_random_vertices(V, w, n_samples, seed=None):
    rng = np.random.default_rng(seed)
    w = np.asarray(w, float)
    w = np.clip(w, 0, None)
    if w.sum() == 0:
        raise ValueError("All weights are zero.")
    p = w / w.sum()
    # sample without replacement by repeated draws (works for large N, modest n)
    chosen = set()
    while len(chosen) < n_samples:
        i = int(rng.choice(len(V), p=p))
        chosen.add(i)
    return np.fromiter(chosen, dtype=int)

if __name__ == "__main__":
    # ---------------------------
    # Create a synthetic "mesh" vertex set (2D grid embedded in 3D)
    # ---------------------------
    seed = 7
    rng = np.random.default_rng(seed)

    nx, ny = 100, 100
    xs = np.linspace(-1, 1, nx)
    ys = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    V2 = np.column_stack([X.ravel(), Y.ravel()])
    V = np.column_stack([V2, np.zeros(len(V2))])  # (N,3), z=0

    # Weight field: mixture of two Gaussian bumps + a small floor
    def gaussian(xy, mu, sigma):
        d2 = ((xy - mu) ** 2).sum(axis=1)
        return np.exp(-0.5 * d2 / (sigma**2))

    w = (
        1.0 * gaussian(V2, np.array([0.45, 0.25]), 0.20)
        + 0.7 * gaussian(V2, np.array([-0.35, -0.15]), 0.25)
        + 0.05
    )

    # Sample settings
    n_samples = 400
    r0 = 0.06  # base radius; increase for more uniformity, decrease if sampler can't fill n_samples
    seed_poisson = 123
    seed_random = 123

    idx_poisson = weighted_poisson_points(V[:, :2], w, n_samples=n_samples, r0=r0, seed=seed_poisson)
    idx_random  = weighted_random_vertices(V, w, n_samples=n_samples, seed=seed_random)

    P_poisson = V2[idx_poisson]
    P_random  = V2[idx_random]

    # Diagnostics
    nn_poisson = _nn_distance(P_poisson)
    nn_random  = _nn_distance(P_random)

    # ---------------------------
    # Plots
    # ---------------------------

    # 1) Weight field (as an image)
    plt.figure()
    plt.title("Weight field w(x, y)")
    plt.imshow(w.reshape(ny, nx), origin="lower", extent=[-1, 1, -1, 1], aspect="equal")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    # 2) Samples overlaid on the domain (random)
    plt.figure()
    plt.title("Weighted random sampling (baseline)")
    plt.scatter(V2[:, 0], V2[:, 1], s=1)      # all vertices
    plt.scatter(P_random[:, 0], P_random[:, 1], s=15)  # selected
    plt.gca().set_aspect("equal", "box")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    # 3) Samples overlaid on the domain (variable-radius Poisson)
    plt.figure()
    plt.title("Weighted Poisson (variable radius)")
    plt.scatter(V2[:, 0], V2[:, 1], s=1)
    plt.scatter(P_poisson[:, 0], P_poisson[:, 1], s=15)
    plt.gca().set_aspect("equal", "box")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    # 4) Nearest-neighbor distance histograms (coverage proxy)
    plt.figure()
    plt.title("Nearest-neighbor distances (bigger => less clumping)")
    plt.hist(nn_random, bins=30, alpha=0.7, label="weighted random")
    plt.hist(nn_poisson, bins=30, alpha=0.7, label="weighted Poisson var-r")
    plt.xlabel("nearest-neighbor distance")
    plt.ylabel("count")
    plt.legend()
    plt.show()

    # 5) Quick summary
    print(f"Random:  mean NN dist = {nn_random.mean():.4f},  min NN dist = {nn_random.min():.4f}")
    print(f"Poisson: mean NN dist = {nn_poisson.mean():.4f},  min NN dist = {nn_poisson.min():.4f}")
    print(f"Poisson filled {len(idx_poisson)}/{n_samples} samples (increase max_trials or reduce r0 if short).")
