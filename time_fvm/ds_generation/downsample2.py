import math
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, LinearRing

# ----------------------------
# 1) Convert edges -> sampled points
# ----------------------------
def edges_to_points(vertices: np.ndarray,
                    edges: np.ndarray,
                    step: float,
                    include_endpoints: bool = True,
                    dedupe_tol: float = 1e-9) -> np.ndarray:
    """
    Sample points along a set of edges.

    vertices: (N,2)
    edges:    (M,2) integer indices into vertices
    step:     approximate spacing along edges

    Returns: (K,2) sampled points along all edges (deduped).
    """
    verts = np.asarray(vertices, float)
    eds = np.asarray(edges, int)

    pts = []
    for i, j in eds:
        a = verts[i]
        b = verts[j]
        seg = b - a
        L = float(np.linalg.norm(seg))
        if L == 0.0:
            continue

        n = max(1, int(math.ceil(L / step)))
        if include_endpoints:
            ts = np.linspace(0.0, 1.0, n + 1)
        else:
            ts = np.linspace(0.0, 1.0, n, endpoint=False)

        for t in ts:
            pts.append(a + t * seg)

    if not pts:
        return np.empty((0, 2), dtype=float)

    pts = np.asarray(pts, float)

    # Deduplicate with a tolerance by quantizing
    scale = 1.0 / dedupe_tol
    keys = np.round(pts * scale).astype(np.int64)
    _, unique_idx = np.unique(keys, axis=0, return_index=True)

    return pts[np.sort(unique_idx)]


# ----------------------------
# 2) Variable-radius Poisson disk sampling inside polygon with boundary clearance
# ----------------------------
def poisson_disk_variable_r(polygon: Polygon,
                            r_of, r_min: float, r_max: float,
                            clearance_of=None,
                            k: int = 30, seed: int = 0) -> np.ndarray:
    """
    Bridson-style Poisson disk sampling with variable radius r(p).
    Enforces separation: ||p-q|| >= 0.5*(r(p)+r(q))
    Optionally enforces boundary clearance: dist(p, polygon.boundary) >= clearance_of(p)
    """
    rng = np.random.default_rng(seed)
    minx, miny, maxx, maxy = polygon.bounds

    # Grid based on global r_min (safe)
    cell = r_min / math.sqrt(2)
    gw = int(math.ceil((maxx - minx) / cell))
    gh = int(math.ceil((maxy - miny) / cell))
    grid = -np.ones((gh, gw), dtype=int)

    pts = []
    active = []

    def grid_coords(p):
        gx = int((p[0] - minx) / cell)
        gy = int((p[1] - miny) / cell)
        return gx, gy

    def min_sep(p, q):
        return 0.5 * (r_of(p) + r_of(q))

    def far_enough_from_boundary(p):
        if clearance_of is None:
            return True
        return Point(p).distance(polygon.boundary) >= float(clearance_of(p))

    def too_close(p):
        rp = float(r_of(p))
        # Conservative scan radius to not miss conflicts
        scan_R = 0.5 * (rp + r_max)
        n = int(math.ceil(scan_R / cell))

        gx, gy = grid_coords(p)
        x0, x1 = max(gx - n, 0), min(gx + n + 1, gw)
        y0, y1 = max(gy - n, 0), min(gy + n + 1, gh)

        for yy in range(y0, y1):
            for xx in range(x0, x1):
                j = grid[yy, xx]
                if j != -1:
                    q = pts[j]
                    dx = p[0] - q[0]
                    dy = p[1] - q[1]
                    if dx*dx + dy*dy < (min_sep(p, q) ** 2):
                        return True
        return False

    # initial point
    for _ in range(20000):
        p0 = (rng.uniform(minx, maxx), rng.uniform(miny, maxy))
        if polygon.contains(Point(p0)) and far_enough_from_boundary(p0):
            pts.append(p0)
            active.append(0)
            gx, gy = grid_coords(p0)
            if 0 <= gx < gw and 0 <= gy < gh:
                grid[gy, gx] = 0
            break
    else:
        return np.empty((0, 2), dtype=float)

    # main loop
    while active:
        a_idx = active[rng.integers(0, len(active))]
        base = pts[a_idx]
        r_base = float(r_of(base))
        found = False

        for _ in range(k):
            ang = rng.uniform(0.0, 2.0 * math.pi)
            rad = rng.uniform(r_base, 2.0 * r_base)
            p = (base[0] + rad * math.cos(ang), base[1] + rad * math.sin(ang))

            if not (minx <= p[0] <= maxx and miny <= p[1] <= maxy):
                continue
            if not polygon.contains(Point(p)):       # respects hole(s)
                continue
            if not far_enough_from_boundary(p):      # boundary clearance
                continue
            if too_close(p):                         # variable-radius separation
                continue

            pts.append(p)
            new_i = len(pts) - 1
            active.append(new_i)
            gx, gy = grid_coords(p)
            if 0 <= gx < gw and 0 <= gy < gh:
                grid[gy, gx] = new_i
            found = True
            break

        if not found:
            active.remove(a_idx)

    return np.asarray(pts, dtype=float)


# ----------------------------
# 3) Example: square with circular hole from vertices+edges, then sample
# ----------------------------
def build_square_and_circle_edges(square_half=1.0, hole_r=0.4, circle_n=96):
    # Square vertices (ordered loop)
    s = square_half
    V_sq = np.array([[-s, -s],
                     [ s, -s],
                     [ s,  s],
                     [-s,  s]], float)
    E_sq = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], int)

    # Circle vertices (ordered loop)
    ang = np.linspace(0.0, 2.0 * np.pi, circle_n, endpoint=False)
    V_c = np.column_stack([hole_r * np.cos(ang), hole_r * np.sin(ang)])
    E_c = np.column_stack([np.arange(circle_n), (np.arange(circle_n) + 1) % circle_n]).astype(int)

    # Combine into one vertex array; shift circle edge indices
    offset = len(V_sq)
    V = np.vstack([V_sq, V_c])
    E_circle = E_c + offset
    E_square = E_sq
    return V, E_square, E_circle, V_sq, V_c

def smoothstep(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)

if __name__ == "__main__":
    V, E_square, E_circle, V_sq_loop, V_circle_loop = build_square_and_circle_edges(
        square_half=1.0, hole_r=0.4, circle_n=96
    )

    # Build shapely polygon from the ordered loops (holes supported)
    poly = Polygon(shell=V_sq_loop.tolist(), holes=[V_circle_loop.tolist()])

    # Turn edge lists into boundary points (useful for plotting / later meshing constraints)
    boundary_step = 0.05
    bpts = np.vstack([
        edges_to_points(V, E_square, step=boundary_step),
        edges_to_points(V, E_circle, step=boundary_step),
    ])

    # Variable spacing field: smaller r near the hole boundary -> denser
    hole_ring = LinearRing(poly.interiors[0])
    r_near, r_far, influence = 0.05, 0.12, 0.35

    def r_of(p):
        d = Point(p).distance(poly.boundary)
        t = smoothstep(d / influence)
        return r_near + (r_far - r_near) * t

    # Boundary clearance: keep points away from *any* boundary (outer + hole)
    def clearance_of(p):
        return 1 * r_of(p)  # tune: 0.25–1.0 * r_of(p)

    pts = poisson_disk_variable_r(
        poly,
        r_of=r_of, r_min=r_near, r_max=r_far,
        clearance_of=clearance_of,
        k=30, seed=7
    )

    # Plot
    fig, ax = plt.subplots()
    ax.scatter(bpts[:, 0], bpts[:, 1], s=6, label="boundary points (from edges)")
    ax.scatter(pts[:, 0], pts[:, 1], s=10, label="Poisson samples")

    # Draw polygon boundaries
    x, y = poly.exterior.xy
    ax.plot(x, y)
    for ring in poly.interiors:
        xi, yi = ring.xy
        ax.plot(xi, yi)

    ax.set_aspect("equal", "box")
    ax.legend()
    ax.set_title(f"Variable-radius Poisson samples with boundary clearance (n={len(pts)})")
    plt.show()
