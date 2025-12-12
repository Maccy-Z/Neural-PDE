import torch
import numpy as np
import os
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
import triangle as tr
from collections import defaultdict
from time_fvm.ds_generation.saving import plot_interp


def load_step(file_path):
    """
    Loads a single time step file and returns the un-normalized primative values.
    """
    data = np.load(file_path)
    prim_mean = data['prim_mean']
    prim_std = data['prim_std']
    cell_primatives_scaled = data['cell_primatives'].astype(np.float32)
    bc_primatives_scaled = data['bc_primatives'].astype(np.float32)

    cell_primatives = cell_primatives_scaled * prim_std + prim_mean
    bc_primatives = bc_primatives_scaled * prim_std + prim_mean

    return data['t'], torch.from_numpy(cell_primatives).float(), torch.from_numpy(bc_primatives).float()


class AveragedGraphs:
    sum_cells: torch.Tensor
    sum_bc: torch.Tensor
    def __init__(self):
        self.count = 0

        self.sum_cells = 0
        self.sum_bc = 0


    def add_step(self, cell_values: torch.Tensor, bc_values: torch.Tensor):
        """ Add a new graph to the average.
            shape = [N_cells, N_comp=4]
        """
        self.sum_cells += cell_values.double()
        self.sum_bc += bc_values.double()
        self.count += 1

    def get_average(self):
        mean_cells = self.sum_cells / self.count
        mean_bc = self.sum_bc / self.count
        return mean_cells.float(), mean_bc.float()


def adaptive_remesh_preserve_holes(
    points,
    triangles,
    u_cells,
    n_vertices_new,
    *,
    p_power=1.0,
    floor=0.1,
    seed=None,
):
    """
    Adaptive remeshing for a 2D triangular unstructured mesh with holes.
    Boundaries (outer and holes) are inferred automatically from triangle
    connectivity and preserved using a constrained Delaunay triangulation.

    Parameters
    ----------
    points : (N, 2) array
        Original vertex coordinates.
    triangles : (M, 3) int array
        Original triangle connectivity (indices into `points`).
    u_cells : (M,) array
        Scalar values at cell centroids (one per triangle).
    n_vertices_new : int
        Number of *interior* sample points to add (boundary vertices are reused).
    p_power : float, optional
        Exponent applied to normalized gradient when building the density.
        1.0 = linear; >1 emphasizes sharp regions more.
    floor : float in (0,1], optional
        Minimum relative density so smooth regions still get some points.
    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    new_points : (N_new, 2) array
        Vertex coordinates of the new mesh (boundary + interior).
    new_triangles : (K, 3) int array
        Triangle connectivity (indices into `new_points`).
    u_nodes_new : (N_new,) array
        Interpolated scalar values at the new mesh vertices.
    u_cells_new : (K,) array
        Interpolated scalar values at the new mesh cell centroids.

    Notes
    -----
    - Holes are detected as inner boundary loops (smaller polygons) using
      boundary edges (edges that belong to only one triangle).
    - For each inner loop, a hole point is placed at the polygon centroid.
      This assumes reasonably “nice” hole shapes so that centroid lies inside.
    - NaNs from LinearNDInterpolator (outside convex hull) are fixed by
      falling back to a nearest-neighbour interpolation.
    """
    rng = np.random.default_rng(seed)

    points = np.asarray(points, float)
    triangles = np.asarray(triangles, int)
    u_cells = np.asarray(u_cells, float)

    # ---------------------------------------------------------
    # 1) Cell centroids & areas
    # ---------------------------------------------------------
    cell_pts = points[triangles].mean(axis=1)  # (M, 2)

    def tri_area(tri_pts):
        return 0.5 * np.abs(
            (tri_pts[:, 1, 0] - tri_pts[:, 0, 0]) * (tri_pts[:, 2, 1] - tri_pts[:, 0, 1])
            - (tri_pts[:, 2, 0] - tri_pts[:, 0, 0]) * (tri_pts[:, 1, 1] - tri_pts[:, 0, 1])
        )

    cell_areas = tri_area(points[triangles])  # (M,)
    M = triangles.shape[0]

    # ---------------------------------------------------------
    # 2) Build edge -> cells map and cell neighbours
    # ---------------------------------------------------------
    edge_to_cells = defaultdict(list)
    for ci, tri in enumerate(triangles):
        a, b, c = tri
        for e in ((a, b), (b, c), (c, a)):
            e_sorted = (min(e), max(e))
            edge_to_cells[e_sorted].append(ci)

    neighbors = [[] for _ in range(M)]
    for cells in edge_to_cells.values():
        if len(cells) == 2:
            c0, c1 = cells
            neighbors[c0].append(c1)
            neighbors[c1].append(c0)

    neighbors = [np.array(nb, int) for nb in neighbors]

    # ---------------------------------------------------------
    # 3) Approximate |∇u| per cell using neighbour differences
    # ---------------------------------------------------------
    grad_mag = np.zeros(M)
    for i in range(M):
        nb = neighbors[i]
        if nb.size == 0:
            grad_mag[i] = 0.0
            continue
        du = u_cells[nb] - u_cells[i]
        dx = cell_pts[nb] - cell_pts[i]
        dist = np.linalg.norm(dx, axis=1) + 1e-12
        grad_mag[i] = np.max(np.abs(du) / dist)

    # ---------------------------------------------------------
    # 4) Build sampling density from gradient
    # ---------------------------------------------------------
    g_norm = grad_mag / (grad_mag.max() + 1e-12)
    rho = floor + (1.0 - floor) * (g_norm ** p_power)
    weights = rho * cell_areas

    weights_sum = weights.sum()
    if weights_sum <= 0:
        raise ValueError("All weights are zero; check u_cells / gradient.")

    prob = weights / weights_sum
    cdf = np.cumsum(prob)

    # ---------------------------------------------------------
    # 5) Sample interior points according to density
    # ---------------------------------------------------------
    r = rng.random(n_vertices_new)
    cell_indices = np.searchsorted(cdf, r)

    def sample_in_tri(tri_pts, r1, r2):
        # Uniform in area inside triangle using barycentric coords
        s = np.sqrt(r1)
        l1 = 1.0 - s
        l2 = s * r2
        l3 = s * (1.0 - r2)
        return (
            l1[..., None] * tri_pts[0]
            + l2[..., None] * tri_pts[1]
            + l3[..., None] * tri_pts[2]
        )

    r1 = rng.random(n_vertices_new)
    r2 = rng.random(n_vertices_new)
    interior_pts = np.empty((n_vertices_new, 2), dtype=float)

    for k, ci in enumerate(cell_indices):
        tri_vertices = points[triangles[ci]]
        interior_pts[k] = sample_in_tri(tri_vertices, r1[k], r2[k])

    # ---------------------------------------------------------
    # 6) Automatically detect boundary edges and loops (outer + holes)
    # ---------------------------------------------------------
    # Boundary edges = edges used by exactly one triangle
    boundary_edges = [
        e for e, cells in edge_to_cells.items() if len(cells) == 1
    ]
    boundary_edges = np.array(boundary_edges, int)  # (E, 2)

    # Build adjacency on boundary vertices
    adj = defaultdict(list)
    for a, b in boundary_edges:
        adj[a].append(b)
        adj[b].append(a)

    # Walk loops
    def extract_boundary_loops(adj_dict):
        loops = []
        visited_edges = set()

        def edge_key(i, j):
            return (min(i, j), max(i, j))

        for start in list(adj_dict.keys()):
            # Check if at least one unused edge from this start
            if all(edge_key(start, nb) in visited_edges for nb in adj_dict[start]):
                continue

            loop = [start]
            curr = start

            while True:
                next_v = None
                for nb in adj_dict[curr]:
                    ek = edge_key(curr, nb)
                    if ek not in visited_edges:
                        next_v = nb
                        visited_edges.add(ek)
                        break

                if next_v is None:
                    break

                if next_v == start:
                    break

                loop.append(next_v)
                curr = next_v

            if len(loop) >= 3:
                loops.append(np.array(loop, int))

        return loops

    loops = extract_boundary_loops(adj)

    if not loops:
        raise RuntimeError("No boundary loops detected; is the mesh closed?")

    # Compute polygon area for each loop, identify outer vs holes
    def polygon_area(coords):
        x = coords[:, 0]
        y = coords[:, 1]
        return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)

    loop_areas = np.array([polygon_area(points[loop]) for loop in loops])

    # Outer boundary = loop with largest |area|
    outer_idx = np.argmax(np.abs(loop_areas))
    outer_loop = loops[outer_idx]
    hole_loops = [loops[i] for i in range(len(loops)) if i != outer_idx]

    # Hole points: centroid of each hole polygon
    hole_points = []
    for hl in hole_loops:
        hp = points[hl].mean(axis=0)
        hole_points.append(hp)
    hole_points = np.array(hole_points) if hole_points else np.empty((0, 2))

    # ---------------------------------------------------------
    # 7) Build PSLG (vertices, segments, holes) for Triangle
    # ---------------------------------------------------------
    # Boundary vertices = all vertices that appear in any loop
    if hole_loops:
        boundary_vertex_indices = np.unique(
            np.concatenate([outer_loop] + hole_loops)
        )
    else:
        boundary_vertex_indices = np.unique(outer_loop)

    boundary_vertices = points[boundary_vertex_indices]

    # Map old vertex indices -> new boundary index
    remap = -np.ones(points.shape[0], int)
    remap[boundary_vertex_indices] = np.arange(len(boundary_vertices))

    # Combined vertices = boundary vertices + interior sampled points
    vertices = np.vstack([boundary_vertices, interior_pts])

    # Build segments from loops (outer + holes)
    segments = []

    def loop_to_segments(loop_indices):
        remapped = remap[loop_indices]
        if remapped.min() < 0:
            raise ValueError("Loop contains non-boundary vertex.")
        m = len(remapped)
        for i in range(m):
            a = remapped[i]
            b = remapped[(i + 1) % m]
            segments.append([a, b])

    loop_to_segments(outer_loop)
    for hl in hole_loops:
        loop_to_segments(hl)

    segments = np.array(segments, int)

    A = {
        "vertices": vertices,
        "segments": segments,
        "holes": hole_points,
    }

    # 'p' = triangulate PSLG; add 'q' / 'aX' if you want quality/area constraints
    B = tr.triangulate(A, "p")
    new_points = B["vertices"]
    new_triangles = B["triangles"]
    # NEW: boundary edges & points in the *new* mesh
    boundary_edges_new = B["segments"]  # (E, 2) indices into new_points
    boundary_vert_idx = np.unique(boundary_edges_new.ravel())
    boundary_points_new = new_points[boundary_vert_idx]  # (Nb, 2)

    # ---------------------------------------------------------
    # 8) Interpolate solution to new mesh, with NaN-fix
    # ---------------------------------------------------------
    # Primary interpolator: linear
    lin_interp = LinearNDInterpolator(cell_pts, u_cells, fill_value=np.nan)
    # Fallback: nearest neighbour (always finite if u_cells is finite)
    near_interp = NearestNDInterpolator(cell_pts, u_cells)

    # Nodal values
    u_nodes_new = lin_interp(new_points)
    u_nodes_new = np.asarray(u_nodes_new, float)
    nan_mask = np.isnan(u_nodes_new)
    if np.any(nan_mask):
        u_nodes_new[nan_mask] = near_interp(new_points[nan_mask])

    # Cell-centred values
    cell_pts_new = new_points[new_triangles].mean(axis=1)
    u_cells_new = lin_interp(cell_pts_new)
    u_cells_new = np.asarray(u_cells_new, float)
    nan_mask_cells = np.isnan(u_cells_new)
    if np.any(nan_mask_cells):
        u_cells_new[nan_mask_cells] = near_interp(cell_pts_new[nan_mask_cells])

    return new_points, new_triangles, u_nodes_new, u_cells_new, boundary_vert_idx
# import numpy as np
# from collections import defaultdict
# from scipy.interpolate import LinearNDInterpolator
# import triangle as tr  # pip install triangle
#
#
# def adaptive_remesh_smooth_boundary(
#     points,
#     triangles,
#     u_cells,
#     n_vertices_interior,
#     *,
#     p_power=1.0,
#     floor=0.1,
#     seed=None,
# ):
#     """
#     Adaptive remeshing for a 2D triangular unstructured mesh with possible holes.
#
#     - Uses a gradient-based sampling density from cell-centred data.
#     - Samples INTERIOR points according to that density.
#     - Automatically detects boundary edges (outer + holes) from triangles.
#     - RESAMPLES boundary loops (outer + holes) into quasi-uniform points
#       to get a smoother boundary.
#     - Builds a constrained Delaunay triangulation (Triangle) that respects
#       those boundary points and segments.
#     - Interpolates the cell-centred field onto the new mesh.
#
#     Parameters
#     ----------
#     points : (N, 2) array
#         Original vertex coordinates.
#     triangles : (M, 3) int array
#         Original triangle connectivity (indices into `points`).
#     u_cells : (M,) array
#         Scalar values at cell centroids (one per triangle).
#     n_vertices_interior : int
#         Number of INTERIOR sample points to add (boundary points are
#         generated separately).
#     p_power : float, optional
#         Exponent for normalized gradient in density function.
#         1.0 = linear; >1 emphasises sharp regions more.
#     floor : float in (0,1], optional
#         Minimum relative density so smooth regions still get some points.
#     seed : int or None, optional
#         Random seed for reproducibility.
#
#     Returns
#     -------
#     new_points : (N_new, 2) array
#         Vertex coordinates of the new mesh (boundary + interior).
#     new_triangles : (K, 3) int array
#         Triangle connectivity (indices into `new_points`).
#     u_nodes_new : (N_new,) array
#         Interpolated scalar values at the new mesh vertices.
#     u_cells_new : (K,) array
#         Interpolated scalar values at the new mesh cell centroids.
#
#     Notes
#     -----
#     - Boundary loops (outer + holes) are detected from edges that belong to
#       exactly one triangle.
#     - Boundary loops are re-sampled uniformly in arc length, which tends to
#       remove jaggedness from irregular vertex spacing.
#     - The number of boundary points is chosen based on a characteristic length
#       scale inferred from domain area and the interior point count.
#     """
#     rng = np.random.default_rng(seed)
#
#     points = np.asarray(points, float)
#     triangles = np.asarray(triangles, int)
#     u_cells = np.asarray(u_cells, float)
#
#     # ---------------------------------------------------------
#     # 1) Cell centroids & areas
#     # ---------------------------------------------------------
#     cell_pts = points[triangles].mean(axis=1)  # (M, 2)
#
#     def tri_area(tri_pts):
#         return 0.5 * np.abs(
#             (tri_pts[:, 1, 0] - tri_pts[:, 0, 0]) * (tri_pts[:, 2, 1] - tri_pts[:, 0, 1])
#             - (tri_pts[:, 2, 0] - tri_pts[:, 0, 0]) * (tri_pts[:, 1, 1] - tri_pts[:, 0, 1])
#         )
#
#     cell_areas = tri_area(points[triangles])  # (M,)
#     M = triangles.shape[0]
#
#     # ---------------------------------------------------------
#     # 2) Build edge -> cells map and cell neighbours
#     # ---------------------------------------------------------
#     edge_to_cells = defaultdict(list)
#     for ci, tri in enumerate(triangles):
#         a, b, c = tri
#         for e in ((a, b), (b, c), (c, a)):
#             e_sorted = (min(e), max(e))
#             edge_to_cells[e_sorted].append(ci)
#
#     neighbors = [[] for _ in range(M)]
#     for cells in edge_to_cells.values():
#         if len(cells) == 2:
#             c0, c1 = cells
#             neighbors[c0].append(c1)
#             neighbors[c1].append(c0)
#
#     neighbors = [np.array(nb, int) for nb in neighbors]
#
#     # ---------------------------------------------------------
#     # 3) Approximate |∇u| per cell using neighbour differences
#     # ---------------------------------------------------------
#     grad_mag = np.zeros(M)
#     for i in range(M):
#         nb = neighbors[i]
#         if nb.size == 0:
#             grad_mag[i] = 0.0
#             continue
#         du = u_cells[nb] - u_cells[i]
#         dx = cell_pts[nb] - cell_pts[i]
#         dist = np.linalg.norm(dx, axis=1) + 1e-12
#         grad_mag[i] = np.max(np.abs(du) / dist)
#
#     # ---------------------------------------------------------
#     # 4) Build sampling density from gradient
#     # ---------------------------------------------------------
#     g_norm = grad_mag / (grad_mag.max() + 1e-12)
#     rho = floor + (1.0 - floor) * (g_norm ** p_power)
#     weights = rho * cell_areas
#
#     weights_sum = weights.sum()
#     if weights_sum <= 0:
#         raise ValueError("All weights are zero; check u_cells / gradient.")
#
#     prob = weights / weights_sum
#     cdf = np.cumsum(prob)
#
#     # ---------------------------------------------------------
#     # 5) Sample interior points according to density
#     # ---------------------------------------------------------
#     def sample_in_tri(tri_pts, r1, r2):
#         # Uniform in area inside triangle using barycentric coords
#         s = np.sqrt(r1)
#         l1 = 1.0 - s
#         l2 = s * r2
#         l3 = s * (1.0 - r2)
#         return (
#             l1[..., None] * tri_pts[0]
#             + l2[..., None] * tri_pts[1]
#             + l3[..., None] * tri_pts[2]
#         )
#
#     if n_vertices_interior > 0:
#         r = rng.random(n_vertices_interior)
#         cell_indices = np.searchsorted(cdf, r)
#
#         r1 = rng.random(n_vertices_interior)
#         r2 = rng.random(n_vertices_interior)
#         interior_pts = np.empty((n_vertices_interior, 2), dtype=float)
#
#         for k, ci in enumerate(cell_indices):
#             tri_vertices = points[triangles[ci]]
#             interior_pts[k] = sample_in_tri(tri_vertices, r1[k], r2[k])
#     else:
#         interior_pts = np.empty((0, 2), dtype=float)
#
#     # ---------------------------------------------------------
#     # 6) Detect boundary edges and loops (outer + holes)
#     # ---------------------------------------------------------
#     # Boundary edges = edges used by exactly one triangle
#     boundary_edges = [
#         e for e, cells in edge_to_cells.items() if len(cells) == 1
#     ]
#     boundary_edges = np.array(boundary_edges, int)  # (E, 2)
#
#     if boundary_edges.size == 0:
#         raise RuntimeError("No boundary edges detected; is the mesh closed?")
#
#     # Build adjacency on boundary vertices
#     adj = defaultdict(list)
#     for a, b in boundary_edges:
#         adj[a].append(b)
#         adj[b].append(a)
#
#     def extract_boundary_loops(adj_dict):
#         loops = []
#         visited_edges = set()
#
#         def edge_key(i, j):
#             return (min(i, j), max(i, j))
#
#         for start in list(adj_dict.keys()):
#             # Check if there is at least one unused edge from this start
#             if all(edge_key(start, nb) in visited_edges for nb in adj_dict[start]):
#                 continue
#
#             loop = [start]
#             curr = start
#
#             while True:
#                 next_v = None
#                 for nb in adj_dict[curr]:
#                     ek = edge_key(curr, nb)
#                     if ek not in visited_edges:
#                         next_v = nb
#                         visited_edges.add(ek)
#                         break
#
#                 if next_v is None or next_v == start:
#                     break
#
#                 loop.append(next_v)
#                 curr = next_v
#
#             if len(loop) >= 3:
#                 loops.append(np.array(loop, int))
#
#         return loops
#
#     loops = extract_boundary_loops(adj)
#     if not loops:
#         raise RuntimeError("No boundary loops constructed; check mesh connectivity.")
#
#     # Classify outer vs inner (holes) by polygon area magnitude
#     def polygon_area(coords):
#         x = coords[:, 0]
#         y = coords[:, 1]
#         return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
#
#     loop_areas = np.array([polygon_area(points[loop]) for loop in loops])
#     outer_idx = np.argmax(np.abs(loop_areas))
#     outer_loop = loops[outer_idx]
#     hole_loops = [loops[i] for i in range(len(loops)) if i != outer_idx]
#
#     # ---------------------------------------------------------
#     # 7) RESAMPLE boundary loops to smooth jaggedness
#     # ---------------------------------------------------------
#     # Characteristic target length scale from domain area & interior density
#     domain_area = cell_areas.sum()
#     if n_vertices_interior > 0:
#         area_per_pt = domain_area / float(n_vertices_interior)
#         h_target = np.sqrt(area_per_pt)
#     else:
#         # Fallback: rough h from average edge length
#         all_edges = points[triangles][:, [0, 1, 2, 0]]
#         se = all_edges.reshape(-1, 2, 2)
#         h_target = np.mean(np.linalg.norm(se[:, 1] - se[:, 0], axis=1))
#
#     def resample_loop(loop_indices, h_target):
#         coords = points[loop_indices]  # (m, 2)
#         m = len(coords)
#         # Edge lengths, closed loop
#         diffs = coords[(np.arange(m) + 1) % m] - coords
#         seg_len = np.linalg.norm(diffs, axis=1)
#         perimeter = seg_len.sum()
#
#         # Number of points along this loop (at least original count)
#         n_pts = max(m, int(np.ceil(perimeter / h_target)))
#
#         # Parameter along perimeter
#         cumlen = np.concatenate([[0.0], np.cumsum(seg_len)])  # length m+1; cumlen[-1] = perimeter
#         s_new = np.linspace(0.0, perimeter, n_pts, endpoint=False)
#
#         new_coords = np.empty((n_pts, 2), dtype=float)
#         # For each target s, find which segment it falls in
#         seg_idx = np.searchsorted(cumlen, s_new, side="right") - 1
#         seg_idx = np.clip(seg_idx, 0, m - 1)
#
#         seg_start_s = cumlen[seg_idx]
#         t = (s_new - seg_start_s) / (seg_len[seg_idx] + 1e-12)
#
#         p0 = coords[seg_idx]
#         p1 = coords[(seg_idx + 1) % m]
#         new_coords = (1.0 - t)[:, None] * p0 + t[:, None] * p1
#
#         return new_coords
#
#     # Resample outer boundary and holes
#     outer_coords = resample_loop(outer_loop, h_target)
#     hole_coords_list = [resample_loop(hl, h_target) for hl in hole_loops]
#
#     # Hole interior points: centroid of each resampled loop
#     hole_points = np.array([hc.mean(axis=0) for hc in hole_coords_list]) if hole_coords_list else np.empty((0, 2))
#
#     # ---------------------------------------------------------
#     # 8) Build PSLG for Triangle: vertices, segments, holes
#     # ---------------------------------------------------------
#     boundary_vertices = [outer_coords] + hole_coords_list
#     boundary_counts = [len(outer_coords)] + [len(hc) for hc in hole_coords_list]
#
#     # Stack all boundary loops into a single array
#     boundary_vertices_all = np.vstack(boundary_vertices) if boundary_vertices else np.empty((0, 2))
#     n_boundary = len(boundary_vertices_all)
#
#     # Combined vertices: boundary + interior
#     vertices = np.vstack([boundary_vertices_all, interior_pts])
#     # segments: edges along each loop, using local indexing
#     segments = []
#     offset = 0
#     for count in boundary_counts:
#         idx = np.arange(offset, offset + count)
#         for i in range(count):
#             a = idx[i]
#             b = idx[(i + 1) % count]
#             segments.append([a, b])
#         offset += count
#
#     segments = np.array(segments, int)
#
#     A = {
#         "vertices": vertices,
#         "segments": segments,
#         "holes": hole_points,
#     }
#
#     # 'p' = triangulate PSLG; you can add 'q' / 'aX' options for quality/area
#     B = tr.triangulate(A, "p")
#     new_points = B["vertices"]
#     new_triangles = B["triangles"]
#
#     # ---------------------------------------------------------
#     # 9) Interpolate solution onto new mesh
#     # ---------------------------------------------------------
#     interp_u = LinearNDInterpolator(cell_pts, u_cells)
#
#     u_nodes_new = interp_u(new_points)
#     cell_pts_new = new_points[new_triangles].mean(axis=1)
#     u_cells_new = interp_u(cell_pts_new)
#
#     return new_points, new_triangles, u_nodes_new, u_cells_new


def main(save_dir='/home/maccyz/Documents/Neural_PDE/time_fvm/artefacts/saves/12-12_00-19-22'):
    """
    Plot out the saved mesh and time step data.
    """
    print(f"\nLoading from '{save_dir}'...")

    # Load mesh properties
    mesh_props_path = os.path.join(save_dir, 'mesh_props.npz')
    mesh_props = np.load(mesh_props_path)
    mesh_props = dict(mesh_props)
    print(f'{mesh_props.keys() = }')

    bc_tags = mesh_props.pop('bc_type_str')
    mesh_props = {k: torch.from_numpy(v) for k, v in mesh_props.items()}

    # Find and load time-step files
    time_files = sorted([f for f in os.listdir(save_dir) if f.startswith('t_') and f.endswith('.npz')])
    print(f"Found {len(time_files)} time-step file(s)")

    # Average graphs over time
    averaged_graphs = AveragedGraphs()
    time_files.sort(key=lambda x: float(x.split('_')[1].replace('.npz', '')))
    for save_i in time_files:
        file_path = os.path.join(save_dir, save_i)
        t, cell_primitives, bc_primitives = load_step(file_path)

        # print(f'{t = }')
        if t > 1: # Skip initial transients
            averaged_graphs.add_step(cell_primitives, bc_primitives)

    mean_cells, mean_bc = averaged_graphs.get_average()
    plot_interp(mesh_props['vertices'], mean_cells.T[:2], mesh_props['triangles'], title=f'Average over {averaged_graphs.count} steps')

    new_points, new_triangles, u_nodes_new, u_cells_new, bc_edges = adaptive_remesh_preserve_holes(
        points=mesh_props['vertices'].numpy(),
        triangles=mesh_props['triangles'].numpy(),
        u_cells=mean_cells[:, 0].numpy(),   # Use x-velocity for adaptivity
        n_vertices_new=mesh_props['vertices'].shape[0] // 4,  # Reduce to 1/4 vertices
        p_power=1.)

    new_points, u_cells_new, new_triangles = torch.from_numpy(new_points).float(), torch.from_numpy(u_cells_new).float(), torch.from_numpy(new_triangles)
    plot_interp(new_points, u_cells_new.reshape(-1, 1).T, new_triangles, title='Adaptively remeshed x-velocity')
    print()
    print(f'{mesh_props['vertices'].shape = }, {new_points.shape = }')
    print(f'{bc_edges = }')


if __name__ == '__main__':
    main()

