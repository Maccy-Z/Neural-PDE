import math
import numpy as np
from collections import defaultdict
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from meshpy.triangle import MeshInfo, build
from shapely.geometry import Point, Polygon
from shapely.prepared import prep


def poisson_disk_variable_r(polygon, r_of, r_min, r_max, clearance_of=None, k=30, seed=0):
    """
    Bridson-style Poisson disk sampling with variable radius r(p).
    Enforces separation: ||p-q|| >= 0.5*(r(p)+r(q))
    Optionally enforces boundary clearance: dist(p, polygon.boundary) >= clearance_of(p)

    Args:
        polygon: shapely Polygon object (supports holes)
        r_of: function that returns radius at point p (tuple or array)
        r_min: minimum radius globally
        r_max: maximum radius globally
        clearance_of: optional function returning clearance distance at point p
        k: number of attempts per active point
        seed: random seed

    Returns:
        (N, 2) array of sampled points
    """
    rng = np.random.default_rng(seed)
    minx, miny, maxx, maxy = polygon.bounds

    # Prepare geometry for faster containment checks (5-10x speedup)
    prepared_polygon = prep(polygon)

    # Grid based on global r_min (safe)
    cell = r_min / math.sqrt(2)
    inv_cell = 1.0 / cell
    gw = int(math.ceil((maxx - minx) * inv_cell))
    gh = int(math.ceil((maxy - miny) * inv_cell))
    grid = -np.ones((gh, gw), dtype=int)

    # Use numpy array for points instead of list for faster indexing
    pts = np.empty((10000, 2), dtype=float)  # Pre-allocate, will grow if needed
    n_pts = 0
    active = []

    # Cache for r_of values to avoid recomputation
    r_cache = np.empty(10000, dtype=float)

    def grid_coords(px, py):
        gx = int((px - minx) * inv_cell)
        gy = int((py - miny) * inv_cell)
        return gx, gy

    def far_enough_from_boundary(px, py):
        if clearance_of is None:
            return True
        # Avoid creating Point object - use prepared geometry if available
        p_point = Point(px, py)
        return p_point.distance(polygon.boundary) >= float(clearance_of((px, py)))

    def too_close(px, py, rp):
        # Conservative scan radius to not miss conflicts
        scan_R = 0.5 * (rp + r_max)
        n = int(math.ceil(scan_R * inv_cell))

        gx, gy = grid_coords(px, py)
        x0, x1 = max(gx - n, 0), min(gx + n + 1, gw)
        y0, y1 = max(gy - n, 0), min(gy + n + 1, gh)

        # Vectorized approach: collect all valid point indices first
        grid_slice = grid[y0:y1, x0:x1]
        valid_mask = grid_slice >= 0

        if not np.any(valid_mask):
            return False

        # Get all valid point indices
        point_indices = grid_slice[valid_mask]

        # Vectorized distance computation
        neighbor_pts = pts[point_indices]
        dx = neighbor_pts[:, 0] - px
        dy = neighbor_pts[:, 1] - py
        dist_sq = dx*dx + dy*dy

        # Vectorized minimum separation check
        neighbor_radii = r_cache[point_indices]
        min_sep = 0.5 * (rp + neighbor_radii)
        min_sep_sq = min_sep * min_sep

        return np.any(dist_sq < min_sep_sq)

    # Initial point - generate batch to improve polygon.contains performance
    batch_size = 100
    for attempt in range(200):  # 200 * 100 = 20000 max attempts
        candidates = rng.uniform([minx, miny], [maxx, maxy], size=(batch_size, 2))

        for px, py in candidates:
            p_point = Point(px, py)
            if prepared_polygon.contains(p_point) and far_enough_from_boundary(px, py):
                pts[0] = [px, py]
                r_cache[0] = float(r_of((px, py)))
                n_pts = 1
                active.append(0)
                gx, gy = grid_coords(px, py)
                if 0 <= gx < gw and 0 <= gy < gh:
                    grid[gy, gx] = 0
                break
        if n_pts > 0:
            break

    if n_pts == 0:
        return np.empty((0, 2), dtype=float)

    # Main loop
    while active:
        # Pick random active point
        active_pick = rng.integers(0, len(active))
        a_idx = active[active_pick]
        base_x, base_y = pts[a_idx]
        r_base = r_cache[a_idx]
        found = False

        # Generate k candidate points at once
        angles = rng.uniform(0.0, 2.0 * math.pi, k)
        radii = rng.uniform(r_base, 2.0 * r_base, k)

        for ang, rad in zip(angles, radii):
            px = base_x + rad * math.cos(ang)
            py = base_y + rad * math.sin(ang)

            # Bounds check
            if not (minx <= px <= maxx and miny <= py <= maxy):
                continue

            # Polygon containment check (most expensive - use prepared geometry)
            if not prepared_polygon.contains(Point(px, py)):
                continue

            # Get radius for this point
            rp = float(r_of((px, py)))

            # Boundary clearance
            if clearance_of is not None:
                if not far_enough_from_boundary(px, py):
                    continue

            # Check separation from existing points
            if too_close(px, py, rp):
                continue

            # Accept point
            # Grow arrays if needed
            if n_pts >= len(pts):
                pts = np.resize(pts, (len(pts) * 2, 2))
                r_cache = np.resize(r_cache, len(r_cache) * 2)

            pts[n_pts] = [px, py]
            r_cache[n_pts] = rp
            active.append(n_pts)
            gx, gy = grid_coords(px, py)
            if 0 <= gx < gw and 0 <= gy < gh:
                grid[gy, gx] = n_pts
            n_pts += 1
            found = True
            break

        if not found:
            # Remove from active list efficiently (swap with last)
            active[active_pick] = active[-1]
            active.pop()

    return pts[:n_pts].copy()


def _tri_centroids_and_areas(points, triangles):
    tri_pts = points[triangles]
    centroids = tri_pts.mean(axis=1)

    x0, y0 = tri_pts[:, 0, 0], tri_pts[:, 0, 1]
    x1, y1 = tri_pts[:, 1, 0], tri_pts[:, 1, 1]
    x2, y2 = tri_pts[:, 2, 0], tri_pts[:, 2, 1]
    areas = 0.5 * np.abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
    return centroids, areas


def _build_edge_maps_and_neighbors(triangles):
    edge_to_cells = defaultdict(list)
    for ci, tri in enumerate(triangles):
        a, b, c = tri
        for e in ((a, b), (b, c), (c, a)):
            e_sorted = (min(e), max(e))
            edge_to_cells[e_sorted].append(ci)

    boundary_edges = np.array(
        [e for e, cells in edge_to_cells.items() if len(cells) == 1],
        dtype=int,
    )

    M = triangles.shape[0]
    neighbors = [[] for _ in range(M)]
    for cells in edge_to_cells.values():
        if len(cells) == 2:
            c0, c1 = cells
            neighbors[c0].append(c1)
            neighbors[c1].append(c0)

    neighbors = [np.array(nb, dtype=int) for nb in neighbors]
    return edge_to_cells, boundary_edges, neighbors


def _cell_gradient_magnitude(cell_pts, u_cells, neighbors):
    M = cell_pts.shape[0]
    grad_mag = np.zeros(M, dtype=float)
    for i in range(M):
        nb = neighbors[i]
        if nb.size == 0:
            continue

        du = np.linalg.norm(u_cells[nb] - u_cells[i], axis=-1)
        dx = cell_pts[nb] - cell_pts[i]
        dist = np.linalg.norm(dx, axis=1) + 1e-12
        grad_mag[i] = np.max(np.abs(du) / dist)

    return grad_mag


def _sample_interior_points(polygon, grad_mag, r_min, r_max, *, p_power=1.0, floor=0.1, clearance_of=None, seed=None):
    """
    Sample interior points using Poisson disk sampling with variable radius based on gradient.

    Args:
        polygon: shapely Polygon object defining the domain (with holes)
        grad_mag: interpolator or callable that returns gradient magnitude at a point
        r_min: minimum radius for sampling
        r_max: maximum radius for sampling
        p_power: gradient weighting power
        floor: minimum relative density
        clearance_of: optional function returning clearance distance at point p
        seed: random seed

    Returns:
        (N, 2) array of sampled interior points
    """
    # Create a radius function based on gradient magnitude
    # Higher gradient -> smaller radius -> denser sampling
    def r_of(p):
        g = grad_mag(p)
        # Normalize gradient (simple approach)
        g_norm = max(0.0, min(1.0, g))  # Clamp to [0, 1]
        # Invert: high gradient -> small radius
        rho = floor + (1.0 - floor) * (1.0 - g_norm ** p_power)
        return r_min + (r_max - r_min) * rho

    # Sample using Poisson disk
    interior_pts = poisson_disk_variable_r(
        polygon, r_of=r_of, r_min=r_min, r_max=r_max,
        clearance_of=clearance_of, k=30, seed=seed
    )

    return interior_pts


def _extract_boundary_loops_from_edges(boundary_edges):
    """
    Extract closed loops from boundary edges.

    Args:
        boundary_edges: (M, 2) array of boundary edge pairs

    Returns:
        loops: list of numpy arrays, each containing vertex indices forming a loop
    """
    adj = defaultdict(list)
    for a, b in boundary_edges:
        adj[a].append(b)
        adj[b].append(a)

    visited_edges = set()

    def edge_key(i, j):
        return (min(i, j), max(i, j))

    loops = []
    for start in list(adj.keys()):
        if all(edge_key(start, nb) in visited_edges for nb in adj[start]):
            continue

        loop = [start]
        curr = start

        while True:
            next_v = None
            for nb in adj[curr]:
                ek = edge_key(curr, nb)
                if ek not in visited_edges:
                    next_v = nb
                    visited_edges.add(ek)
                    break

            if next_v is None or next_v == start:
                break

            loop.append(next_v)
            curr = next_v

        if len(loop) >= 3:
            loops.append(np.array(loop, dtype=int))

    return loops


def _split_outer_and_holes(points, loops):
    """
    Separate the outer boundary loop from hole loops based on signed area.
    The loop with the largest absolute area is considered the outer boundary.

    Args:
        points: (N, 2) array of vertex coordinates
        loops: list of loops (each is array of vertex indices)

    Returns:
        outer_loop: array of vertex indices for outer boundary
        hole_loops: list of arrays for hole boundaries
    """
    if not loops:
        raise RuntimeError("No boundary loops detected; is the mesh closed?")

    def polygon_area(coords):
        x = coords[:, 0]
        y = coords[:, 1]
        return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)

    loop_areas = np.array([polygon_area(points[loop]) for loop in loops])
    outer_idx = int(np.argmax(np.abs(loop_areas)))
    outer_loop = loops[outer_idx]
    hole_loops = [loops[i] for i in range(len(loops)) if i != outer_idx]

    return outer_loop, hole_loops


def _compute_hole_seeds(points, hole_loops):
    """
    Compute seed points (centroids) for hole loops.

    Args:
        points: (N, 2) array of vertex coordinates
        hole_loops: list of arrays, each containing vertex indices for a hole

    Returns:
        hole_seeds: (H, 2) array of hole seed points
    """
    if not hole_loops:
        return np.empty((0, 2), dtype=float)

    hole_seeds = []
    for hole_loop in hole_loops:
        centroid = points[hole_loop].mean(axis=0)
        hole_seeds.append(centroid)

    return np.array(hole_seeds, dtype=float)


def _triangulate_pslg(A, opts="pq"):
    """
    Triangulate using meshpy instead of triangle.
    opts string: 'p' = PSLG, 'q' = quality (min angle), etc.

    Returns:
        new_points: vertex coordinates
        new_triangles: triangulation
        boundary_edges_new: boundary edges
        point_tags: list of tags for each point (None if no markers provided)
    """
    mesh_info = MeshInfo()

    # Set vertices with point markers if available
    point_markers = A.get("point_markers")
    mesh_info.set_points(A["vertices"].tolist(), point_markers=point_markers.tolist())

    # Set segments (edges) with segment markers if available
    segment_markers = A.get("segment_markers")
    mesh_info.set_facets(A["segments"].tolist(), facet_markers=segment_markers.tolist())

    # Set holes if any
    if A["holes"].size > 0:
        holes = A["holes"].tolist()
        mesh_info.set_holes(holes)

    # Build mesh with options
    # Parse opts string: 'p' for PSLG, 'q' for quality
    max_volume = None
    min_angle = None

    if 'q' in opts:
        min_angle = 20.0  # default minimum angle in degrees

    # Build the mesh
    mesh = build(mesh_info, max_volume=max_volume, min_angle=min_angle,
                 allow_boundary_steiner=True, generate_faces=True)

    # Extract results
    new_points = np.array(mesh.points, dtype=float)
    new_triangles = np.array(mesh.elements, dtype=int)

    # Extract boundary segments (facets)
    boundary_edges_new = np.array(mesh.facets, dtype=int)

    # Extract point markers and convert back to tags
    marker_to_tag = A.get("marker_to_tag", {})
    mesh_point_markers = np.array(mesh.point_markers, dtype=int)
    point_tags = []
    for marker in mesh_point_markers:
        if marker in marker_to_tag:
            point_tags.append(marker_to_tag[marker])
        else:
            point_tags.append(None)  # Interior or unmarked point


    return new_points, new_triangles, boundary_edges_new, point_tags


def _interpolate_with_nan_fix(xy_src, values_src, xy_query):
    lin = LinearNDInterpolator(xy_src, values_src, fill_value=np.nan)
    near = NearestNDInterpolator(xy_src, values_src)

    out = np.asarray(lin(xy_query), float)
    nan_mask = np.isnan(out)

    if np.any(nan_mask):
        nan_rows = np.any(nan_mask, axis=1) if out.ndim == 2 else nan_mask
        vals_nn = np.asarray(near(xy_query[nan_rows]), float)

        if out.ndim == 1:
            out[nan_rows] = vals_nn
        else:
            out[nan_mask] = vals_nn[np.isnan(out[nan_rows])]

    return out


def _subsample_boundary_edges(points, bc_edges, bc_tags, *, boundary_keep_ratio=1.0, boundary_min_points=8):
    """
    Subsample boundary edges while preserving topology and tags.
    Groups edges by tag, subsamples each group, and returns new boundary points and edges.

    Args:
        points: (N, 2) array of all mesh vertex coordinates
        bc_edges: (M, 2) array where each row is [vertex_idx1, vertex_idx2]
        bc_tags: list of M string tags corresponding to each edge
        boundary_keep_ratio: fraction of boundary points to keep per tag group
        boundary_min_points: minimum points to keep per tag group

    Returns:
        new_bc_points: subsampled boundary points
        new_bc_point_tags: corresponding tags for new points (one tag per point)
        new_bc_edges: edges with local vertex IDs
        new_bc_edge_tags: tag for each edge (preserves original edge tags)
    """
    bc_edges = np.asarray(bc_edges, int)
    points = np.asarray(points, float)

    if boundary_keep_ratio >= 1.0:
        # Extract unique boundary vertices
        bc_vertex_indices = np.unique(bc_edges.flatten())
        bc_points = points[bc_vertex_indices]

        # Create remapping from global to local indices
        remap = -np.ones(points.shape[0], dtype=int)
        remap[bc_vertex_indices] = np.arange(len(bc_vertex_indices))

        # Group edges by tag
        unique_tags = list(set(bc_tags))
        tag_groups = {tag: [] for tag in unique_tags}
        for i, tag in enumerate(bc_tags):
            tag_groups[tag].append(bc_edges[i])

        point_tags = [None] * len(bc_vertex_indices)
        new_edges_list = []
        new_edge_tags_list = []

        for tag in unique_tags:
            edges = np.array(tag_groups[tag], dtype=int)

            # Build adjacency for this tag group
            adj = defaultdict(list)
            for a, b in edges:
                adj[a].append(b)
                adj[b].append(a)

            # Extract ordered chains for this tag
            visited = set()
            for start_v in list(adj.keys()):
                if start_v in visited:
                    continue

                # Build chain
                current = start_v
                chain = [current]
                visited.add(current)

                while True:
                    neighbors = [n for n in adj[current] if n not in visited]
                    if not neighbors:
                        break
                    current = neighbors[0]
                    chain.append(current)
                    visited.add(current)

                if len(chain) >= 2:
                    # Assign tags to these points
                    for v in chain:
                        local_v = remap[v]
                        if point_tags[local_v] is None:
                            point_tags[local_v] = tag

                    # Create edges for this chain with proper tag
                    for j in range(len(chain) - 1):
                        v1_local = remap[chain[j]]
                        v2_local = remap[chain[j + 1]]
                        new_edges_list.append([v1_local, v2_local])
                        new_edge_tags_list.append(tag)

                    # Check if closed and add closing edge
                    if chain[-1] in adj[chain[0]] and chain[0] in adj[chain[-1]]:
                        v1_local = remap[chain[-1]]
                        v2_local = remap[chain[0]]
                        new_edges_list.append([v1_local, v2_local])
                        new_edge_tags_list.append(tag)

        new_bc_edges = np.array(new_edges_list, dtype=int) if new_edges_list else np.empty((0, 2), dtype=int)
        return bc_points, point_tags, new_bc_edges, new_edge_tags_list

    # Group edges by tag
    unique_tags = list(set(bc_tags))
    tag_groups = {tag: [] for tag in unique_tags}

    for i, tag in enumerate(bc_tags):
        tag_groups[tag].append(bc_edges[i])

    # For each tag, build ordered chains of vertices
    new_points_list = []
    new_point_tags_list = []
    new_edges_list = []
    new_edge_tags_list = []
    global_to_local = {}  # Maps original vertex index to new local index
    current_idx = 0

    for tag in unique_tags:
        edges = np.array(tag_groups[tag], dtype=int)

        # Build adjacency for this tag group
        adj = defaultdict(list)
        for a, b in edges:
            adj[a].append(b)
            adj[b].append(a)

        # Extract ordered chains (handles both open and closed boundaries)
        visited = set()
        chains = []

        for start_v in list(adj.keys()):
            if start_v in visited:
                continue

            # Find endpoint (degree 1) or any unvisited vertex
            current = start_v
            chain = [current]
            visited.add(current)

            # Traverse forward
            while True:
                neighbors = [n for n in adj[current] if n not in visited]
                if not neighbors:
                    break
                current = neighbors[0]
                chain.append(current)
                visited.add(current)

            if len(chain) >= 2:
                chains.append(np.array(chain, dtype=int))

        # Subsample each chain
        for chain in chains:
            L = len(chain)
            n_keep = max(boundary_min_points, int(np.ceil(L * boundary_keep_ratio)))
            n_keep = min(L, n_keep)

            if n_keep >= L:
                selected = chain
            else:
                # Uniformly sample along the chain
                pos = np.linspace(0, L - 1, n_keep)
                selected = chain[np.unique(np.round(pos).astype(int))]

            # Add points and create mapping
            for global_idx in selected:
                if global_idx not in global_to_local:
                    global_to_local[global_idx] = current_idx
                    new_points_list.append(points[global_idx])
                    new_point_tags_list.append(tag)
                    current_idx += 1

            # Create edges for this chain with proper tags
            for j in range(len(selected) - 1):
                v1_global = selected[j]
                v2_global = selected[j + 1]
                v1_local = global_to_local[v1_global]
                v2_local = global_to_local[v2_global]
                new_edges_list.append([v1_local, v2_local])
                new_edge_tags_list.append(tag)

            # Check if this is a closed loop and add closing edge
            if len(selected) >= 2:
                first_v = selected[0]
                last_v = selected[-1]
                is_closed = (last_v in adj[first_v]) and (first_v in adj[last_v])

                if is_closed:
                    v1_local = global_to_local[last_v]
                    v2_local = global_to_local[first_v]
                    new_edges_list.append([v1_local, v2_local])
                    new_edge_tags_list.append(tag)

    new_bc_points = np.array(new_points_list, dtype=float)
    new_bc_edges = np.array(new_edges_list, dtype=int) if new_edges_list else np.empty((0, 2), dtype=int)

    return new_bc_points, new_point_tags_list, new_bc_edges, new_edge_tags_list


def _build_pslg_from_boundary(bc_points, bc_edges, interior_pts, bc_point_tags=None, bc_edge_tags=None, holes=None):
    """
    Build PSLG directly from boundary points, edges, and interior points.

    Args:
        bc_points: (N, 2) boundary vertex coordinates (already local indices 0..N-1)
        bc_edges: (M, 2) array of boundary edge indices [v1_idx, v2_idx] (local to bc_points)
        interior_pts: (K, 2) interior sample points
        bc_point_tags: optional list of N tags (one per boundary point)
        bc_edge_tags: optional list of M tags (one per boundary edge) - takes precedence over bc_point_tags
        holes: optional (H, 2) hole seed points

    Returns:
        A: dictionary with 'vertices', 'segments', 'holes', 'point_markers', 'segment_markers', 'tag_to_marker', 'marker_to_tag'
    """
    n_boundary = bc_points.shape[0]

    # Combine boundary and interior points
    vertices = np.vstack([bc_points, interior_pts])

    # Segments use boundary edge indices directly
    segments = bc_edges.copy()

    # Handle holes
    if holes is None or (isinstance(holes, np.ndarray) and holes.size == 0):
        holes = np.empty((0, 2), dtype=float)
    else:
        holes = np.asarray(holes, float)

    # Create point markers and segment markers from tags
    point_markers = None
    segment_markers = None
    tag_to_marker = {}
    marker_to_tag = {}

    if bc_point_tags is not None or bc_edge_tags is not None:
        # Collect all unique tags from both point tags and edge tags
        all_tags = []
        if bc_point_tags is not None:
            all_tags.extend([tag for tag in bc_point_tags if tag is not None])
        if bc_edge_tags is not None:
            all_tags.extend([tag for tag in bc_edge_tags if tag is not None])

        unique_tags = []
        for tag in all_tags:
            if tag not in unique_tags:
                unique_tags.append(tag)

        # Map unique tags to integer markers (starting from 1)
        for i, tag in enumerate(unique_tags):
            marker_id = i + 1  # Start from 1, 0 is typically default/interior
            tag_to_marker[tag] = marker_id
            marker_to_tag[marker_id] = tag

        # Create point markers if point tags provided
        if bc_point_tags is not None:
            point_markers = np.zeros(len(vertices), dtype=int)
            for i, tag in enumerate(bc_point_tags):
                if tag is not None:
                    point_markers[i] = tag_to_marker[tag]

        # Create segment markers
        # CRITICAL: Use edge tags if provided, otherwise infer from vertex tags
        segment_markers = np.zeros(len(segments), dtype=int)

        if bc_edge_tags is not None:
            # Use explicit edge tags (preferred - avoids bleeding at junctions)
            for i, tag in enumerate(bc_edge_tags):
                if tag is not None:
                    segment_markers[i] = tag_to_marker[tag]
        elif bc_point_tags is not None:
            # Fallback: infer from vertex tags (can cause bleeding at junctions)
            for i, (v1, v2) in enumerate(segments):
                tag1 = bc_point_tags[v1] if v1 < len(bc_point_tags) else None
                tag2 = bc_point_tags[v2] if v2 < len(bc_point_tags) else None

                if tag1 == tag2 and tag1 is not None:
                    segment_markers[i] = tag_to_marker[tag1]
                elif tag1 is not None:
                    # Tags differ or tag2 is None - use tag1
                    segment_markers[i] = tag_to_marker[tag1]
                elif tag2 is not None:
                    segment_markers[i] = tag_to_marker[tag2]


    A = {
        "vertices": vertices,
        "segments": segments,
        "holes": holes,
        "n_boundary": n_boundary,
        "point_markers": point_markers,
        "segment_markers": segment_markers,
        "tag_to_marker": tag_to_marker,
        "marker_to_tag": marker_to_tag,
    }
    return A


def adaptive_remesh(
    points,
    triangles,
    u_cells,
    bc_edges,
    bc_tags,
    *,
    p_power=1.0,
    floor=0.1,
    g_quant=0.95,
    seed=None,
    r_min=0.01,
    r_max=0.1,
    boundary_keep_ratio=0.5,
    boundary_min_points=8,
    holes=None,
):
    """
    Adaptive remeshing based on solution gradient using simplified Poisson disk sampling.

    Args:
        points: (N, 2) current mesh vertices
        triangles: (M, 3) current triangulation
        u_cells: (M, d) solution values at cell centers
        bc_edges: (K, 2) boundary edges [v1_idx, v2_idx] (indices into points)
        bc_tags: list of K string tags corresponding to each boundary edge
        ----- Mesh adaptivity parameters -----
        p_power: gradient weighting power
        floor: minimum sampling density (0 to 1)
        g_quant: gradient quantile for normalization
        seed: random seed
        r_min: minimum radius for Poisson sampling (dense regions)
        r_max: maximum radius for Poisson sampling (coarse regions)
        boundary_keep_ratio: fraction of boundary vertices to keep
        boundary_min_points: minimum boundary points per tag group
        holes: optional (H, 2) array of hole seed points (if None, auto-detect from mesh)

    Returns:
        new_points: remeshed vertices
        new_triangles: new triangulation
        u_nodes_new: interpolated solution at new vertices
        final_bc_tags: preserved boundary tags (one tag per boundary point)
        new_bc_edges: new boundary edges (indices into new_points)
    """


    points = np.asarray(points, float)
    triangles = np.asarray(triangles, int)
    u_cells = np.asarray(u_cells, float)
    bc_edges = np.asarray(bc_edges, int)

    # 1) Cell centroids & areas
    cell_pts, cell_areas = _tri_centroids_and_areas(points, triangles)

    # 2) Edge maps + neighbours
    _, boundary_edges_detected, neighbors = _build_edge_maps_and_neighbors(triangles)

    # 3) Detect holes from mesh topology if not provided
    if holes is None:
        loops = _extract_boundary_loops_from_edges(boundary_edges_detected)
        if len(loops) > 1:
            outer_loop, hole_loops = _split_outer_and_holes(points, loops)
            holes = _compute_hole_seeds(points, hole_loops)
        else:
            holes = np.empty((0, 2), dtype=float)
    else:
        holes = np.asarray(holes, float)

    # 4) |∇u| per cell
    grad_mag = _cell_gradient_magnitude(cell_pts, u_cells, neighbors)

    # 5) Create gradient magnitude interpolator (for use in sampling)
    # Normalize gradient for use in radius function
    g_max = np.quantile(grad_mag, g_quant)
    grad_mag_norm = np.clip(grad_mag / (g_max + 1e-12), 0.0, 1.0)

    # Create interpolator for normalized gradient
    from scipy.interpolate import LinearNDInterpolator
    grad_interp_lin = LinearNDInterpolator(cell_pts, grad_mag_norm, fill_value=0.0)

    def grad_interp(p):
        val = float(grad_interp_lin(p))
        return val

    # 6) Subsample boundary edges first
    new_bc_points, new_bc_tags, bc_edges_for_pslg, bc_edge_tags = _subsample_boundary_edges(
        points,
        bc_edges,
        bc_tags,
        boundary_keep_ratio=boundary_keep_ratio,
        boundary_min_points=boundary_min_points,
    )

    # 7) Build shapely Polygon from boundary loops
    # Extract boundary loops from the subsampled boundary
    loops = _extract_boundary_loops_from_boundary_points(new_bc_points, bc_edges_for_pslg)

    if len(loops) == 0:
        raise RuntimeError("No boundary loops detected from subsampled boundary")

    # Separate outer boundary and holes
    if len(loops) > 1:
        outer_loop_coords = loops[0]  # Assume first is outer
        hole_coords = loops[1:]
        polygon = Polygon(shell=outer_loop_coords, holes=hole_coords)
    else:
        outer_loop_coords = loops[0]
        polygon = Polygon(shell=outer_loop_coords)

    # 8) Sample interior points using Poisson disk with gradient-based variable radius
    # Optional: add boundary clearance
    def clearance_of(p):
        # Clearance proportional to local radius
        g = grad_interp(p)
        rho = floor + (1.0 - floor) * (1.0 - g ** p_power)
        r_local = r_min + (r_max - r_min) * rho
        return 0.75 * r_local  # Keep points away from boundary

    interior_pts = _sample_interior_points(
        polygon, grad_interp, r_min, r_max,
        p_power=p_power, floor=floor,
        clearance_of=clearance_of, seed=seed
    )

    # 9) Build PSLG and triangulate
    A = _build_pslg_from_boundary(new_bc_points, bc_edges_for_pslg, interior_pts,
                                   bc_point_tags=new_bc_tags, bc_edge_tags=bc_edge_tags, holes=holes)
    new_points, new_triangles, new_bc_edges, new_point_tags = _triangulate_pslg(A, opts="pq")

    # 10) Interpolate solution
    u_nodes_new = _interpolate_with_nan_fix(cell_pts, u_cells, new_points)

    return (
        new_points,
        new_triangles,
        u_nodes_new,
        new_point_tags,
        new_bc_edges,
    )


def _extract_boundary_loops_from_boundary_points(bc_points, bc_edges):
    """
    Extract boundary loops as coordinate arrays from boundary points and edges.

    Args:
        bc_points: (N, 2) array of boundary vertex coordinates
        bc_edges: (M, 2) array of boundary edge indices

    Returns:
        loops: list of (K, 2) arrays, each containing coordinates of a closed loop
    """
    if len(bc_edges) == 0:
        return []

    # Build adjacency
    adj = defaultdict(list)
    for a, b in bc_edges:
        adj[a].append(b)
        adj[b].append(a)

    visited_edges = set()

    def edge_key(i, j):
        return (min(i, j), max(i, j))

    loops = []
    for start in list(adj.keys()):
        if all(edge_key(start, nb) in visited_edges for nb in adj[start]):
            continue

        loop = [start]
        curr = start

        while True:
            next_v = None
            for nb in adj[curr]:
                ek = edge_key(curr, nb)
                if ek not in visited_edges:
                    next_v = nb
                    visited_edges.add(ek)
                    break

            if next_v is None or next_v == start:
                break

            loop.append(next_v)
            curr = next_v

        if len(loop) >= 3:
            # Convert indices to coordinates
            loop_coords = bc_points[loop]
            loops.append(loop_coords)

    return loops

