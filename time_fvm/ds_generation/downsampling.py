import numpy as np
from collections import defaultdict
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from meshpy.triangle import MeshInfo, build


def weighted_poisson_points_fast(
    P, w, n_samples, r0, seed=None, max_trials=300000, batch=4096
):
    rng = np.random.default_rng(seed)

    P = np.asarray(P, float)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError("P must have shape (N, 2).")
    N = P.shape[0]

    w = np.asarray(w, float)
    w = np.clip(w, 1e-12, None)
    p = w / w.sum()
    cdf = np.cumsum(p)
    w_mean = w.mean()

    radii = r0 / np.sqrt(w / w_mean)

    r_min = float(radii.min())
    cell_size = r_min / np.sqrt(2.0)
    if not np.isfinite(cell_size) or cell_size <= 0:
        cell_size = r0 / np.sqrt(2.0)
    inv_cell = 1.0 / cell_size

    grid = {}
    selected_idx = []
    selected_xy = []

    def cell_of(x0, x1):
        return (int(np.floor(x0 * inv_cell)), int(np.floor(x1 * inv_cell)))

    trials = 0
    while trials < max_trials and len(selected_idx) < n_samples:
        k = min(batch, max_trials - trials)
        u = rng.random(k)
        cand = np.searchsorted(cdf, u, side="right")
        trials += k

        for i in cand:
            if len(selected_idx) >= n_samples:
                break

            x0, x1 = float(P[i, 0]), float(P[i, 1])
            ri = float(radii[i])
            ri2 = ri * ri

            gx, gy = cell_of(x0, x1)
            m = int(np.ceil(ri * inv_cell))

            ok = True
            for dx in range(-m, m + 1):
                for dy in range(-m, m + 1):
                    lst = grid.get((gx + dx, gy + dy))
                    if not lst:
                        continue
                    for j in lst:
                        y0, y1 = selected_xy[j]
                        d0 = x0 - y0
                        d1 = x1 - y1
                        if d0 * d0 + d1 * d1 < ri2:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break

            if ok:
                selected_idx.append(int(i))
                selected_xy.append((x0, x1))
                j = len(selected_xy) - 1
                grid.setdefault((gx, gy), []).append(j)

    return np.array(selected_idx, dtype=int)


def _sample_in_tri(tri_pts, r1, r2):
    s = np.sqrt(r1)
    l1 = 1.0 - s
    l2 = s * r2
    l3 = s * (1.0 - r2)
    return (
        l1[..., None] * tri_pts[0]
        + l2[..., None] * tri_pts[1]
        + l3[..., None] * tri_pts[2]
    )


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


def _sampling_prob_from_gradient(grad_mag, cell_areas, *, p_power=1.0, floor=0.1, g_quant=0.95):
    g_norm = grad_mag / np.quantile(grad_mag, g_quant)
    rho = floor + (1.0 - floor) * (g_norm ** p_power)
    weights = rho * cell_areas
    return weights / weights.sum()


def _sample_interior_points(points, triangles, cell_pts, prob, n_vertices_new, *, seed=None, r0=0.02):
    rng = np.random.default_rng(seed)

    selected_cells = weighted_poisson_points_fast(
        cell_pts, prob, n_vertices_new, r0, seed=seed
    )

    n_sel = int(selected_cells.size)
    if n_sel == 0:
        return np.empty((0, 2), dtype=float), selected_cells

    r1 = rng.random(n_sel)
    r2 = rng.random(n_sel)

    interior_pts = np.empty((n_sel, 2), dtype=float)
    for k, ci in enumerate(selected_cells):
        tri_vertices = points[triangles[ci]]
        interior_pts[k] = _sample_in_tri(tri_vertices, r1[k], r2[k])

    return interior_pts, selected_cells


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


# ---- NEW: subsample loops (edge decimation) -----------------

def _subsample_loop(loop_indices, *, keep_ratio=1.0, min_points=8):
    """
    Uniformly subsample a closed loop of vertex indices.
    Keeps roughly keep_ratio of vertices (uniformly spaced).
    """
    loop_indices = np.asarray(loop_indices, dtype=int)
    L = int(loop_indices.size)
    if L <= 3 or keep_ratio >= 1.0:
        return loop_indices

    n_keep = int(np.ceil(L * keep_ratio))
    n_keep = max(3, min(L, n_keep))
    n_keep = max(min_points, n_keep) if L >= min_points else L

    if n_keep >= L:
        return loop_indices

    # Uniformly spaced picks along the loop
    pos = np.linspace(0, L, n_keep, endpoint=False)
    idx = np.unique(np.floor(pos).astype(int))
    # If unique() reduced count due to flooring, pad deterministically
    if idx.size < 3:
        idx = np.array([0, L // 3, 2 * L // 3], dtype=int)
    return loop_indices[idx]


def _subsample_loops(points, outer_loop, hole_loops, *, boundary_keep_ratio=1.0, boundary_min_points=8):
    outer2 = _subsample_loop(outer_loop, keep_ratio=boundary_keep_ratio, min_points=boundary_min_points)
    holes2 = [
        _subsample_loop(h, keep_ratio=boundary_keep_ratio, min_points=boundary_min_points)
        for h in hole_loops
    ]

    # recompute hole points after subsampling (centroid of polygon vertices)
    hole_points = np.array([points[h].mean(axis=0) for h in holes2], dtype=float) if holes2 else np.empty((0, 2))
    return outer2, holes2, hole_points


def _build_pslg(points, outer_loop, hole_loops, hole_points, interior_pts):
    if hole_loops:
        boundary_vertex_indices = np.unique(np.concatenate([outer_loop] + hole_loops))
    else:
        boundary_vertex_indices = np.unique(outer_loop)

    boundary_vertices = points[boundary_vertex_indices]

    remap = -np.ones(points.shape[0], dtype=int)
    remap[boundary_vertex_indices] = np.arange(len(boundary_vertices))

    vertices = np.vstack([boundary_vertices, interior_pts])

    segments = []

    def add_loop_segments(loop_indices):
        remapped = remap[loop_indices]
        if remapped.min() < 0:
            raise ValueError("Loop contains non-boundary vertex.")
        m = len(remapped)
        for i in range(m):
            a = int(remapped[i])
            b = int(remapped[(i + 1) % m])
            segments.append([a, b])

    add_loop_segments(outer_loop)
    for hl in hole_loops:
        add_loop_segments(hl)

    A = {
        "vertices": vertices,
        "segments": np.asarray(segments, dtype=int),
        "holes": hole_points,
    }
    return A


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
    if point_markers is not None:
        mesh_info.set_points(A["vertices"].tolist(), point_markers=point_markers.tolist())
    else:
        mesh_info.set_points(A["vertices"].tolist())

    # Set segments (edges) with segment markers if available
    # CRITICAL: Segment markers ensure Steiner points inherit the correct tag
    segment_markers = A.get("segment_markers")
    if segment_markers is not None:
        mesh_info.set_facets(A["segments"].tolist(), facet_markers=segment_markers.tolist())
    else:
        mesh_info.set_facets(A["segments"].tolist())

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
    if hasattr(mesh, 'facets') and mesh.facets:
        boundary_edges_new = np.array(mesh.facets, dtype=int)
    else:
        boundary_edges_new = np.empty((0, 2), dtype=int)

    # Extract point markers and convert back to tags
    point_tags = None
    if point_markers is not None and hasattr(mesh, 'point_markers'):
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
        new_bc_vertex_ids: boundary vertex IDs (local indices 0 to len(new_bc_points)-1)
        new_bc_point_tags: corresponding tags for new points (one tag per point)
        bc_vertex_indices: original vertex indices used in boundary
        loop_groups: list of arrays, where each array contains vertex IDs for one independent loop
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

        # Group edges by tag and build loop groups properly
        unique_tags = list(set(bc_tags))
        tag_groups = {tag: [] for tag in unique_tags}
        for i, tag in enumerate(bc_tags):
            tag_groups[tag].append(bc_edges[i])

        point_tags = [None] * len(bc_vertex_indices)
        loop_groups = []
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
                    # Convert to local indices and store as loop group
                    loop_local_ids = [remap[v] for v in chain]
                    loop_groups.append(np.array(loop_local_ids, dtype=int))

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

        bc_vertex_ids = np.arange(len(bc_vertex_indices), dtype=int)
        new_bc_edges = np.array(new_edges_list, dtype=int) if new_edges_list else np.empty((0, 2), dtype=int)
        return bc_points, bc_vertex_ids, point_tags, bc_vertex_indices, loop_groups, new_bc_edges, new_edge_tags_list

    # Group edges by tag
    unique_tags = list(set(bc_tags))
    tag_groups = {tag: [] for tag in unique_tags}

    for i, tag in enumerate(bc_tags):
        tag_groups[tag].append(bc_edges[i])

    # For each tag, build ordered chains of vertices
    new_points_list = []
    new_point_tags_list = []
    new_edges_list = []  # Track edges with local indices
    new_edge_tags_list = []  # Track tag for each edge
    loop_groups = []  # Track which vertices belong to each independent loop
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

            # Track the local vertex IDs for this loop
            loop_vertex_ids = []

            # Add points and create mapping
            for global_idx in selected:
                if global_idx not in global_to_local:
                    global_to_local[global_idx] = current_idx
                    new_points_list.append(points[global_idx])
                    new_point_tags_list.append(tag)  # Assign tag to point
                    loop_vertex_ids.append(current_idx)
                    current_idx += 1
                else:
                    # Point already added (shouldn't happen in well-formed input)
                    loop_vertex_ids.append(global_to_local[global_idx])

            # Create edges for this chain with proper tags
            # IMPORTANT: All edges within a tag group get that tag
            for j in range(len(selected) - 1):
                v1_global = selected[j]
                v2_global = selected[j + 1]
                v1_local = global_to_local[v1_global]
                v2_local = global_to_local[v2_global]
                new_edges_list.append([v1_local, v2_local])
                new_edge_tags_list.append(tag)

            # Check if this is a closed loop
            if len(selected) >= 2:
                first_v = selected[0]
                last_v = selected[-1]
                is_closed = (last_v in adj[first_v]) and (first_v in adj[last_v])

                if is_closed:
                    # Add closing edge
                    v1_local = global_to_local[last_v]
                    v2_local = global_to_local[first_v]
                    new_edges_list.append([v1_local, v2_local])
                    new_edge_tags_list.append(tag)

            # Store this loop's vertex IDs
            if len(loop_vertex_ids) >= 2:
                loop_groups.append(np.array(loop_vertex_ids, dtype=int))

    new_bc_points = np.array(new_points_list, dtype=float)
    new_bc_vertex_ids = np.arange(len(new_bc_points), dtype=int)
    new_bc_edges = np.array(new_edges_list, dtype=int) if new_edges_list else np.empty((0, 2), dtype=int)
    bc_vertex_indices = np.array(list(global_to_local.keys()), dtype=int)

    return new_bc_points, new_bc_vertex_ids, new_point_tags_list, bc_vertex_indices, loop_groups, new_bc_edges, new_edge_tags_list


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
    n_vertices_new,
    bc_edges,
    bc_tags,
    *,
    p_power=1.0,
    floor=0.1,
    g_quant=0.95,
    seed=None,
    r0=0.02,
    boundary_keep_ratio=0.5,
    boundary_min_points=8,
    holes=None,
):
    """
    Adaptive remeshing based on solution gradient.

    Args:
        points: (N, 2) current mesh vertices
        triangles: (M, 3) current triangulation
        u_cells: (M, d) solution values at cell centers
        n_vertices_new: target number of interior vertices
        bc_edges: (K, 2) boundary edges [v1_idx, v2_idx] (indices into points)
        bc_tags: list of K string tags corresponding to each boundary edge
        ----- Mesh adaptivity parameters -----
        p_power: gradient weighting power
        floor: minimum sampling probability
        g_quant: gradient quantile for normalization
        seed: random seed
        r0: base Poisson disk radius
        boundary_keep_ratio: fraction of boundary vertices to keep
        boundary_min_points: minimum boundary points per tag group
        holes: optional (H, 2) array of hole seed points (if None, auto-detect from mesh)

    Returns:
        new_points: remeshed vertices
    Returns:
        new_points: remeshed vertices
        new_triangles: new triangulation
        u_nodes_new: interpolated solution at new vertices
        u_cells_new: interpolated solution at new cell centers
        new_bc_edges: new boundary edges (indices into new_points)
        new_bc_tags: preserved boundary tags (one tag per boundary point)
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
            print(f"Detected {len(hole_loops)} hole(s) in mesh")
        else:
            holes = np.empty((0, 2), dtype=float)
    else:
        holes = np.asarray(holes, float)

    # 4) |∇u| per cell
    grad_mag = _cell_gradient_magnitude(cell_pts, u_cells, neighbors)

    # 5) Sampling probability
    prob = _sampling_prob_from_gradient(
        grad_mag, cell_areas, p_power=p_power, floor=floor, g_quant=g_quant
    )

    # 6) Sample interior points
    interior_pts, _ = _sample_interior_points(
        points, triangles, cell_pts, prob, n_vertices_new, seed=seed, r0=r0
    )

    # 7) Subsample boundary edges
    new_bc_points, new_bc_vertex_ids, new_bc_tags, bc_vertex_indices, loop_groups, bc_edges_for_pslg, bc_edge_tags = _subsample_boundary_edges(
        points,
        bc_edges,
        bc_tags,
        boundary_keep_ratio=boundary_keep_ratio,
        boundary_min_points=boundary_min_points,
    )

    # 8) Build PSLG and triangulate
    # Use edges and edge tags directly from _subsample_boundary_edges
    A = _build_pslg_from_boundary(new_bc_points, bc_edges_for_pslg, interior_pts,
                                   bc_point_tags=new_bc_tags, bc_edge_tags=bc_edge_tags, holes=holes)
    new_points, new_triangles, _, point_tags_from_meshpy = _triangulate_pslg(A, opts="pq")
    cell_pts_new = new_points[new_triangles].mean(axis=1)

    # 10) Interpolate solution
    u_nodes_new = _interpolate_with_nan_fix(cell_pts, u_cells, new_points)
    u_cells_new = _interpolate_with_nan_fix(cell_pts, u_cells, cell_pts_new)

    # 11) Extract boundary vertex IDs and tags from meshpy output
    # Meshpy may have added boundary points, so we need to identify them from the point markers
    if point_tags_from_meshpy is not None:
        # Boundary vertices are those with non-None tags
        boundary_mask = [tag is not None for tag in point_tags_from_meshpy]
        final_bc_vertex_ids = np.where(boundary_mask)[0]
        final_bc_tags = [tag for tag in point_tags_from_meshpy if tag is not None]
    else:
        # Fallback: assume first n_boundary points are boundary (old behavior)
        n_boundary = new_bc_points.shape[0]
        final_bc_vertex_ids = np.arange(n_boundary, dtype=int)
        final_bc_tags = new_bc_tags
    return (
        new_points,
        new_triangles,
        u_nodes_new,
        u_cells_new,
        final_bc_vertex_ids,
        final_bc_tags,
    )
