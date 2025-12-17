import torch
from cprint import c_print
import time


def build_sparse_gradient_matrix(combined_neigh, G_mat, dim, n_cells, n_boundaries):
    """
    Build a sparse gradient matrix for one spatial dimension using both cell neighbors and boundary edges.

    Args:
        combined_neigh Tensor: For each cell i, a 1D tensor of neighbor cell indices.
        G_mat (Tensor): For each cell i, a 2D tensor of shape [n_dims, num_total_neighbors_i].
                              The first columns correspond to cell neighbors and the remaining columns to edges.
        dim (int): The spatial dimension (0 for x, 1 for y) to build the gradient matrix.
        n_cells (int): Total number of cells.
        n_boundaries (int): Total number of boundary edges.

    Returns:
        A (torch.sparse.FloatTensor): A sparse matrix of shape [n_cells, n_cells+n_boundaries] that computes
                                      the gradient along the given dimension.
    """
    rows, cols, vals = [], [], []
    for i in range(n_cells):
        diag_val = 0.0
        # Loop over the combined neighbors.
        for k in range(combined_neigh[i].shape[0]):
            neighbor_idx = int(combined_neigh[i, k].item())
            # Use the appropriate weight from G_mat[i] (first cell_neigh.shape[0] entries correspond to cells)
            g_val = G_mat[i][dim, k]
            rows.append(i)
            cols.append(neighbor_idx)
            vals.append(g_val)
            diag_val += g_val

        # Diagonal entry: subtract the sum of all off-diagonals.
        rows.append(i)
        cols.append(i)
        vals.append(-diag_val)

    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.tensor(vals, dtype=G_mat[0].dtype)
    A = torch.sparse_coo_tensor(indices, values, (n_cells, n_cells + n_boundaries))
    return A


class FVMMesh:
    n_cells: int
    n_edges: int
    n_bc_edge: int
    # Local only for saving / plotting
    edges: torch.Tensor  # shape = (n_edges, 2)
    vertices: torch.Tensor  # shape = (n_vertices, 2)
    triangles: torch.Tensor  # shape = (n_cells, 3)
    # Used for FVM calculations
    bc_edge_mask: torch.Tensor  # shape = (n_edges)
    areas: torch.Tensor  # shape = (n_cells)
    normals: torch.Tensor  # shape = (n_edges, 2)
    lengths: torch.Tensor  # shape = (n_edges)
    centroids: torch.Tensor  # shape = (n_cells, 2)
    midpoints: torch.Tensor  # shape = (n_edges, 2)
    tri_to_edge: torch.Tensor  # shape = (n_cells, 3)
    tri_edge_signs: torch.Tensor  # shape = (n_cells, 3)
    edge_to_tri: dict[int, torch.Tensor]  # shape = {n_edges}[2]        # Mapping edge to triangle indices. Ordered [antiparallel, parallel] to edge normal.

    # Only for interior edges
    normals_main: torch.Tensor  # shape = (n_edge_main, 2)
    cell_grad_stuff: tuple # Stuff needed to calculate gradient on a cell
    edge_to_tri_main: torch.Tensor # shape = (n_edge_main, 2)              # Mapping edge to triangle indices for non boundary edges

    # Only for boundary edges
    edge_to_tri_bc: torch.Tensor # shape = (n_edge_bc, 1)              # Mapping edge to triangle indices for boundary edges
    normals_bc: torch.Tensor  # shape = (n_edge_bc, 2)

    def __init__(self, vertices, triangles, edges, bc_edge_mask, device="cuda"):
        self.vertices = vertices
        self.triangles = triangles
        self.edges = edges
        self.bc_edge_mask = bc_edge_mask
        self.device = device

        self.n_cells = triangles.shape[0]
        self.n_edges = edges.shape[0]
        self.n_bc_edge = bc_edge_mask.sum().item()
        assert edges.shape[0] == bc_edge_mask.shape[0], f'Different number of edges from bc edge mask {edges.shape = }, {bc_edge_mask.shape = }'

        c_print(f'Computing mesh properties', color="bright_magenta")
        self._compute_edge_props(vertices, triangles, edges)

    def _grad_weighting(self, tri_to_edge, edge_to_tri_ord, centroids, midpoints, normals):
        """ Use least squares formula to compute gradient weighting.
            grad(u) = A^-1 * b
            A = sum_i (d_i d_i^T)
            b = sum_i d_i (u_i - u_c)
        """

        bound_edge_idxs = torch.nonzero(self.bc_edge_mask, as_tuple=False).flatten()
        global_to_local = {int(global_idx): local_idx for local_idx, global_idx in enumerate(bound_edge_idxs)}
        edge_to_tri_comb = []
        for e, cell in edge_to_tri_ord.items():
            if len(cell) == 2:
                edge_to_tri_comb.append(cell)
            else:
                bc_edge_id = global_to_local[e] + self.n_cells
                bc_edge_id = torch.tensor([bc_edge_id, bc_edge_id])

                edge_to_tri_comb.append(bc_edge_id)
        edge_to_tri_comb = torch.stack(edge_to_tri_comb)

        combined_neigh, neigh_cents, combined_bc = [], [], []
        for cell_id, edges in enumerate(tri_to_edge):  # Must keep this order. Neighbor id: torch.cat([Us, Us_bc_edge])
            # Get neighboring cells
            neighbors, centers, is_bc = [], [], []
            for e in edges:
                e = e.item()
                if len(edge_to_tri_ord[e]) == 2:
                    """ Interior Edge"""
                    is_bc.append(False)
                    tris = edge_to_tri_ord[e]
                    neigh_cell = tris[tris != cell_id]
                    centers.append(centroids[neigh_cell])  # [1, 2]
                    neighbors.append(neigh_cell.item())
                else:
                    """ Boundary edge """
                    is_bc.append(True)
                    midpoint = midpoints[e].unsqueeze(0)
                    centers.append(midpoint)
                    glob_edge_idx = global_to_local[e]
                    neighbors.append(glob_edge_idx + self.n_cells)

            combined_neigh.append(torch.tensor(neighbors))
            neigh_cents.append(torch.cat(centers))  # [3, 2]
            combined_bc.append(torch.tensor(is_bc)) # [3]
        combined_neigh = torch.stack(combined_neigh).int()
        neigh_cents = torch.stack(neigh_cents)  # [n_cells, 3, 2]
        combined_bc = torch.stack(combined_bc).bool()  # [n_cells, 3]

        # --- Compute gradient vectors in batch ---
        # For each cell, compute the displacement vectors d_i = (neighbor center - cell center)
        center_expanded = centroids.unsqueeze(1)  # shape: [n_cells, 1, 2]
        d = neigh_cents - center_expanded  # shape: [n_cells, 3, 2]
        # Compute weights per neighbor: w_i = 1 / norm(d_i) ** k
        w = 1 / torch.norm(d.double(), dim=2) ** 0.25 # shape: [n_cells, 3]
        w[combined_bc] *= 7.5
        w2 = w ** 2  # shape: [n_cells, 3]
        # Compute A = dᵀ @ diag(w²) @ d for each cell.
        # dᵀ has shape [n_cells, 2, 3] and d * w2.unsqueeze(-1) scales each 2D neighbor vector.
        dT = d.transpose(1, 2).double()  # shape: [n_cells, 2, 3]
        A = torch.bmm(dT, d * w2.unsqueeze(-1))  # shape: [n_cells, 2, 2]
        # Invert A for each cell.
        A_inv = torch.inverse(A.double())  # shape: [n_cells, 2, 2]
        # Finally, compute the gradient matrix as A_inv @ dᵀ @ diag(w²)
        # Multiply dᵀ by w2 along the neighbor dimension:
        A_inv_di_T = torch.bmm(A_inv, dT * w2.unsqueeze(1)).float()  # shape: [n_cells, 2, 3]

        G_mats = []
        for i in range(2):
            G_mat = build_sparse_gradient_matrix(combined_neigh, A_inv_di_T, i, self.n_cells, self.n_bc_edge)
            G_mats.append(G_mat)

        # Get displacement between cells with edge indexing. In direction of right to left
        cell_disps, edge_dist_bc = [], []
        for e, cells in edge_to_tri_ord.items():
            if cells.shape[0] == 1:
                # BC cell / edge: Distance from centroid to edge.
                n_hat = normals[e] / torch.norm(normals[e], dim=-1, keepdim=True)
                f = midpoints[e]
                p = centroids[cells[0]]
                disp = n_hat * torch.dot(f - p, n_hat)
                sign = torch.sign(torch.dot(f - p, n_hat))
                dist = torch.norm(disp) * sign
                edge_dist_bc.append(dist)
            else:
                # Main cell / edge: Distance between centroids
                d = centroids[cells[1]] - centroids[cells[0]]
                cell_disps.append(d)
        cell_disps = torch.stack(cell_disps)
        edge_dist_bc = torch.stack(edge_dist_bc)

        return cell_disps, edge_dist_bc, G_mats, combined_neigh, edge_to_tri_comb

    def _compute_edge_props(self, vertices, triangles, edges):
        # Compute edge normals and lengths
        edge_vertex = vertices[edges]
        edge_vectors = edge_vertex[:, 1] - edge_vertex[:, 0]        # Ordering is used as edge index from here.
        normals = torch.stack([edge_vectors[:, 1], -edge_vectors[:, 0]], dim=1)
        midpoints = torch.mean(edge_vertex, dim=1)      # shape = [n_edges, 2]
        self.edge_vertex = edge_vertex                  # shape = [n_edges, 2, 2]
        self.normals = normals                          # shape = [n_edges, 2]
        self.midpoints = midpoints                      # shape = [n_edges, 2]

        # Triangle area and centroid
        tri_points = vertices[triangles]
        self.areas = self._tri_area(tri_points)
        self.centroids = torch.mean(tri_points, dim=1)  # shape = [n_cells, 2]

        # Compute mapping of edges to triangles
        tri_to_edge = self._get_tri_edges(triangles, edges) # shape = [n_cells, 3]
        self.tri_to_edge = tri_to_edge
        unique_edges, _ = torch.unique(tri_to_edge, sorted=True, return_inverse=True)
        _edge_to_tri, tri_edge_idxs = {}, {}
        for edge in unique_edges:
            pos = (edge == tri_to_edge).nonzero()
            _edge_to_tri[edge.item()] = pos[:, 0]
            tri_edge_idxs[edge.item()] = pos[:, 1]

        # Sort triangle in order of edge signed direction. ORDER: [-, +], so cell on right comes first.
        self.tri_edge_signs, edge_to_tri, cent_to_edge_disp = self._tri_edge_sign(self.centroids, midpoints, tri_to_edge, self.normals, _edge_to_tri, tri_edge_idxs)
        self.edge_to_tri = edge_to_tri

        # Split tensors into edge and main
        normals_main, edge_to_tri_main = [], []
        edge_to_tri_bc, normals_bc = [], []
        for e_idx, e_bc in enumerate(self.bc_edge_mask):
            if e_bc:
                # Precompute tensors for boundary edges
                edge_to_tri_bc.append(edge_to_tri[e_idx])
                normals_bc.append(normals[e_idx])
            else:
                # Precompute tensors for interior edges
                normals_main.append(normals[e_idx])
                edge_to_tri_main.append(edge_to_tri[e_idx])

        self.normals_main = torch.stack(normals_main)
        self.edge_to_tri_main = torch.stack(edge_to_tri_main)
        self.cent_to_edge_disp = cent_to_edge_disp
        self.edge_to_tri_bc = torch.stack(edge_to_tri_bc).squeeze()
        self.normals_bc = torch.stack(normals_bc)

        # Compute grad weighting
        self.cell_grad_stuff = self._grad_weighting(tri_to_edge, edge_to_tri, self.centroids, midpoints, normals)

    def _tri_area(self, vertices):
        """ vertices.shape = (n_cells, 3, 2) """
        a, b, c = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        # Compute the vectors for each triangle
        ab = b - a  # shape [n, 2]
        ac = c - a  # shape [n, 2]
        # Compute the 2D cross product (determinant) for each triangle
        cross = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]  # shape [n]
        # Triangle area is half the absolute value of the cross product
        area = 0.5 * torch.abs(cross)

        return area

    def _tri_edge_sign(self, centroids, midpoints, tri_to_edge, normals, edge_to_tri, tri_edge_idxs):
        """ Compute which triangle is on the left and right of each edge.
            For ordering, cell on Left comes first, then right
            Signs: 1 if on left, -1 if on right.
        """
        # signs = []
        # cent_to_edge_disp = []
        # for tri_idx, (edge, center) in enumerate(zip(tri_to_edge, centroids)):
        #     midpoint = midpoints[edge]      # shape = [3, 2]
        #     normal = normals[edge]          # shape = [3, 2]
        #
        #     p_diff = midpoint - center      # shape = [3, 2]
        #
        #     # Or normals dot (midpt-center)
        #     norm_hat = normal / torch.norm(normal, dim=-1, keepdim=True)
        #     dist_dot = torch.sum(norm_hat * p_diff, dim=-1)
        #
        #     sign_dot = torch.sign(dist_dot)
        #     signs.append(sign_dot)
        #
        #     # Compute shortest vector from centroid to line.
        #     r = p_diff
        #     cent_to_edge_disp.append(r)
        #
        # signs = torch.stack(signs).long()       # shape = [n_cells, 3]
        # cent_to_edge_disp = torch.stack(cent_to_edge_disp)  # shape = [n_cells, 3, 2]

        midpoints_tri = midpoints[tri_to_edge]  # shape: [n_cells, 3, 2]
        normals_tri = normals[tri_to_edge]  # shape: [n_cells, 3, 2]
        # Compute the difference between each edge midpoint and the centroid.
        p_diff = midpoints_tri - centroids.unsqueeze(1)  # shape: [n_cells, 3, 2]
        # Normalize the normals along the last dimension.
        norms = torch.norm(normals_tri, dim=-1, keepdim=True)  # shape: [n_cells, 3, 1]
        norm_hat = normals_tri / norms  # shape: [n_cells, 3, 2]
        # Compute the dot product and then its sign.
        dist_dot = torch.sum(norm_hat * p_diff, dim=-1)  # shape: [n_cells, 3]
        signs = torch.sign(dist_dot).long()  # shape: [n_cells, 3]
        # The displacement vectors are simply p_diff.
        cent_to_edge_disp = p_diff  # shape: [n_cells, 3, 2]

        edge_to_tri_ordered = {}
        p_m, m_p = torch.tensor([1, -1]), torch.tensor([-1, 1])
        for edge in sorted(edge_to_tri.keys()):
            tri_idx = edge_to_tri[edge]
            tri_edge = tri_edge_idxs[edge]

            order = signs[tri_idx, tri_edge]

            # Boundary edges only have 1 triangle
            if order.shape[0] == 1:
                assert self.bc_edge_mask[edge] == True, "Inconsistent boundary bug"
            else:
                if torch.all(order == m_p):
                    tri_idx = torch.flip(tri_idx, dims=[0])
            edge_to_tri_ordered[edge] = tri_idx

        return signs, edge_to_tri_ordered, cent_to_edge_disp

    def _get_tri_edges(self, triangles, edges):
        """
            Compute which edges belong to each triangle
            triangles.shape = (n_cells, 3)
            edges.shape = (n_edges, 2)
        """
        # 1) Normalize each edge (sort nodes in ascending order).
        # -------------------------------------------------------
        # edges_sorted will be shape [m, 2] with each row sorted.
        edges_sorted, _ = edges.sort(dim=1)

        # 2) Build a lookup: (nodeA, nodeB) -> edge_index
        # -----------------------------------------------
        edge_dict = {}
        for idx, e in enumerate(edges_sorted):
            # Make a tuple key (nodeA, nodeB)
            key = (e[0].item(), e[1].item())
            edge_dict[key] = idx

        # 3) For each triangle, find the 3 edges
        # --------------------------------------
        # We'll create a result tensor of shape [num_triangles, 3],
        # each row will store the indices of the 3 edges of that triangle.

        tri_to_edge = []
        for tri in triangles:
            # Extract triangle nodes (v0, v1, v2)
            v0 = tri[0].item()
            v1 = tri[1].item()
            v2 = tri[2].item()

            # Sort each pair so we can look it up in the edge_dict
            e1 = tuple(sorted((v0, v1)))
            e2 = tuple(sorted((v1, v2)))
            e3 = tuple(sorted((v2, v0)))

            # Get the edge indices
            edge_indices = [edge_dict[e1], edge_dict[e2], edge_dict[e3]]
            #edge_indices = sorted(edge_indices)
            tri_to_edge.append(edge_indices)

        tri_to_edge = torch.tensor(tri_to_edge)
        return tri_to_edge
