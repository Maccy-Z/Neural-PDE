import torch


def build_sparse_gradient_matrix(cell_to_neigh_idx, G_mat, dim):
    """
    Build a sparse gradient matrix for one spatial dimension.

    Args:
        cell_to_neigh_idx (list[Tensor]): List of 1D tensors, where cell_to_neigh_idx[i]
                                          holds the neighbor indices for cell i.
        G_mat (list[Tensor]): List of 2D tensors. For each cell i, G_mat[i] has shape
                              [n_dims, num_neighbors_i]. The row corresponding to `dim`
                              gives the weights for that cell in that spatial dimension.
        dim (int): The spatial dimension for which to build the matrix (e.g., 0 for x, 1 for y).

    Returns:
        A (torch.sparse.FloatTensor): A sparse matrix of shape [n_cells, n_cells] that
                                      computes the contribution along the given dimension.
    """
    n_cells = len(cell_to_neigh_idx)
    rows, cols = [], []
    vals = []

    for i in range(n_cells):
        diag_val = 0.0  # To accumulate the weight for the diagonal (self) contribution.
        # Get the neighbors for cell i.
        neighs = cell_to_neigh_idx[i]
        num_neigh = neighs.shape[0]

        for k in range(num_neigh):
            j = int(neighs[k].item())
            # Extract the gradient weight for cell i, neighbor k in the given dimension.
            g_val = G_mat[i][dim, k]
            # Off-diagonal: add contribution from neighbor j.
            rows.append(i), cols.append(j)
            vals.append(g_val)
            diag_val += g_val

        # Diagonal: subtract the sum of the weights.
        rows.append(i), cols.append(i)
        vals.append(-diag_val)

    # Build the sparse matrix.
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.tensor(vals, dtype=G_mat[0].dtype)
    A = torch.sparse_coo_tensor(indices, values, (n_cells, n_cells))
    return A

class FVMMesh:
    n_cells: int
    n_edges: int
    n_bc_edge: int

    areas: torch.Tensor  # shape = (n_cells)
    normals: torch.Tensor  # shape = (n_edges, 2)
    lengths: torch.Tensor  # shape = (n_edges)
    centroids: torch.Tensor  # shape = (n_cells, 2)
    tri_to_edge: torch.Tensor  # shape = (n_cells, 3)
    tri_edge_sign: torch.Tensor  # shape = (n_cells, 3)
    edge_to_tri: dict[int, torch.Tensor]  # shape = {n_edges}[2]        # Mapping edge to triangle indices. Ordered [antiparallel, parallel] to edge normal.

    # Only for interior edges
    normals_main: torch.Tensor  # shape = (n_edge_main, 2)
    cell_grad_stuff: tuple # Stuff needed to calculate gradient on a cell
    edge_to_tri_main: torch.Tensor # shape = (n_edge_main, 2)              # Mapping edge to triangle indices for non boundary edges
    edge_to_tri_w_main: torch.Tensor  # shape = (n_edge_main, 2)         # Weights for interpolation of face value

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

        self._compute_edge_props(vertices, triangles, edges)

        self._compute_bc_props(bc_edge_mask)


    def _compute_bc_props(self, bc_edge_mask):
        """ Boundary indices are specified w.r.t. boundary mask, not main edge index. """

    def _grad_weighting(self, tri_to_edge, edge_to_tri_ord, centroids, midpoints, normals):
        """ Use least squares formula to compute gradient weighting.
            grad(u) = A^-1 * b
            A = sum_i (d_i d_i^T)
            b = sum_i d_i (u_i - u_c)

         """
        cell_to_neigh_idx = []
        cell_A_inv = []
        cell_d_i = []

        # Weighting matrix for gradient using least squares
        for cell_id, edges in enumerate(tri_to_edge):
            # Get neighboring cells
            neighbors = []
            for e in edges:
                neighbors.append(edge_to_tri_ord[e.item()])
            neighbors = torch.unique(torch.cat(neighbors))
            mask = neighbors != cell_id
            neighbors = neighbors[mask]

            # Compute gradient matrix and distance vectors
            center = centroids[cell_id]      # [2]
            neighbors_cent = centroids[neighbors]  # [3, 2]

            d_i = neighbors_cent - center

            A = d_i.T @ d_i
            # If a cell only has 1 neighbor, assume gradient is in direction of cell.
            if len(neighbors) == 1:
                A_inv = torch.pinverse(A)
            else:
                A_inv = torch.inverse(A)

            cell_to_neigh_idx.append(neighbors)
            cell_A_inv.append(A_inv)
            cell_d_i.append(d_i)

        # Get displacement between cells with edge indexing. In direction of normal
        edge_dist, edge_dist_bc = [], []
        for e, cells in edge_to_tri_ord.items():
            if cells.shape[0] == 1:
                # BC cell / edge: Distance from centroid to edge.
                n_hat = normals[e] / torch.norm(normals[e], dim=-1, keepdim=True)
                f = midpoints[e]
                p = centroids[cells[0]]
                disp = n_hat * torch.dot(f - p, n_hat)
                dist = torch.norm(disp)
                edge_dist_bc.append(dist)
                continue
            else:
                # Main cell / edge: Distance between centroids
                d = centroids[cells[1]] - centroids[cells[0]]
                edge_dist.append(d)

        edge_dist = torch.stack(edge_dist)
        edge_dist_bc = torch.stack(edge_dist_bc)


        # Premultiply A_inv with d_i.T
        A_inv_di_T = []
        for A_inv, d_i in zip(cell_A_inv, cell_d_i):
            A_inv_di_T.append(A_inv @ d_i.T)

        G_mats = []
        for i in range(2):
            G_mat = build_sparse_gradient_matrix(cell_to_neigh_idx, A_inv_di_T, i)
            G_mats.append(G_mat)

        return edge_dist, edge_dist_bc, G_mats


    def _compute_edge_props(self, vertices, triangles, edges):
        # Compute edge normals and lengths
        edge_vertex = vertices[edges]
        edge_vectors = edge_vertex[:, 1] - edge_vertex[:, 0]        # Ordering is used as edge index from here.
        normals = torch.stack([edge_vectors[:, 1], -edge_vectors[:, 0]], dim=1)
        self.normals = normals                          # shape = [n_edges, 2]
        midpoints = torch.mean(edge_vertex, dim=1)      # shape = [n_edges, 2]

        self.midpoints = midpoints                      # shape = [n_edges, 2]
        self.edge_vertex = edge_vertex                  # shape = [n_edges, 2, 2]

        # Triangle area and centroid
        tri_points = vertices[triangles]
        self.areas = self._tri_area(tri_points)
        self.centroids = torch.mean(tri_points, dim=1)  # shape = [n_cells, 2]

        # Compute mapping of edges to triangles
        tri_to_edge = self._get_tri_edges(triangles, edges) # shape = [n_cells, 3]
        self.tri_to_edge = tri_to_edge
        unique_edges, _ = torch.unique(tri_to_edge, sorted=True, return_inverse=True)
        edge_to_tri, tri_edge_idxs = {}, {}
        for edge in unique_edges:
            pos = (edge == tri_to_edge).nonzero()
            edge_to_tri[edge.item()] = pos[:, 0]
            tri_edge_idxs[edge.item()] = pos[:, 1]

        # Sort triangle in order of edge signed direction
        self.tri_edge_signs = self._tri_edge_sign(self.centroids, edge_vectors, midpoints, tri_to_edge, self.normals)
        # ORDER: [-, +], so edge normal parallel to center comes last.
        edge_to_tri_ordered = {}
        p_m, m_p = torch.tensor([1, -1]), torch.tensor([-1, 1])
        for edge in sorted(edge_to_tri.keys()):
            tri_idx = edge_to_tri[edge]
            tri_edge = tri_edge_idxs[edge]

            order = self.tri_edge_signs[tri_idx, tri_edge]

            # Boundary edges only have 1 triangle
            if order.shape[0] == 1:
                assert self.bc_edge_mask[edge] == True, "Inconsistent boundary bug"
            else:
                if torch.all(order == p_m):
                    tri_idx = torch.flip(tri_idx, dims=[0])
            edge_to_tri_ordered[edge] = tri_idx

        self.edge_to_tri = edge_to_tri_ordered

        # Compute distance from triangle centroid to edge midpoint
        # weight = [d_far / (d_far + d_near)]
        edge_to_tri_w = {}
        for edge, tri in edge_to_tri_ordered.items():
            if tri.shape[0] == 1:
                edge_to_tri_w[edge] = None
                continue

            v = midpoints[edge] - self.centroids[tri]
            n = self.normals[edge]
            n_hat = n / torch.norm(n, dim=-1, keepdim=True)
            d = torch.abs(torch.sum(v * n_hat, dim=-1))

            w_anti = d[1] / (d[0] + d[1])
            w_para = d[0] / (d[0] + d[1])
            edge_to_tri_w[edge] = torch.stack([w_anti, w_para], dim=0)

        # Compute grad weighting
        self.cell_grad_stuff = self._grad_weighting(tri_to_edge, self.edge_to_tri, self.centroids, midpoints, normals)

        # Precompute tensors for interior edges
        normals_main, edge_to_tri_main, edge_to_tri_w_main = [], [], []
        for e_idx, e_bc in enumerate(self.bc_edge_mask):
            if e_bc:
                continue
            normals_main.append(normals[e_idx])
            edge_to_tri_main.append(self.edge_to_tri[e_idx])
            edge_to_tri_w_main.append(edge_to_tri_w[e_idx])

        self.normals_main = torch.stack(normals_main)
        self.edge_to_tri_main = torch.stack(edge_to_tri_main)
        self.edge_to_tri_w_main = torch.stack(edge_to_tri_w_main)

        # Precompute tensors for boundary edges
        edge_to_tri_bc, normals_bc = [], []
        for e_idx, e_bc in enumerate(self.bc_edge_mask):
            if not e_bc:
                continue
            edge_to_tri_bc.append(self.edge_to_tri[e_idx])
            normals_bc.append(normals[e_idx])
        self.edge_to_tri_bc = torch.stack(edge_to_tri_bc).squeeze()
        self.normals_bc = torch.stack(normals_bc)

    def _tri_area(self, vertices):
        """ vertices.shape = (n_cells, 3, 2) """
        a = vertices[:, 0]
        b = vertices[:, 1]
        c = vertices[:, 2]

        # Compute the vectors for each triangle
        ab = b - a  # shape [n, 2]
        ac = c - a  # shape [n, 2]

        # Compute the 2D cross product (determinant) for each triangle
        cross = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]  # shape [n]

        # Triangle area is half the absolute value of the cross product
        area = 0.5 * torch.abs(cross)

        return area

    def _tri_edge_sign(self, centroids, edge_vectors, midpoints, tri_to_edge, normals):
        signs = []

        for edge, center in zip(tri_to_edge, centroids):
            edge_vect = edge_vectors[edge]      # shape = [3, 2]
            midpoint = midpoints[edge]      # shape = [3, 2]
            normal = normals[edge]          # shape = [3, 2]

            p_diff = midpoint - center      # shape = [3, 2]
            #p_diff = p_diff / torch.norm(p_diff, dim=-1, keepdim=True)
            edge_vect = edge_vect / torch.norm(edge_vect, dim=-1, keepdim=True)

            # (midpt-center) X edge_vect
            cross = edge_vect[:, 0] * p_diff[:, 1] - edge_vect[:, 1] * p_diff[:, 0]
            sign_X = -torch.sign(cross)
            # Or normals dot (center - midpoint)
            dot = torch.sum(normal * p_diff, dim=-1)
            sign_dot = torch.sign(dot)

            assert torch.all(sign_X == sign_dot), f'{sign_X = }, {sign_dot = }'

            signs.append(sign_dot)

        signs = torch.stack(signs).long()
        return signs

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
            edge_indices = [
                edge_dict[e1],
                edge_dict[e2],
                edge_dict[e3]
            ]
            tri_to_edge.append(edge_indices)

        tri_to_edge = torch.tensor(tri_to_edge)
        return tri_to_edge
