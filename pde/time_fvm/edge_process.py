import torch
from pde.time_fvm.fvm_mesh import FVMMesh
from pde.graph_grid.fvm_store import Edge

import torch


def combine_edge_operators(A_main, A_bc, b_bc, bc_edge_mask, n_edges, n_cells, n_comp, device):
    """
    Combines a main-edge operator and a boundary-edge operator into a single global operator.

    Parameters:
      A_main      : sparse COO tensor of shape (n_edges_m*n_comp, n_cells*n_comp)
                    -- the main-edge operator (with local row ordering).
      A_bc        : sparse COO tensor of shape (n_edges_bc*n_comp, n_cells*n_comp)
                    -- the boundary-edge operator (with local row ordering).
      b_bc        : tensor of shape (n_edges_bc*n_comp,)
                    -- the offset vector for boundary edges.
      bc_edge_mask: Boolean tensor of shape (n_edges,)
                    -- True if the global edge is a boundary edge.
      n_edges     : int, total number of global edges.
      n_cells     : int, number of cells.
      n_comp      : int, number of components.
      device      : torch.device

    Returns:
      A_all       : sparse COO tensor of shape (n_edges*n_comp, n_cells*n_comp)
                    -- the combined operator.
      b_all       : tensor of shape (n_edges*n_comp,)
                    -- the combined offset vector.
    """

    # Get the COO indices and values for the two operators.
    # (They must be in COO format.)
    A_main_indices = A_main._indices()  # shape (2, L_main)
    A_main_values = A_main._values()  # shape (L_main,)
    A_bc_indices = A_bc._indices()  # shape (2, L_bc)
    A_bc_values = A_bc._values()  # shape (L_bc,)

    # We will build lists of row indices, column indices, and values for the global operator.
    global_rows = []
    global_cols = []
    global_vals = []
    b_all_list = []  # offset for each global row

    # Counters for the local row index in A_main and A_bc.
    # They indicate which main (or bc) edge (block) we are currently processing.
    main_counter = 0
    bc_counter = 0

    # Loop over all global edges.
    for i in range(n_edges):
        # For each edge, process all components.
        for c in range(n_comp):
            # Compute the flattened (global) row index for edge i and component c.
            global_row = i * n_comp + c

            if not bc_edge_mask[i]:
                # --- Main edge ---
                # The corresponding local row in A_main is:
                local_row = main_counter * n_comp + c
                # Find the entries in A_main corresponding to this local row.
                mask = (A_main_indices[0, :] == local_row)
                # (These entries come with column indices and values.)
                cols = A_main_indices[1, :][mask]
                vals = A_main_values[mask]
                # Append these entries, but with the global row instead of the local row.
                for col, val in zip(cols.tolist(), vals.tolist()):
                    global_rows.append(global_row)
                    global_cols.append(col)
                    global_vals.append(val)
                # For a main edge, no offset is added.
                b_all_list.append(0.0)
            else:
                # --- Boundary edge ---
                local_row = bc_counter * n_comp + c
                mask = (A_bc_indices[0, :] == local_row)
                cols = A_bc_indices[1, :][mask]
                vals = A_bc_values[mask]
                for col, val in zip(cols.tolist(), vals.tolist()):
                    global_rows.append(global_row)
                    global_cols.append(col)
                    global_vals.append(val)
                # The offset for boundary edges comes from b_bc.
                # (Assume b_bc is a 1D tensor of length n_edges_bc*n_comp.)
                b_all_list.append(b_bc[local_row].item())
        # Update the local counters.
        if not bc_edge_mask[i]:
            main_counter += 1
        else:
            bc_counter += 1

    # Convert the lists into tensors.
    indices = torch.tensor([global_rows, global_cols], dtype=torch.long, device=device)
    values = torch.tensor(global_vals, dtype=A_main_values.dtype, device=device)

    # The global operator acts on flattened cell fields of length n_cells*n_comp and produces
    # an output of length n_edges*n_comp.
    size_all = (n_edges * n_comp, n_cells * n_comp)
    A_all = torch.sparse_coo_tensor(indices, values, size=size_all).coalesce()
    b_all = torch.tensor(b_all_list, dtype=A_main_values.dtype, device=device)

    return A_all.to_sparse_csr(), b_all

def lift_sparse_matrix(A_old, n_comp):
    """
    Lift a sparse matrix so that it acts on a flattened multi-component vector.
        U_out = torch.sparse.mm(A_old, U)   # U_out has shape (M, n_comp)
        U_out = torch.sparse.mm(A_new, U.flatten()).reshape(M, n_comp)
    A_old : torch.sparse.Tensor
        A sparse matrix in COO format of shape (M, N).
    n_comp : int
        The number of components (i.e. the second dimension of U).
    A_new : torch.sparse.Tensor
        The "lifted" sparse matrix of shape (M*n_comp, N*n_comp) that operates on a flattened U.
    """
    # Get the original indices and values.
    # indices_old is a tensor of shape (2, nnz), where nnz is the number of nonzero entries.
    A_old = A_old.coalesce()

    indices_old = A_old.indices()  # shape: (2, nnz)
    values_old = A_old.values()  # shape: (nnz,)
    M, N = A_old.size()
    nnz = values_old.size(0)
    device = A_old.device

    # Create a vector for the component indices: 0, 1, ..., n_comp - 1.
    comp = torch.arange(n_comp, device=device)  # shape: (n_comp,)

    # For each nonzero entry in A_old, we replicate the index for each component.
    # The new row index for an entry originally at row i becomes:
    #    i_new = i * n_comp + c   for c in 0,..., n_comp-1.
    new_rows = indices_old[0].unsqueeze(1) * n_comp + comp.unsqueeze(0)  # shape: (nnz, n_comp)
    new_cols = indices_old[1].unsqueeze(1) * n_comp + comp.unsqueeze(0)  # shape: (nnz, n_comp)
    new_vals = values_old.unsqueeze(1).expand(nnz, n_comp)  # shape: (nnz, n_comp)

    # Flatten these arrays to create the COO indices for A_new.
    new_rows = new_rows.reshape(-1)  # shape: (nnz * n_comp,)
    new_cols = new_cols.reshape(-1)
    new_vals = new_vals.reshape(-1)

    new_indices = torch.stack([new_rows, new_cols], dim=0)  # shape: (2, nnz * n_comp)
    new_size = (M * n_comp, N * n_comp)

    A_new = torch.sparse_coo_tensor(new_indices, new_vals, size=new_size, device=device)
    return A_new


class FVMEdgeInfo:
    device: str
    n_edges: int
    n_cells: int
    n_component: int

    # Shared
    edge_len: torch.Tensor  # shape = (n_edges)
    normals: torch.Tensor  # shape = (n_edges, 2)
    # Main mesh
    n_edges_m: int
    #tri_to_edge: torch.Tensor  # shape = (n_cells, 3)
    edge_to_tri_main: torch.Tensor  # shape = [n_edges_m, 2], ordered so triangle parallel to edge normal comes last, antiparallel first.
    edge_to_tri_w: torch.Tensor  # shape = [n_edges_m, 2]
    cell_dist: torch.Tensor  # shape = (n_edges_m)

    # Boundary condition
    n_edges_bc: int             # Number of boundary edges
    bc_edge_mask: torch.Tensor  # shape = (n_edges)
    normals_bc: torch.Tensor  # shape = (n_edges_bc, 2)
    edge_to_tri_bc: torch.Tensor  # shape = (n_edges_bc)
    # dirich_mask: torch.Tensor # shape = (n_edges_bc, n_component)
    # neumann_mask: torch.Tensor # shape = (n_edges_bc, n_component)
    # dirich_val: torch.Tensor # shape: Us[dirich_mask] = dirich_val
    # neumann_val: torch.Tensor # shape: Us[neumann_mask] = neumann_val

    # Gradients
    G_mats: list[torch.Tensor]  # shape = [2](n_cells, n_cells)  Gradient matrix for every cell
    cell_disps: torch.Tensor  # shape = (n_edges_m, 2)     Displacement vector between cell centroids, for every edge_main
    edge_dists_bc: torch.Tensor  # shape = (n_bc_edges, 3)     Distance between cell centroids, for every edge_bc

    # Temporary Variables
    cell_grads: torch.Tensor  # shape = (n_cells, 2, N_component)  Gradient of cell values
    grad_faces_n: torch.Tensor  # shape = (n_edges, N_component)  n . grad(u) on faces
    U_face: torch.Tensor  # shape = (n_edges, N_component)  Face values

    def __init__(self, mesh: FVMMesh, n_comp, bc_tags, device="cpu"):
        self.device = device
        self.n_edges = mesh.n_edges
        self.n_cells = mesh.n_cells
        self.n_component = n_comp

        self.normals_main = mesh.normals_main.to(device)
        self.edge_to_tri_main = mesh.edge_to_tri_main.to(device)
        self.edge_to_tri_w = mesh.edge_to_tri_w_main.to(device)
        #self.tri_to_edge = mesh.tri_to_edge.to(device)

        self.edge_to_tri_bc = mesh.edge_to_tri_bc.to(device)
        self.bc_edge_mask = mesh.bc_edge_mask.to(device)
        self.normals_bc = mesh.normals_bc.to(device)

        (cell_disps, edge_dists_bc, G_mats) = mesh.cell_grad_stuff
        self.cell_disps = cell_disps.to(device)
        self.edge_dists_bc = edge_dists_bc.to(device).unsqueeze(-1).expand(-1, self.n_component)
        self.G_mats = []
        for G in G_mats:
            self.G_mats.append(G.to(device))

        self.cell_dist = torch.norm(self.cell_disps, dim=1).to(device)
        self.normals = mesh.normals.to(device)
        self.edge_len = torch.norm(self.normals, dim=1).to(device)
        # print(self.cell_dist.shape)
        # print(self.cell_dist.min())
        # exit(9)
        self.bc_tags = bc_tags # {edge_num: bc_tag}
        self._init_bc(bc_tags)

        self._build_spm_face_vals()
        self._build_spm_face_grads()


    def _build_spm_face_grads(self):
        n_edges = self.edge_to_tri_main.shape[0]  # number of faces (edges)
        n_cells = self.n_cells
        """ Main faces"""
        # For each face, we have two contributions.
        # Create row indices: each face i gives two rows (one per contribution).
        rows = torch.arange(n_edges, device=self.device).repeat_interleave(2)

        # Flatten the cell indices from self.edge_to_tri_main.
        cols = self.edge_to_tri_main.reshape(-1)

        # We want, for each face i, to assign:
        #   - For the first cell (cols entry from self.edge_to_tri_main[i, 0]): -1/d_i
        #   - For the second cell (cols entry from self.edge_to_tri_main[i, 1]): +1/d_i
        #
        # To do this, we first repeat the cell distances for each face:
        cell_dist_rep = self.cell_dist.repeat_interleave(2)  # shape (2*n_edges,)
        # Create a vector with the appropriate signs: first -1 then +1 for each face.
        face_signs = torch.tensor([-1, 1], device=self.device, dtype=torch.float32).repeat(n_edges)
        # Now compute the nonzero values.
        vals = face_signs / cell_dist_rep

        # Build the sparse matrix A_face.
        A_face = torch.sparse_coo_tensor(
            torch.stack([rows, cols]),
            vals,
            size=(n_edges, n_cells))

        A_face_grad_main = lift_sparse_matrix(A_face, self.n_component)

        """ Boundary faces """
        n_bc = self.n_edges_bc  # number of boundary edges
        n_comp = self.n_component  # number of components
        n_cells = self.n_cells
        # --- Prepare flattened indices for boundary rows ---
        # Each boundary edge gives n_comp rows.
        bc_rows = torch.arange(n_bc, device=self.device).unsqueeze(1).expand(n_bc, n_comp).reshape(-1)
        # Also record the component index for each entry.
        comp_idx = torch.arange(n_comp, device=self.device).unsqueeze(0).expand(n_bc, n_comp).reshape(-1)

        # Flatten the condition masks.
        dirich_mask_flat = self.dirich_mask.reshape(-1)  # True where gradient BC is given as Dirichlet
        neum_mask_flat = self.neumann_mask.reshape(-1)  # True where gradient BC is Neumann
        # Identify the flattened rows corresponding to Dirichlet gradient entries, with edge index .
        dirich_rows = torch.nonzero(dirich_mask_flat, as_tuple=False).squeeze(1)
        dirich_edge_idx = bc_rows[dirich_rows]
        # Neumann
        neum_rows_all = torch.nonzero(neum_mask_flat, as_tuple=False).squeeze(1)
        neum_comp = comp_idx[neum_rows_all]

        # --- Build the sparse matrix A_grad_bc ---
        # For Dirichlet entries, we want:
        #   coefficient = -1 / edge_dists_bc[edge]  at the column corresponding to
        #   cell = self.edge_to_tri_bc[edge] and component c.
        # For boundary edge i and component c, the cell value is at: col = self.edge_to_tri_bc[i] * n_comp + c
        cols = self.edge_to_tri_bc[dirich_edge_idx] * n_comp + comp_idx[dirich_rows]
        # The coefficient for each Dirichlet entry is -1/edge_dists_bc (for the corresponding boundary edge).
        vals = -1.0 / self.edge_dists_bc[dirich_edge_idx, 0]  # shape: (n_dirich_entries,)
        # The size of the lifted matrix is (n_bc*n_comp, n_cells*n_comp)
        size_grad = (n_bc * n_comp, n_cells * n_comp)
        indices = torch.stack([dirich_rows, cols], dim=0)
        A_grad_bc = torch.sparse_coo_tensor(indices, vals, size=size_grad)

        # --- Build the offset vector b_grad_bc ---
        # For Dirichlet entries:
        #   b = (dirich_val)/edge_dists_bc (applied componentwise)
        # For Neumann entries:
        #   b = neumann_val (applied componentwise)
        b_grad = torch.zeros(n_bc * n_comp, dtype=torch.float32, device=self.device)
        # Handle Dirichlet gradient entries:
        b_grad[dirich_rows] = self.dirich_val / self.edge_dists_bc[dirich_edge_idx, 0]
        # Handle Neumann entries:
        b_grad[neum_rows_all] = self.neumann_val[neum_comp]

        self.A_face_grad_bc = A_grad_bc
        self.b_face_grad_bc = b_grad

        self.A_face_grad, self.b_face_grad = combine_edge_operators(A_face_grad_main, A_grad_bc, b_grad, self.bc_edge_mask, self.n_edges, self.n_cells, self.n_component, self.device)


    def _build_spm_face_vals(self):
        """ Compute edge values using sparse matrix multiplication, including BC. """
        n_comp = self.n_component

        """ Sparse matrix for main values """
        # For each main edge, we have 2 contributions.
        # Build the row indices: each edge i appears twice (once for each cell).
        rows = torch.arange(self.n_edges_m, device=self.device).repeat_interleave(2)
        # Flatten the cell indices and weights.
        cols = self.edge_to_tri_main.reshape(-1)
        vals = self.edge_to_tri_w.reshape(-1)
        # Create the sparse matrix. Its shape is (n_edges_main, n_cells)
        A_main = torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, size=(self.n_edges_m, self.n_cells))
        A_main = lift_sparse_matrix(A_main, n_comp)

        """ Sparse matrix for boundary conditions"""
        n_bc = self.n_edges_bc  # number of boundary edges
        # --- Build indices for the flattened boundary condition rows ---
        # Each boundary edge i gives rise to n_comp rows.
        bc_rows = torch.arange(n_bc, device=self.device).unsqueeze(1).expand(n_bc, n_comp).reshape(-1)
        # For each row we also need the corresponding component index.
        comp_idx = torch.arange(n_comp, device=self.device).unsqueeze(0).expand(n_bc, n_comp).reshape(-1)
        # Flatten the BC-type masks: each is now a vector of length (n_bc * n_comp)
        dirich_mask = self.dirich_mask.reshape(-1)
        neum_mask = self.neumann_mask.reshape(-1)

        # --- Build the sparse matrix A_bc ---
        # For a Neumann entry (i,c), we want to pick the cell value from (Dirichlet rows remain zero):
        #   column = self.edge_to_tri_bc[i] * n_comp + c
        neum_rows = torch.nonzero(neum_mask, as_tuple=False).squeeze(1)
        # For each such row, recover the corresponding boundary edge index:
        edge_idx_for_neum = bc_rows[neum_rows]
        # And the flattened column index is:
        cols = self.edge_to_tri_bc[edge_idx_for_neum] * n_comp + comp_idx[neum_rows]
        rows = neum_rows  # These are the row indices in the flattened BC vector.
        vals = torch.ones_like(rows, dtype=torch.float32, device=self.device)
        # The sparse matrix has shape (n_bc * n_comp, n_cells * n_comp)
        size_A = (n_bc * n_comp, self.n_cells * n_comp)
        indices = torch.stack([rows, cols], dim=0)
        A_bc = torch.sparse_coo_tensor(indices, vals, size=size_A)

        # --- Build the offset vector b ---
        # b will have one entry for each flattened boundary condition.
        b = torch.zeros(n_bc * n_comp, dtype=torch.float32, device=self.device)
        # For Dirichlet entries, b should be the prescribed value.
        dirich_rows = torch.nonzero(dirich_mask, as_tuple=False).squeeze(1)
        b[dirich_rows] = self.dirich_val
        # For Neumann entries, b is the offset computed from the edge distance.
        neum_comp = comp_idx[neum_rows]
        b[neum_rows] = self.neumann_val[neum_comp] / self.edge_dists_bc[self.neumann_mask]

        self.A_face_val, self.b_face_val = combine_edge_operators(A_main, A_bc, b, self.bc_edge_mask, self.n_edges, self.n_cells, self.n_component, self.device)


    def _init_bc(self, bc_tags: dict[int, Edge]):
        self.n_edges_m = self.n_edges - self.bc_edge_mask.sum().item()
        self.n_edges_bc = self.bc_edge_mask.sum().item()

        dirich_mask, neumann_mask = [], []
        dirich_val, neumann_val = [], []
        for bc_idx, e_type in bc_tags.items():
            # print(f'{e_type = }')
            # print(e_type.dirichlet())

            dirich_mask.append(e_type.dirichlet())
            neumann_mask.append(e_type.neumann())
            dirich_val.append(e_type.U)
            neumann_val.append(e_type.dUdn)

        self.dirich_mask, self.neumann_mask = torch.tensor(dirich_mask, device=self.device), torch.tensor(neumann_mask, device=self.device)
        dirich_val, neumann_val = torch.tensor(dirich_val, dtype=torch.float32, device=self.device), torch.tensor(neumann_val, dtype=torch.float32, device=self.device)
        self.dirich_val = dirich_val[dirich_mask]
        self.neumann_val = neumann_val[neumann_mask]

        assert self.dirich_mask.shape[0] == self.bc_edge_mask.sum(), f'Wrong mask shape'
        assert self.neumann_mask.shape[0] == self.bc_edge_mask.sum(), f'Wrong mask shape'

        # bc_indices = torch.nonzero(self.bc_edge_mask, as_tuple=False).squeeze()
        self.dirich_all = torch.zeros(self.n_edges, self.n_component, dtype=torch.bool, device=self.device)
        self.dirich_all[self.bc_edge_mask] = self.dirich_mask
        self.neum_all = torch.zeros(self.n_edges, self.n_component, dtype=torch.bool, device=self.device)
        self.neum_all[self.bc_edge_mask] = self.neumann_mask

        # self.neumann_idx = bc_indices[self.neumann_mask]


    def precompute_shared(self, Us):
        """ Precompute shared values that are used multiple times later """
        self.cell_grads = self._cell_grads(Us)
        self.grad_faces_n = self._face_grads(Us)
        self.U_face = self._face_vals(Us)

    def _cell_grads(self, Us):
        """ Vectorised gradient computation
            Gradient = G @ (u_neigh - u_cell)
            Us.shape = (n_cells, N_component)
            Returns: Gradient matrix of shape (n_cells, 2, N_component)
        """
        grad_x = torch.sparse.mm(self.G_mats[0], Us)  # Shape: [n_cells, N_component]
        grad_y = torch.sparse.mm(self.G_mats[1], Us)  # Shape: [n_cells, N_component]
        cell_grads = torch.stack([grad_x, grad_y], dim=1)  # Shape: [n_cells, 2, N_component]

        self.done_cell=True
        return cell_grads

    def _face_grads(self, Us):
        """ n . grad(U) on faces.
            Us.shape = (n_cells, N_component)
            Returns: shape = [n_edges, N_component]

            Non-orthogonal correction: n . grad(U)_f = C du + (n - C d) . grad(U)_f
            NOTE: Must be called after _cell_grads() to ensure up to date cell_grads
        """
        dUdn_face = torch.empty((self.n_edges, self.n_component), device=self.device)

        # # # On faces
        # U_centroid = Us[self.edge_to_tri_main]      # shape = [n_edges, 2, N_component]
        # dU = U_centroid[:, 1] - U_centroid[:, 0]
        #
        # # normals_hat = self.normals_main / torch.norm(self.normals_main, dim=1).unsqueeze(-1)
        # # C = 1 / (normals_hat * self.cell_disps).sum(dim=1, keepdim=True)
        # #
        # # grad_impl = C * dU
        # # """ NEw - cell corrected """
        # # # Interpolate cell gradients to face
        # # grad_U_main = self.cell_grads[self.edge_to_tri_main]  # shape = [n_edges, 2, d_dims=2, N_component]
        # # w = self.edge_to_tri_w.unsqueeze(-1).unsqueeze(-1)  # shape: [n_edges, 2, 1, 1]
        # # grad_U_face = (w * grad_U_main).sum(dim=1)  # shape: [n_edges, 2, n_component]
        # #
        # # corr_expl = (normals_hat - C * self.cell_disps)#.unsqueeze(1) * grad_U_face
        # # grad_expl = (corr_expl.unsqueeze(-1) * grad_U_face).sum(dim=1)       # shape = [n_edges, N_component]
        # #
        # # dUdn_face_m_new = grad_impl + grad_expl
        # # dUdn_face[~self.bc_edge_mask] = dUdn_face_m_new
        #
        # """ OLD """
        # dUdn_face_m = dU / self.cell_dist.unsqueeze(-1)       # shape = [n_edges, N_component]
        # dUdn_face[~self.bc_edge_mask] = dUdn_face_m
        #
        # On boundary. Either u or du/dn is given
        u_centroid_bc = Us[self.edge_to_tri_bc]  # shape = [n_bc_edges, N_component]
        # Dirichlet: n.grad(u) = 1/d * (u_bc - u)
        U_cent_bc_dir = u_centroid_bc[self.dirich_mask]     # shape = [n_dirich_edges]
        edge_dists = self.edge_dists_bc[self.dirich_mask]    # shape = [n_dirich_edges]
        dudn_face_bc_dir = (self.dirich_val - U_cent_bc_dir) / edge_dists
        dUdn_face[self.dirich_all] = dudn_face_bc_dir
        # Neumann: n.grad(u) = du/dn
        dUdn_face[self.neum_all] = self.neumann_val


        assert self.done_cell, f'Cell grads not computed'

        # """ SPARSE"""
        # Us_flat = Us.flatten()
        # dUdn_face_flat = torch.mv(self.A_face_grad, Us_flat) + self.b_face_grad
        # dUdn_face = dUdn_face_flat.reshape(self.n_edges, self.n_component)
        #
        # self.done_cell = None # Reset
        return dUdn_face

    def _face_vals(self, Us):
        """ U_face, with linear interpolation """
        # # Main edges
        # U_face = torch.empty((self.n_edges, self.n_component), device=self.device)
        # #Weighted linear interpolation of two cell values
        # U_centroid = Us[self.edge_to_tri_main]  # [n_edges_m, 2, n_component]
        # w = self.edge_to_tri_w.unsqueeze(-1)  # shape: [n_edges, 2, 1]
        # U_lin_main = (w * U_centroid).sum(dim=1)  # shape: [n_edges, n_component]
        # U_face[~self.bc_edge_mask] = U_lin_main
        #
        # # Boundary edges
        # u_centroid_bc = Us[self.edge_to_tri_bc] # shape = [n_bc_edges, N_component]
        # # Dirichlet
        # U_face[self.dirich_all] = self.dirich_val
        # # Neumann
        # U_cent_bc_neum = u_centroid_bc[self.neumann_mask]        # shape = [n_neum_edges]
        # U_face_neum = U_cent_bc_neum + self.neumann_val / self.edge_dists_bc[self.neumann_mask]
        # U_face[self.neum_all] = U_face_neum

        # Vx = Us[:, 0]
        # v_min = Vx.min()
        # print(torch.where(Vx == v_min))
        # print(torch.where(U_face == v_min))
        # Main edges

        Us_flat = Us.flatten()
        U_face_flat = torch.mv(self.A_face_val, Us_flat) + self.b_face_val
        U_face = U_face_flat.reshape(self.n_edges, self.n_component)
        return U_face
