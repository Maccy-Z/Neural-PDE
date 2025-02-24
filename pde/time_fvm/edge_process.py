import torch
from pde.time_fvm.fvm_mesh import FVMMesh
from pde.graph_grid.fvm_store import Edge
from cprint import c_print
import torch


def invert_selection_matrix(num_blocks, block_size, selected_indices, device=None, dtype=torch.float32):
    """
    Create a matrix that "inverts" the selection performed by create_selection_matrix().

    Given a flattened vector of shape (num_blocks * len(selected_indices)),
    this function creates a sparse matrix that maps it to a flattened vector of shape
    (num_blocks * block_size) by inserting each block’s values into the positions given by selected_indices.

    In other words, if S = create_selection_matrix(num_blocks, block_size, selected_indices) extracts
    the values, then this function returns a matrix S_inv such that:

            x_extended = torch.zeros(num_blocks, block_size)
            x_extended[:, selected_indices] = x
        OR:
            S_inv @ (S @ x) = x_extended

    where x_extended is the larger vector with the selected entries inserted into positions specified by selected_indices.

    Args:
        num_blocks (int): Number of blocks.
        block_size (int): Size of the full block.
        selected_indices (list or iterable): Indices within each block that were selected.
        device (torch.device, optional): Device to create the matrix on.
        dtype (torch.dtype, optional): Data type for the matrix.

    Returns:
        torch.Tensor: A sparse matrix of shape
          (num_blocks * block_size, num_blocks * len(selected_indices))
    """
    selected_indices = list(selected_indices)
    num_selected = len(selected_indices)
    total_rows = num_blocks * block_size
    total_cols = num_blocks * num_selected

    row_indices = []
    col_indices = []
    values = []

    for block in range(num_blocks):
        for j, idx in enumerate(selected_indices):
            row = block * block_size + idx
            col = block * num_selected + j
            row_indices.append(row)
            col_indices.append(col)
            values.append(1.0)

    indices = torch.tensor([row_indices, col_indices], dtype=torch.long, device=device)
    values = torch.tensor(values, dtype=dtype, device=device)
    S_inv = torch.sparse_coo_tensor(indices, values, (total_rows, total_cols)).to_dense()
    return S_inv


def create_selection_matrix(num_blocks, block_size, selected_indices, device=None, dtype=torch.float32):
    """
    Create a selection matrix that extracts specified indices from each block of a flattened tensor.

    Given a flattened tensor composed of num_blocks blocks (each of length block_size),
    this function builds a selection matrix E such that:

        E @ x == x.view(num_blocks, block_size)[:, selected_indices]

    The resulting matrix E has shape (num_blocks * len(selected_indices), num_blocks * block_size).

    Args:
        num_blocks (int): The number of blocks in the flattened tensor.
        block_size (int): The size of each block.
        selected_indices (list or 1D tensor): Indices to select from each block.
            Each value must satisfy 0 <= index < block_size.
        device (torch.device, optional): The device on which to create the tensor.
        dtype (torch.dtype, optional): The data type of the resulting tensor.

    Returns:
        torch.Tensor: The selection matrix of shape (num_blocks * len(selected_indices), num_blocks * block_size).
    """
    selected_indices = list(selected_indices)  # ensure it's a list
    num_selected = len(selected_indices)
    total_rows = num_blocks * num_selected
    total_cols = num_blocks * block_size
    E = torch.zeros(total_rows, total_cols, device=device, dtype=dtype)

    for block in range(num_blocks):
        for j, sel in enumerate(selected_indices):
            row = block * num_selected + j
            col = block * block_size + sel
            E[row, col] = 1.0
    return E


def create_block_diagonal(normals):
    """
    Create a block diagonal matrix D that has, for each block i,
    a block of shape (2, 1) equal to normals[i, :].

    Args:
        normals (torch.Tensor): Tensor of shape (n_edges, 2).

    Returns:
        torch.Tensor: A block diagonal matrix of shape (n_edges*2, n_edges).
    """
    n_edges = normals.shape[0]
    D = torch.zeros(n_edges * 2, n_edges, dtype=normals.dtype, device=normals.device)
    for i in range(n_edges):
        # Place normals[i, :] as a column in the i-th block.
        D[2 * i:2 * i + 2, i] = normals[i, :]
    return D


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
    edge_to_tri_main: torch.Tensor  # shape = [n_edges_m, 2], ordered so triangle parallel to edge normal comes last, antiparallel first.
    edge_to_tri_w: torch.Tensor  # shape = [n_edges_m, 2]
    cell_dist: torch.Tensor  # shape = (n_edges_m)
    tri_edge_signs: torch.Tensor  # shape = (3 * n_cells)
    tri_to_edge: torch.Tensor  # shape = (3 * n_cells)
    cent_to_edge_disp: torch.Tensor  # shape = (n_cells, 3, 2)  # Displacement vector between cell centroids, for every edge

    # Boundary condition
    n_edges_bc: int             # Number of boundary edges
    bc_edge_mask: torch.Tensor  # shape = (n_edges)
    #normals_bc: torch.Tensor  # shape = (n_edges_bc, 2)
    edge_to_tri_bc: torch.Tensor  # shape = (n_edges_bc)

    # Gradients
    G_mats: list[torch.Tensor]  # shape = [2](n_cells, n_cells)  Gradient matrix for every cell
    # cell_disps: torch.Tensor  # shape = (n_edges_m, 2)     Displacement vector between cell centroids, for every edge_main
    edge_dists_bc: torch.Tensor  # shape = (n_bc_edges, 3)     Distance between cell centroids, for every edge_bc
    neigh_combine: torch.Tensor # shape = (n_cell, 2). Used for masking neighbors of cell incl boundary edges, in format [Us, U_face_bc]
    # disps_combine: torch.Tensor # shape = [n_cells, n_neigh=3, 2]. Used for masking neighbors of cell incl boundary edges.

    # Temporary Variables
    cell_grads: torch.Tensor  # shape = (n_cells, 2, N_component)  Gradient of cell values
    grad_faces_n: torch.Tensor  # shape = (n_edges, N_component)  n . grad(u) on faces
    U_face: torch.Tensor  # shape = (n_edges, 2, N_component)  Face values, on both sides of the face
    # Primitive face variables
    Vs_faces: torch.Tensor  # shape = (n_edges, 2, 2)  Face values
    rho_faces: torch.Tensor  # shape = (n_edges, 2, 1)  Face values

    def __init__(self, mesh: FVMMesh, n_comp, bc_tags, device="cpu"):
        self.device = device
        self.mesh = mesh
        self.n_edges = mesh.n_edges
        self.n_cells = mesh.n_cells
        self.n_component = n_comp

        self.normals_main = mesh.normals_main.to(device)
        self.edge_to_tri_main = mesh.edge_to_tri_main.to(device)
        self.edge_to_tri_w = mesh.edge_to_tri_w_main.to(device)
        self.cent_to_edge_disp = mesh.cent_to_edge_disp.to(device).unsqueeze(-1)
        self.tri_edge_signs = (-self.mesh.tri_edge_signs + 1 / 2).int().view(3*self.n_cells).to(device)
        self.tri_to_edge = mesh.tri_to_edge.view(3*self.n_cells).to(device)

        self.edge_to_tri_bc = mesh.edge_to_tri_bc.to(device)
        self.bc_edge_mask = mesh.bc_edge_mask.to(device)
        #self.normals_bc = mesh.normals_bc.to(device)

        (cell_disps, edge_dists_bc, G_mats, neigh_combine, disps_combine) = mesh.cell_grad_stuff
        cell_disps = cell_disps.to(device)
        self.edge_dists_bc = edge_dists_bc.to(device).unsqueeze(-1).expand(-1, self.n_component)
        self.G_mats = []
        for G in G_mats:
            self.G_mats.append(G.to(device))
        self.neigh_combine = neigh_combine.to(device)
        self.cell_dist = torch.norm(cell_disps, dim=1).to(device)
        self.normals = mesh.normals.to(device)
        self.edge_len = torch.norm(self.normals, dim=1).to(device)

        self.bc_tags = bc_tags # {edge_num: bc_tag}
        self._init_bc(bc_tags)

        self._build_spm_face_vals()
        self._build_spm_face_grads()


    def _build_spm_face_grads(self):
        n_edges = self.edge_to_tri_main.shape[0]  # number of faces (edges)
        n_cells = self.n_cells
        n_bc = self.n_edges_bc  # number of boundary edges
        n_comp = self.n_component  # number of components

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
        """ Compute bc edge values using sparse matrix multiplication. """

        device = self.device
        n_bc = self.n_edges_bc  # number of boundary edges
        n_comp = self.n_component
        n_cells = self.n_cells

        # Total number of flattened BC rows.
        N = n_bc * n_comp

        # Create flattened indices for the boundary rows and the corresponding component.
        # Each boundary edge gives rise to n_comp rows.
        bc_rows = torch.arange(n_bc, device=device).unsqueeze(1).expand(n_bc, n_comp).reshape(-1)
        comp_idx = torch.arange(n_comp, device=device).unsqueeze(0).expand(n_bc, n_comp).reshape(-1)

        # Reshape the condition masks to a flat vector of length N.
        dirich_mask = self.dirich_mask.reshape(-1)  # For Dirichlet conditions.
        neum_mask = self.neumann_mask.reshape(-1)  # For Neumann conditions.

        # --- Build sparse matrix A ---
        # For Neumann entries, we want to extract the cell value from Us.
        # For each Neumann row, the corresponding column in Us (flattened) is given by:
        #   col = self.edge_to_tri_bc[ edge_index ] * n_comp + component
        neum_indices = torch.nonzero(neum_mask, as_tuple=False).squeeze(1)  # indices where Neumann is True.
        A_rows = neum_indices
        # bc_rows[neum_indices] gives the corresponding boundary edge for each flattened row.
        A_cols = self.edge_to_tri_bc[bc_rows[neum_indices]] * n_comp + comp_idx[neum_indices]
        A_vals = torch.ones_like(A_rows, dtype=torch.float32, device=device)

        size_A = (N, n_cells * n_comp)
        self.A_bc = torch.sparse_coo_tensor(torch.stack([A_rows, A_cols], dim=0), A_vals, size=size_A).coalesce().to_sparse_csr()

        # Build the offset vector b.
        self.b_bc = torch.empty(N, device=device, dtype=torch.float32)
        # For Dirichlet entries, the prescribed value should override any extracted value.
        self.b_bc[dirich_mask] = self.dirich_val
        # For Neumann entries, add the offset computed from the edge distance.
        # Here, we select the proper component value from self.neumann_val using comp_idx.
        self.b_bc[neum_mask] = self.neumann_val[comp_idx[neum_mask]] / self.edge_dists_bc.flatten()[neum_mask]

        """ Regrouping """
        # flat_rows = self.tri_to_edge * 2 + self.tri_edge_signs
        # cols = torch.arange(3*self.n_cells, device=self.device)
        # # All nonzero values are 1.0:
        # vals = torch.ones_like(flat_rows, dtype=torch.float32)
        # # Build the sparse selection matrix: shape (n_edges*2, n_cell_entries)
        # self.S_cells = torch.sparse_coo_tensor(
        #     torch.stack([flat_rows, cols]),
        #     vals,
        #     size=(self.n_edges * 2, 3*self.n_cells)
        # ).to_sparse_csr()


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
        self.dirich_val = dirich_val[self.dirich_mask]
        self.neumann_val = neumann_val[self.neumann_mask]

        assert self.dirich_mask.shape[0] == self.bc_edge_mask.sum(), f'Wrong mask shape'
        assert self.neumann_mask.shape[0] == self.bc_edge_mask.sum(), f'Wrong mask shape'

        # bc_indices = torch.nonzero(self.bc_edge_mask, as_tuple=False).squeeze()
        self.dirich_all = torch.zeros(self.n_edges, self.n_component, dtype=torch.bool, device=self.device)
        self.dirich_all[self.bc_edge_mask] = self.dirich_mask
        self.neum_all = torch.zeros(self.n_edges, self.n_component, dtype=torch.bool, device=self.device)
        self.neum_all[self.bc_edge_mask] = self.neumann_mask

        # self.neumann_idx = bc_indices[self.neumann_mask]
        # # Matrix for nomal dot V
        # I = torch.eye(self.n_edges, device="cuda")  # Shape (m, m)
        # normals_mat =  (I.unsqueeze(-1) * self.normals.unsqueeze(1)).reshape(self.n_edges, self.n_edges * 2)
        # self.E = create_selection_matrix(self.n_edges, 3, [0, 1], device=self.device)
        # self.normal_dot_V = normals_mat @ self.E
        # # Matrix for normal * rho
        # D = create_block_diagonal(self.normals)
        # self.E_div = create_selection_matrix(self.n_edges, 3, [2], device=self.device)
        # self.normal_rho = D @ self.E_div


    def precompute_shared(self, Us):
        """ Precompute shared values that are used multiple times later.
            Us.shape = [n_cells, n_component] """
        U_face_bc = self._bc_face_vals(Us)
        self.cell_grads = self._cell_grads(Us, U_face_bc) # shape = [n_cells, 2, n_component]
        self.grad_faces_n = self._face_grads(Us)        # shape = [n_faces, n_component]

        """ Gradient schemes """
        """ Limited B-J scheme """
        U_cent = Us.unsqueeze(1)        # shape = [n_cells, 1, n_component]
        Us_cell_face = torch.cat([Us, U_face_bc])
        Us_neigh = Us_cell_face[self.neigh_combine]  # shape = [n_cells, neigh=3, n_component]

        # Uncorrected update
        grads = self.cell_grads.unsqueeze(1)     # shape = [n_cells, 1, dims=2, n_component]
        dU = (grads * self.cent_to_edge_disp).sum(dim=2)  # shape = [n_cells, neigh=3, n_component]

        # Select limiting neighbor values and compute gradient limiter
        diff = Us_neigh - U_cent
        U_upper = torch.clamp(diff, min=0)  # positive differences (or 0 if diff is negative)
        U_lower = torch.clamp(diff, max=0)  # negative differences (or 0 if diff is positive)
        # U_upper = torch.maximum(U_cent, Us_neigh) - U_cent      # shape = [n_cells, neigh=3, n_component]
        # U_lower = torch.minimum(U_cent, Us_neigh) - U_cent
        numerator = torch.where(dU > 0, U_upper, U_lower)
        r = numerator / (dU+1e-8)
        phi = self._phi(r)                  # shape = [n_cells, neigh=3, n_component]
        Us_face = U_cent + phi * dU      # shape = [n_cells, neigh=3, n_component]

        self.U_face = torch.empty((self.n_edges, 2, self.n_component), device=self.device)
        self.U_face[self.tri_to_edge, self.tri_edge_signs] = Us_face.view(3*self.n_cells, 3)
        self.U_face[self.bc_edge_mask] = U_face_bc.unsqueeze(1)

        # U_face_flat = torch.mm(self.S_cells, Us_face.view(3*self.n_cells, 3))
        # self.U_face = U_face_flat.view(self.n_edges, 2, self.n_component)
        # self.U_face[self.bc_edge_mask] = U_face_bc.unsqueeze(1)

        self.Vs_faces = self.U_face[:, :, [0, 1]]  # shape = [n_edges, edges=2, n_comp=2]
        self.rho_faces = self.U_face[:, :, [2]]  # shape = [n_edges, edges=2, dims=1]

        """ Mean interpolation """
        # grads = self.cell_grads[self.edge_to_tri_main]
        # U_cent = Us[self.edge_to_tri_main]  # shape = [n_edges_m, 2, n_component]
        # r = self.mesh.e_c_to_e_disp_m.unsqueeze(-1).cuda() # shape = [n_edges_m, cells=2, dims=2, 1]
        # dU = (grads * r).sum(dim=2)  # shape = [n_edges_m, cells=2, n_component]
        # U_test = U_cent + dU

        # self.U_face[~self.bc_edge_mask] = U_test # torch.stack([U_r, U_l], dim=1)
        """ TESTS """
        # print()
        # # print(f'{self.cent_to_edge_disp[1087, 2].squeeze()}')
        # # print(f'{self.cent_to_edge_disp[335, 2].squeeze()}')
        # # exit(9)
        # # # print(f'{self.mesh.tri_to_edge[1087] = }')  # [2796, 1120, 1681]
        # # # print(f'{self.mesh.edge_to_tri[1120] = }') # [1087, 1088]
        # print(f'{Us[1087, 1] = }')
        # print(f'{Us[335, 1] = }')
        # # print(f'{Us[335, 2] = }')
        #
        # print(phi[1087, :, 1])
        # # print(f'{self.cell_grads[1087, :, 1] = }')
        # # print(f'{self.cell_grads[335, :, 1] = }')
        # c_print(f'{Us_neigh[1087, :, 1] = }', color="cyan")
        # c_print(f'{Us_face[1087, :, 1] = }', color="cyan")

        #exit(7)

    def _bc_face_vals(self, Us):
        """ U_face, with linear interpolation.
            return.shape: [n_edges_bc, n_component]

         """
        # # Main edges
        # # #Weighted linear interpolation of two cell values
        # # U_centroid = Us[self.edge_to_tri_main]  # [n_edges_m, 2, n_component]
        # # w = self.edge_to_tri_w.unsqueeze(-1)  # shape: [n_edges, 2, 1]
        # # U_lin_main = (w * U_centroid).sum(dim=1)  # shape: [n_edges, n_component]
        # # U_face[~self.bc_edge_mask] = U_lin_main
        # #
        # U_face = torch.empty((self.n_edges_bc, self.n_component), device=self.device)
        # # Boundary edges
        # u_centroid_bc = Us[self.edge_to_tri_bc] # shape = [n_bc_edges, N_component]
        # # Dirichlet
        # U_face[self.dirich_mask] = self.dirich_val
        # # Neumann
        # U_cent_bc_neum = u_centroid_bc[self.neumann_mask]        # shape = [n_neum_edges]
        # U_face_neum = U_cent_bc_neum + self.neumann_val / self.edge_dists_bc[self.neumann_mask]
        # U_face[self.neumann_mask] = U_face_neum


        Us_flat = Us.reshape(-1)
        # Final U_face in flattened form.
        U_face_flat = torch.mv(self.A_bc, Us_flat) + self.b_bc      # shape = [n_edges_bc * n_component]
        # Reshape back to (n_edges_bc, n_component)
        U_face = U_face_flat.view(self.n_edges_bc, self.n_component)

        return  U_face

    def _phi(self, r):
        # VENKATAKRISHNAN
        # eps = 0.3*3e-3
        # _r = r**2 + r + eps
        # phi = (_r + r) / (_r + 2)
        # phi = torch.clamp(phi, min=0, max=1.)       # shape = [n_cells, neigh=3, n_component]

        # BJ
        phi = torch.clamp(r, min=0., max=1.)       # shape = [n_cells, neigh=3, n_component]

        # Cell wide clamping
        # phi = torch.min(phi, dim=1, keepdim=True).values        # shape = [n_cells, neigh=1, n_component]
        # phi = torch.mean(phi, dim=1, keepdim=True)        # shape = [n_cells, neigh=1, n_component]

        #phi = torch.min(phi, dim=2, keepdim=True).values        # shape = [n_cells, neigh=1, n_comp=1]

        return phi

    def _cell_grads(self, Us, U_face_bc):
        """ Vectorised gradient computation
            Gradient = G @ (u_neigh - u_cell)
            Us.shape = (n_cells, N_component)
            Returns: Gradient matrix of shape (n_cells, 2, N_component)
        """
        Us_cell_face = torch.cat([Us, U_face_bc])

        grad_x = torch.sparse.mm(self.G_mats[0], Us_cell_face)  # Shape: [n_cells, N_component]
        grad_y = torch.sparse.mm(self.G_mats[1], Us_cell_face)  # Shape: [n_cells, N_component]
        cell_grads = torch.stack([grad_x, grad_y], dim=1)  # Shape: [n_cells, 2, N_component]
        # print(grad_x)
        # exit(7)
        #

        # cell_grads = cell_grads * phi_cand
        # max_grad = cell_grads[:, 0, 2].abs().max()
        # print(f'{max_grad = }')
        # print(torch.where(cell_grads.abs()==max_grad))
        return cell_grads

    def _face_grads(self, Us):
        """ n . grad(U) on faces.
            Us.shape = (n_cells, N_component)
            Returns: shape = [n_edges, N_component]

            Non-orthogonal correction: n . grad(U)_f = C du + (n - C d) . grad(U)_f
            NOTE: Must be called after _cell_grads() to ensure up to date cell_grads
        """
        #
        # On faces
        # U_centroid = Us[self.edge_to_tri_main]      # shape = [n_edges, 2, N_component]
        # dU = U_centroid[:, 1] - U_centroid[:, 0]        # shape = [n_edges, N_component]
        # print(f'{U_centroid[494] = }')
        # print(f'{dU[494] = }')
        #
        """ NEW """
        # normals_hat = self.normals_main / torch.norm(self.normals_main, dim=1).unsqueeze(-1)
        # C = 1 / (normals_hat * self.cell_disps).sum(dim=1, keepdim=True)
        #
        # grad_impl = C * dU
        # """ NEw - cell corrected """
        # # Interpolate cell gradients to face
        # grad_U_main = self.cell_grads[self.edge_to_tri_main]  # shape = [n_edges, 2, d_dims=2, N_component]
        # w = self.edge_to_tri_w.unsqueeze(-1).unsqueeze(-1)  # shape: [n_edges, 2, 1, 1]
        # grad_U_face = (w * grad_U_main).sum(dim=1)  # shape: [n_edges, 2, n_component]
        #
        # corr_expl = (normals_hat - C * self.cell_disps)#.unsqueeze(1) * grad_U_face
        # grad_expl = (corr_expl.unsqueeze(-1) * grad_U_face).sum(dim=1)       # shape = [n_edges, N_component]
        #
        # dUdn_face_m_new = grad_impl + grad_expl
        # dUdn_face[~self.bc_edge_mask] = dUdn_face_m_new
        # # # #9


        """ OLD """
        # U_centroid = Us[self.edge_to_tri_main]      # shape = [n_edges, 2, N_component]
        # dU = U_centroid[:, 1] - U_centroid[:, 0]        # shape = [n_edges, N_component]
        # dUdn_face = torch.empty((self.n_edges, self.n_component), device=self.device)
        # dUdn_face_m = dU / self.cell_dist.unsqueeze(-1)       # shape = [n_edges, N_component]
        # # print(f'{dUdn_face.shape = }, {self.bc_edge_mask.shape = }')
        # dUdn_face[~self.bc_edge_mask] = dUdn_face_m
        # #
        # # On boundary. Either u or du/dn is given
        # # Neumann: n.grad(u) = du/dn
        # dUdn_face[self.neum_all] = self.neumann_val
        #
        # # Dirichlet: n.grad(u) = 1/d * (u_bc - u)
        # u_centroid_bc = Us[self.edge_to_tri_bc]  # shape = [n_bc_edges, N_component]
        # U_cent_bc_dir = u_centroid_bc[self.dirich_mask]     # shape = [n_dirich_edges]
        # edge_dists = self.edge_dists_bc[self.dirich_mask]    # shape = [n_dirich_edges]
        # dudn_face_bc_dir = (self.dirich_val - U_cent_bc_dir) / edge_dists
        # dUdn_face[self.dirich_all] = dudn_face_bc_dir

        """ SPARSE"""
        Us_flat = Us.flatten()
        dUdn_face_flat = torch.mv(self.A_face_grad, Us_flat) + self.b_face_grad
        dUdn_face = dUdn_face_flat.reshape(self.n_edges, self.n_component)

        return dUdn_face

