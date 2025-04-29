import torch


def to_csr(A: torch.Tensor, device):
    """ Convert a dense matrix to sparse CSR format """
    if A.layout != torch.sparse_csr:
        A = A.to_sparse_csr()

    A = A.to(device)
    return torch.sparse_csr_tensor(A.crow_indices().to(torch.int32), A.col_indices().to(torch.int32), A.values(), size=A.size(), device=device)



def create_insertion_matrix(num_blocks, full_block_size, selected_indices, device=None, dtype=torch.float32):
    """
    Instead of fluxes[:, idxs] = A, use fluxes = S @ A.flatten()

    Create a sparse matrix S that maps a flattened tensor with shape
      (num_blocks * len(selected_indices))
    to a flattened tensor with shape
      (num_blocks * full_block_size)
    by scattering the values into positions determined by selected_indices for each block.

    For each block i and for each local index j (with v = selected_indices[j]),
    set:
        S[i * full_block_size + v,  i * len(selected_indices) + j] = 1.

    Args:
        num_blocks (int): Number of blocks (e.g. n_edges).
        full_block_size (int): Size of the full block (e.g. n_component).
        selected_indices (list or iterable): Indices within each block where values should be inserted.
        device (torch.device, optional): Device for the resulting tensor.
        dtype (torch.dtype, optional): Data type for the values.

    Returns:
        torch.Tensor: A sparse matrix of shape (num_blocks * full_block_size, num_blocks * len(selected_indices)).
    """
    num_selected = len(selected_indices)
    total_rows = num_blocks * full_block_size
    total_cols = num_blocks * num_selected

    row_indices = []
    col_indices = []
    values = []

    for block in range(num_blocks):
        for j, v in enumerate(selected_indices):
            row = block * full_block_size + v
            col = block * num_selected + j
            row_indices.append(row)
            col_indices.append(col)
            values.append(1.0)

    indices = torch.tensor([row_indices, col_indices], dtype=torch.long, device=device)
    values = torch.tensor(values, dtype=dtype, device=device)
    S = torch.sparse_coo_tensor(indices, values, (total_rows, total_cols))
    return S


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


# def create_selection_matrix(num_blocks, block_size, selected_indices, device=None, dtype=torch.float32):
#     """
#     Create a selection matrix that extracts specified indices from each block of a flattened tensor.
#
#     Given a flattened tensor composed of num_blocks blocks (each of length block_size),
#     this function builds a selection matrix E such that:
#
#         E @ x == x.view(num_blocks, block_size)[:, selected_indices]
#
#     The resulting matrix E has shape (num_blocks * len(selected_indices), num_blocks * block_size).
#
#     Args:
#         num_blocks (int): The number of blocks in the flattened tensor.
#         block_size (int): The size of each block.
#         selected_indices (list or 1D tensor): Indices to select from each block.
#             Each value must satisfy 0 <= index < block_size.
#         device (torch.device, optional): The device on which to create the tensor.
#         dtype (torch.dtype, optional): The data type of the resulting tensor.
#
#     Returns:
#         torch.Tensor: The selection matrix of shape (num_blocks * len(selected_indices), num_blocks * block_size).
#     """
#     selected_indices = list(selected_indices)  # ensure it's a list
#     num_selected = len(selected_indices)
#     total_rows = num_blocks * num_selected
#     total_cols = num_blocks * block_size
#     E = torch.zeros(total_rows, total_cols, device=device, dtype=dtype)
#
#     for block in range(num_blocks):
#         for j, sel in enumerate(selected_indices):
#             row = block * num_selected + j
#             col = block * block_size + sel
#             E[row, col] = 1.0
#     return E
def create_selection_matrix(n_blocks, block_size, selected_dims, weights=None):
    """
    Constructs a sparse selection matrix A that selects (and optionally weights) entries
    from a block-structured vector.

    The resulting matrix A has shape (n_blocks * len(selected_dims), n_blocks * block_size)
    so that for each block i and for each selected dimension index r:

        A[i * len(selected_dims) + r, i * block_size + selected_dims[r]] = weight
          (or 1 if weights is None)

    This can be used, for example, to represent an operation like:

        visc.flatten() = A * E_props.grad_faces_n.flatten()

    where each block corresponds to an edge and selected_dims are the columns (dimensions)
    chosen from each block.

    Args:
        n_blocks (int): Number of blocks (e.g., m, the number of edges).
        block_size (int): The size of each block (e.g., p, the total number of components).
        selected_dims (list or 1D tensor): The indices to select from each block.
        weights (Tensor, optional): A tensor of shape (n_blocks, len(selected_dims)) containing
                                    weights for each selected entry. If provided, these values are
                                    used as the nonzero entries in A. Defaults to None (all ones).

    Returns:
        torch.sparse.FloatTensor: The sparse selection matrix A.
    """
    k = len(selected_dims)
    rows = []
    cols = []
    vals = []

    # Loop over each block and each selected index.
    for i in range(n_blocks):
        for r, d in enumerate(selected_dims):
            rows.append(i * k + r)
            cols.append(i * block_size + int(d))
            if weights is not None:
                vals.append(weights[i, r].item())
            else:
                vals.append(1.0)

    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.tensor(vals, dtype=torch.float32)

    # Construct the sparse matrix of shape (n_blocks * k, n_blocks * block_size)
    A = torch.sparse_coo_tensor(indices, values, size=(n_blocks * k, n_blocks * block_size))
    return A


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

    """# Get the COO indices and values for the two operators.
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
    b_all = torch.tensor(b_all_list, dtype=A_main_values.dtype, device=device)"""

    # Assume the following inputs are given:
    # A_main: sparse COO tensor of shape (n_edges_m*n_comp, n_cells*n_comp)
    # A_bc: sparse COO tensor of shape (n_edges_bc*n_comp, n_cells*n_comp)
    # b_bc: tensor of shape (n_edges_bc*n_comp,)
    # bc_edge_mask: Boolean tensor of shape (n_edges,), where True indicates a boundary edge.
    # n_edges, n_cells, n_comp, device are given.

    # First, compute the mapping for global main and boundary edges.
    # The ordering of the local operators corresponds to the order of the global edges.
    main_edge_global_indices = torch.where(~bc_edge_mask)[0]  # shape: (n_main_edges,)
    bc_edge_global_indices = torch.where(bc_edge_mask)[0]  # shape: (n_bc_edges,)

    A_main_indices = A_main._indices()  # shape: (2, L_main)
    A_main_values = A_main._values()  # shape: (L_main,)
    A_bc_indices = A_bc._indices()  # shape: (2, L_bc)
    A_bc_values = A_bc._values()  # shape: (L_bc,)

    # --- Process A_main ---
    # For each local row in A_main, determine its main edge index and component:
    local_rows_main = A_main_indices[0, :]  # local row indices in A_main (range: 0 to n_edges_m*n_comp - 1)
    j_main = local_rows_main // n_comp  # index into main_edge_global_indices
    c_main = local_rows_main % n_comp  # component index

    # Map to global row index: for main edges the global row is (global_edge_index * n_comp + component)
    global_rows_main = main_edge_global_indices[j_main] * n_comp + c_main
    global_cols_main = A_main_indices[1, :]


    # --- Process A_bc ---
    local_rows_bc = A_bc_indices[0, :]  # local row indices in A_bc (range: 0 to n_edges_bc*n_comp - 1)
    j_bc = local_rows_bc // n_comp  # index into bc_edge_global_indices
    c_bc = local_rows_bc % n_comp  # component index

    global_rows_bc = bc_edge_global_indices[j_bc] * n_comp + c_bc
    global_cols_bc = A_bc_indices[1, :]

    # --- Combine the main and boundary contributions ---
    global_rows = torch.cat([global_rows_main, global_rows_bc], dim=0)
    global_cols = torch.cat([global_cols_main, global_cols_bc], dim=0)
    global_vals = torch.cat([A_main_values, A_bc_values], dim=0)
    global_indices = torch.stack([global_rows, global_cols], dim=0)

    # Build the global sparse operator of shape (n_edges*n_comp, n_cells*n_comp).
    A_all = torch.sparse_coo_tensor(global_indices, global_vals,
                                    size=(n_edges * n_comp, n_cells * n_comp),
                                    device=device, dtype=A_main.dtype).coalesce()

    # --- Build the global offset vector b_all ---
    b_all = torch.zeros(n_edges * n_comp, device=device, dtype=b_bc.dtype)
    # For the boundary rows, compute the global row indices similarly.
    # Create a vector for the local rows in the boundary operator.
    r_bc = torch.arange(b_bc.numel(), device=device)
    j_bc_for_b = r_bc // n_comp  # which boundary edge block each entry belongs to
    c_bc_for_b = r_bc % n_comp  # component within the block

    global_b_rows = bc_edge_global_indices[j_bc_for_b] * n_comp + c_bc_for_b
    b_all[global_b_rows] = b_bc


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

    # # For now, we don't need rho entries.
    # rho_mask = (new_indices[0] % 3 == 2)
    # new_indices = new_indices[:, ~rho_mask]
    # new_vals = new_vals[~rho_mask]

    A_new = torch.sparse_coo_tensor(new_indices, new_vals, size=new_size, device=device).coalesce()

    return A_new


