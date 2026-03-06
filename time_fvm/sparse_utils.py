import torch
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.tri as tri
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_points(Xs, values, lims=None, title="", show_index=False, Xlims=None):
    Xs = Xs.cpu()
    values = values.cpu()

    if len(values.shape) == 1:
        values = values.unsqueeze(0)
        fig, axes = plt.subplots(1, 1, figsize=(12, 9))
        axes = [axes]
    else:
        n_plots = values.shape[0]
        fig, axes = plt.subplots(n_plots, 1, figsize=(8, n_plots * 4))

    # Loop over each batch
    if Xlims is None:
        Xlims = (Xs[:, 0].min(), Xs[:, 0].max()), (Xs[:, 1].min(), Xs[:, 1].max())

    for i, ax in enumerate(axes):
        ax.set_title(f"{title} - Batch {i}")
        if lims is None:
            sc = ax.scatter(Xs[:, 0], Xs[:, 1], c=values[i], cmap='viridis')
        else:
            sc = ax.scatter(Xs[:, 0], Xs[:, 1], c=values[i], cmap='viridis',
                            vmin=lims[0], vmax=lims[1])

        if show_index:
            for i, X in enumerate(Xs):
                x, y = X
                if (Xlims[0][0] <= x <= Xlims[0][1]) and (Xlims[1][0] <= y <= Xlims[1][1]):
                    ax.text(x, y, f"{i}", fontsize=8)
        fig.colorbar(sc, ax=ax)
        ax.set_aspect('equal', adjustable='box')

        ax.set_xlim(Xlims[0])
        ax.set_ylim(Xlims[1])

    # plt.tight_layout()
    plt.show()


def plot_interp_cell(Xs, values, triangles, Xlims=None, title="", edgecolors="none"):
    """
    Xs: Tensor of vertex coordinates (N x 2)
    values: Tensor of face-based values.
            If values is 1D, it's assumed to be defined on the triangulation faces.
            If 2D, each row is treated as a separate batch. shape = (B, M)
    triangles: Tensor of triangle vertex indices (M x 3).
    Xlims: Optional tuple ((xmin, xmax), (ymin, ymax)) to set the plot limits.
    title: Plot title.
    """
    # Convert to numpy arrays.
    Xs = Xs.cpu().numpy()
    values = values.cpu().numpy()
    triangles = triangles.cpu().numpy()

    # If values is 1D, expand to a batch of one.
    values = values.squeeze()
    if len(values.shape) == 1:
        values = values[None, :]

    n_plots = values.shape[0]
    if n_plots <= 3:
        fig, axes = plt.subplots(n_plots, 1, figsize=(8, n_plots * 6))
    else:
        # Use a near-square layout for larger batches.
        n_cols = int(np.ceil(np.sqrt(n_plots)))
        n_rows = int(np.ceil(n_plots / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

    axes = np.atleast_1d(axes).ravel()
    plot_axes = axes[:n_plots]
    for ax in axes[n_plots:]:
        ax.axis('off')

    # Create a triangulation from the vertex locations.
    triang = tri.Triangulation(Xs[:, 0], Xs[:, 1], triangles)

    if isinstance(title, str):
        title = [title] * n_plots

    # Determine plot limits.
    if Xlims is not None:
        xlim, ylim = Xlims
    else:
        xlim = (Xs[:, 0].min(), Xs[:, 0].max())
        ylim = (Xs[:, 1].min(), Xs[:, 1].max())

    in_region = []
    for tri_indices in triang.triangles:
        verts = np.column_stack((Xs[tri_indices, 0], Xs[tri_indices, 1]))
        if np.all((verts[:, 0] >= xlim[0]) & (verts[:, 0] <= xlim[1]) &
                  (verts[:, 1] >= ylim[0]) & (verts[:, 1] <= ylim[1])):
            in_region.append(True)
        else:
            in_region.append(False)
    in_region = np.array(in_region)

    # Create a new triangulation using only the triangles inside the region.
    new_triangles = triang.triangles[in_region]
    new_triang = tri.Triangulation(Xs[:, 0], Xs[:, 1], triangles=new_triangles)

    # Loop over each batch and plot only the triangles inside the region.
    for i, ax in enumerate(plot_axes):
        ax.set_title(f"{title[i]}")
        # Filter the face-based values for the triangles inside the region.
        new_facecolors = values[i][in_region]

        # Plot using the new triangulation and corresponding facecolors.
        tc = ax.tripcolor(new_triang, facecolors=new_facecolors, edgecolors=edgecolors,
                          cmap='viridis', shading='flat')

        # Attach a dedicated colorbar axis to keep colorbar height matched to this subplot.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.08)
        fig.colorbar(tc, cax=cax)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()


def plot_interp_vertex(Xs, values, triangles=None, Xlims=None, title="", edgecolors="none"):
    """
    Xs:     shape = [n_points, 2] Tensor of vertex coordinates
    values: shape = [n_plots, n_points]. Tensor of face-based values.
            If values is 1D, it's assumed to be defined on the triangulation faces.
            If 2D, each row is treated as a separate batch.
    Xlims: Optional tuple ((xmin, xmax), (ymin, ymax)) to set the plot limits.
    title: Plot title.
    """

    # Convert to numpy arrays.
    Xs = Xs.cpu().numpy()
    values = values.cpu().numpy()

    # If values is 1D, expand to a batch of one.
    if len(values.shape) == 1:
        values = values[None, :]
        fig, axes = plt.subplots(1, 1, figsize=(6, 4))
        axes = [axes]
    else:
        n_plots, n_points = values.shape
        if n_plots > n_points:
            print("Warning ")
            raise ValueError(f"Number of plots ({n_plots}) exceeds number of points ({n_points}).")

        fig, axes = plt.subplots(n_plots, 1, figsize=(6, n_plots * 4))
        if n_plots == 1:
            axes = [axes]

    # Make title into list[str] for each plot.
    if isinstance(title, str):
        title = [title] * len(axes)

    # Determine plot limits.
    if Xlims is not None:
        xlim, ylim = Xlims
    else:
        xlim = (Xs[:, 0].min(), Xs[:, 0].max())
        ylim = (Xs[:, 1].min(), Xs[:, 1].max())


    triang = tri.Triangulation(Xs[:, 0], Xs[:, 1], triangles)
    # Filter out triangles that are outside the specified region.
    tri_mask = []
    for tri_indices in triang.triangles:
        x_vert, y_vert = Xs[tri_indices, 0], Xs[tri_indices, 1]
        if np.all((x_vert[0] >= xlim[0]) & (x_vert[0] <= xlim[1]) &
                  (y_vert[1] >= ylim[0]) & (y_vert[1] <= ylim[1])):
            tri_mask.append(True)
        else:
            tri_mask.append(False)
    tri_mask = np.array(tri_mask)
    vert_idx = np.unique(triang.triangles[tri_mask])
    vertex_mask = np.zeros(Xs.shape[0], dtype=bool)
    vertex_mask[vert_idx] = True
    triang.set_mask(~tri_mask)

    # Loop over each batch and plot only the triangles inside the region.
    for i, ax in enumerate(axes):
        ax.set_title(f"{title[i]}")
        v = np.ma.array(values[i], mask=~vertex_mask)
        tc = ax.tripcolor(triang, v, shading='flat', cmap='viridis', edgecolors=edgecolors)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.08)
        fig.colorbar(tc, cax=cax)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()


def plot_edges(coords, edge_idx, colors=None, title="", show_index=False, lims=None, Xlims=None):
    """ Plot the edges of the mesh.
        coords.shape = (n, 2)
        edge_idx.shape = (m, 2)
        If 'color' is provided, it should be a torch tensor. In the case that
        'color' is 1D, it is assumed to be (m,) and converted to (m,1) for plotting.
    """
    # Convert inputs from torch tensors to numpy arrays.
    coords = coords.cpu().detach().numpy()
    edge_idx = edge_idx.cpu().detach().numpy()

    if colors is None:
        colors = torch.zeros(len(edge_idx))

    # If color is a 1D tensor, unsqueeze to (m, 1) and create a single subplot.
    if len(colors.shape) == 1:
        colors = colors.unsqueeze(-1)
        fig, axes = plt.subplots(1, 1, figsize=(16, 12))
        axes = [axes]
    else:
        n_plots = colors.shape[1]
        fig, axes = plt.subplots(n_plots, 1, figsize=(8, n_plots * 4))

    colormap = plt.get_cmap("viridis")
    edge_colors = colors.cpu().detach().numpy().T  # shape = (n_plots, m)

    # Extract the coordinates of each edge. Each row in 'points' is an edge defined
    # by its two endpoints (shape: (m, 2, 2)).
    points = coords[edge_idx]

    # Plot each batch (subplot).
    for i, ax in enumerate(axes):
        ax.set_aspect('equal', adjustable='box')

        # Plot each edge using the corresponding color.
        if Xlims is None:
            xmin, xmax = coords[:, 0].min(), coords[:, 0].max()
            ymin, ymax = coords[:, 1].min(), coords[:, 1].max()
        else:
            (xmin, xmax), (ymin, ymax) = Xlims

        edge_nums, edge_scalars = [], []
        for j, (edge, s) in enumerate(zip(points, edge_colors[i], strict=True)):
            midpoint = edge.mean(axis=0)
            if not ((xmin <= midpoint[0] <= xmax) and (ymin <= midpoint[1] <= ymax)):
                continue
            edge_scalars.append(s)
            edge_nums.append(j)

        edge_scalars = np.array(edge_scalars)
        min_c, max_c = edge_scalars.min(), edge_scalars.max()
        edge_scalars = (edge_scalars - min_c) / (max_c - min_c + 1e-9)

        for j, s in zip(edge_nums, edge_scalars, strict=True):
            c = colormap(s)
            edge = points[j]
            ax.plot(edge[:, 0], edge[:, 1], color=c)
            if show_index:
                midpoint = edge.mean(axis=0)
                ax.text(midpoint[0], midpoint[1], f"{j}", fontsize=8)
                ax.annotate(
                    '',  # No text
                    xy=(edge[1]),  # Arrow tip (end of the line)
                    xytext=midpoint,  # Arrow tail (start of the line)
                    arrowprops=dict(arrowstyle='->', lw=.5)
                )

            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])

        # If colors are provided, create a ScalarMappable for the colorbar.
        ax.set_title(f"{title} - Batch {i}, [min={min_c.item():.3g}, max={max_c.item():.3g}]")

        # Use the original scalar range for this batch.
        norm = plt.Normalize(vmin=min_c.item(), vmax=max_c.item())
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        # Optional: attach the actual scalar array (could also use an empty array)
        cbar = fig.colorbar(sm, ax=ax)
    plt.tight_layout()
    plt.show()


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


