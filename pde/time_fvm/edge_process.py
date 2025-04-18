import math
from cprint import c_print
import torch

from pde.time_fvm.fvm_mesh import FVMMesh
from pde.graph_grid.fvm_store import Edge
from pde.time_fvm.config_fvm import ConfigFVM
from pde.time_fvm.sparse_utils import create_insertion_matrix, lift_sparse_matrix, combine_edge_operators

# def create_insertion_matrix(num_blocks, full_block_size, selected_indices, device=None, dtype=torch.float32):
#     """
#     Instead of fluxes[:, idxs] = A, use fluxes = S @ A.flatten()
#
#     Create a sparse matrix S that maps a flattened tensor with shape
#       (num_blocks * len(selected_indices))
#     to a flattened tensor with shape
#       (num_blocks * full_block_size)
#     by scattering the values into positions determined by selected_indices for each block.
#
#     For each block i and for each local index j (with v = selected_indices[j]),
#     set:
#         S[i * full_block_size + v,  i * len(selected_indices) + j] = 1.
#
#     Args:
#         num_blocks (int): Number of blocks (e.g. n_edges).
#         full_block_size (int): Size of the full block (e.g. n_component).
#         selected_indices (list or iterable): Indices within each block where values should be inserted.
#         device (torch.device, optional): Device for the resulting tensor.
#         dtype (torch.dtype, optional): Data type for the values.
#
#     Returns:
#         torch.Tensor: A sparse matrix of shape (num_blocks * full_block_size, num_blocks * len(selected_indices)).
#     """
#     num_selected = len(selected_indices)
#     total_rows = num_blocks * full_block_size
#     total_cols = num_blocks * num_selected
#
#     row_indices = []
#     col_indices = []
#     values = []
#
#     for block in range(num_blocks):
#         for j, v in enumerate(selected_indices):
#             row = block * full_block_size + v
#             col = block * num_selected + j
#             row_indices.append(row)
#             col_indices.append(col)
#             values.append(1.0)
#
#     indices = torch.tensor([row_indices, col_indices], dtype=torch.long, device=device)
#     values = torch.tensor(values, dtype=dtype, device=device)
#     S = torch.sparse_coo_tensor(indices, values, (total_rows, total_cols))
#     return S
#
#
# def invert_selection_matrix(num_blocks, block_size, selected_indices, device=None, dtype=torch.float32):
#     """
#     Create a matrix that "inverts" the selection performed by create_selection_matrix().
#
#     Given a flattened vector of shape (num_blocks * len(selected_indices)),
#     this function creates a sparse matrix that maps it to a flattened vector of shape
#     (num_blocks * block_size) by inserting each block’s values into the positions given by selected_indices.
#
#     In other words, if S = create_selection_matrix(num_blocks, block_size, selected_indices) extracts
#     the values, then this function returns a matrix S_inv such that:
#
#             x_extended = torch.zeros(num_blocks, block_size)
#             x_extended[:, selected_indices] = x
#         OR:
#             S_inv @ (S @ x) = x_extended
#
#     where x_extended is the larger vector with the selected entries inserted into positions specified by selected_indices.
#
#     Args:
#         num_blocks (int): Number of blocks.
#         block_size (int): Size of the full block.
#         selected_indices (list or iterable): Indices within each block that were selected.
#         device (torch.device, optional): Device to create the matrix on.
#         dtype (torch.dtype, optional): Data type for the matrix.
#
#     Returns:
#         torch.Tensor: A sparse matrix of shape
#           (num_blocks * block_size, num_blocks * len(selected_indices))
#     """
#     selected_indices = list(selected_indices)
#     num_selected = len(selected_indices)
#     total_rows = num_blocks * block_size
#     total_cols = num_blocks * num_selected
#
#     row_indices = []
#     col_indices = []
#     values = []
#
#     for block in range(num_blocks):
#         for j, idx in enumerate(selected_indices):
#             row = block * block_size + idx
#             col = block * num_selected + j
#             row_indices.append(row)
#             col_indices.append(col)
#             values.append(1.0)
#
#     indices = torch.tensor([row_indices, col_indices], dtype=torch.long, device=device)
#     values = torch.tensor(values, dtype=dtype, device=device)
#     S_inv = torch.sparse_coo_tensor(indices, values, (total_rows, total_cols)).to_dense()
#     return S_inv
#
#
# # def create_selection_matrix(num_blocks, block_size, selected_indices, device=None, dtype=torch.float32):
# #     """
# #     Create a selection matrix that extracts specified indices from each block of a flattened tensor.
# #
# #     Given a flattened tensor composed of num_blocks blocks (each of length block_size),
# #     this function builds a selection matrix E such that:
# #
# #         E @ x == x.view(num_blocks, block_size)[:, selected_indices]
# #
# #     The resulting matrix E has shape (num_blocks * len(selected_indices), num_blocks * block_size).
# #
# #     Args:
# #         num_blocks (int): The number of blocks in the flattened tensor.
# #         block_size (int): The size of each block.
# #         selected_indices (list or 1D tensor): Indices to select from each block.
# #             Each value must satisfy 0 <= index < block_size.
# #         device (torch.device, optional): The device on which to create the tensor.
# #         dtype (torch.dtype, optional): The data type of the resulting tensor.
# #
# #     Returns:
# #         torch.Tensor: The selection matrix of shape (num_blocks * len(selected_indices), num_blocks * block_size).
# #     """
# #     selected_indices = list(selected_indices)  # ensure it's a list
# #     num_selected = len(selected_indices)
# #     total_rows = num_blocks * num_selected
# #     total_cols = num_blocks * block_size
# #     E = torch.zeros(total_rows, total_cols, device=device, dtype=dtype)
# #
# #     for block in range(num_blocks):
# #         for j, sel in enumerate(selected_indices):
# #             row = block * num_selected + j
# #             col = block * block_size + sel
# #             E[row, col] = 1.0
# #     return E
# def create_selection_matrix(n_blocks, block_size, selected_dims, weights=None):
#     """
#     Constructs a sparse selection matrix A that selects (and optionally weights) entries
#     from a block-structured vector.
#
#     The resulting matrix A has shape (n_blocks * len(selected_dims), n_blocks * block_size)
#     so that for each block i and for each selected dimension index r:
#
#         A[i * len(selected_dims) + r, i * block_size + selected_dims[r]] = weight
#           (or 1 if weights is None)
#
#     This can be used, for example, to represent an operation like:
#
#         visc.flatten() = A * E_props.grad_faces_n.flatten()
#
#     where each block corresponds to an edge and selected_dims are the columns (dimensions)
#     chosen from each block.
#
#     Args:
#         n_blocks (int): Number of blocks (e.g., m, the number of edges).
#         block_size (int): The size of each block (e.g., p, the total number of components).
#         selected_dims (list or 1D tensor): The indices to select from each block.
#         weights (Tensor, optional): A tensor of shape (n_blocks, len(selected_dims)) containing
#                                     weights for each selected entry. If provided, these values are
#                                     used as the nonzero entries in A. Defaults to None (all ones).
#
#     Returns:
#         torch.sparse.FloatTensor: The sparse selection matrix A.
#     """
#     k = len(selected_dims)
#     rows = []
#     cols = []
#     vals = []
#
#     # Loop over each block and each selected index.
#     for i in range(n_blocks):
#         for r, d in enumerate(selected_dims):
#             rows.append(i * k + r)
#             cols.append(i * block_size + int(d))
#             if weights is not None:
#                 vals.append(weights[i, r].item())
#             else:
#                 vals.append(1.0)
#
#     indices = torch.tensor([rows, cols], dtype=torch.long)
#     values = torch.tensor(vals, dtype=torch.float32)
#
#     # Construct the sparse matrix of shape (n_blocks * k, n_blocks * block_size)
#     A = torch.sparse_coo_tensor(indices, values, size=(n_blocks * k, n_blocks * block_size))
#     return A
#
#
# def create_block_diagonal(normals):
#     """
#     Create a block diagonal matrix D that has, for each block i,
#     a block of shape (2, 1) equal to normals[i, :].
#
#     Args:
#         normals (torch.Tensor): Tensor of shape (n_edges, 2).
#
#     Returns:
#         torch.Tensor: A block diagonal matrix of shape (n_edges*2, n_edges).
#     """
#     n_edges = normals.shape[0]
#     D = torch.zeros(n_edges * 2, n_edges, dtype=normals.dtype, device=normals.device)
#     for i in range(n_edges):
#         # Place normals[i, :] as a column in the i-th block.
#         D[2 * i:2 * i + 2, i] = normals[i, :]
#     return D
#
#
# def combine_edge_operators(A_main, A_bc, b_bc, bc_edge_mask, n_edges, n_cells, n_comp, device):
#     """
#     Combines a main-edge operator and a boundary-edge operator into a single global operator.
#
#     Parameters:
#       A_main      : sparse COO tensor of shape (n_edges_m*n_comp, n_cells*n_comp)
#                     -- the main-edge operator (with local row ordering).
#       A_bc        : sparse COO tensor of shape (n_edges_bc*n_comp, n_cells*n_comp)
#                     -- the boundary-edge operator (with local row ordering).
#       b_bc        : tensor of shape (n_edges_bc*n_comp,)
#                     -- the offset vector for boundary edges.
#       bc_edge_mask: Boolean tensor of shape (n_edges,)
#                     -- True if the global edge is a boundary edge.
#       n_edges     : int, total number of global edges.
#       n_cells     : int, number of cells.
#       n_comp      : int, number of components.
#       device      : torch.device
#
#     Returns:
#       A_all       : sparse COO tensor of shape (n_edges*n_comp, n_cells*n_comp)
#                     -- the combined operator.
#       b_all       : tensor of shape (n_edges*n_comp,)
#                     -- the combined offset vector.
#     """
#
#     """# Get the COO indices and values for the two operators.
#     # (They must be in COO format.)
#     A_main_indices = A_main._indices()  # shape (2, L_main)
#     A_main_values = A_main._values()  # shape (L_main,)
#     A_bc_indices = A_bc._indices()  # shape (2, L_bc)
#     A_bc_values = A_bc._values()  # shape (L_bc,)
#
#     # We will build lists of row indices, column indices, and values for the global operator.
#     global_rows = []
#     global_cols = []
#     global_vals = []
#     b_all_list = []  # offset for each global row
#
#     # Counters for the local row index in A_main and A_bc.
#     # They indicate which main (or bc) edge (block) we are currently processing.
#     main_counter = 0
#     bc_counter = 0
#
#     # Loop over all global edges.
#     for i in range(n_edges):
#         # For each edge, process all components.
#         for c in range(n_comp):
#             # Compute the flattened (global) row index for edge i and component c.
#             global_row = i * n_comp + c
#
#             if not bc_edge_mask[i]:
#                 # --- Main edge ---
#                 # The corresponding local row in A_main is:
#                 local_row = main_counter * n_comp + c
#                 # Find the entries in A_main corresponding to this local row.
#                 mask = (A_main_indices[0, :] == local_row)
#                 # (These entries come with column indices and values.)
#                 cols = A_main_indices[1, :][mask]
#                 vals = A_main_values[mask]
#                 # Append these entries, but with the global row instead of the local row.
#                 for col, val in zip(cols.tolist(), vals.tolist()):
#                     global_rows.append(global_row)
#                     global_cols.append(col)
#                     global_vals.append(val)
#                 # For a main edge, no offset is added.
#                 b_all_list.append(0.0)
#             else:
#                 # --- Boundary edge ---
#                 local_row = bc_counter * n_comp + c
#                 mask = (A_bc_indices[0, :] == local_row)
#                 cols = A_bc_indices[1, :][mask]
#                 vals = A_bc_values[mask]
#                 for col, val in zip(cols.tolist(), vals.tolist()):
#                     global_rows.append(global_row)
#                     global_cols.append(col)
#                     global_vals.append(val)
#                 # The offset for boundary edges comes from b_bc.
#                 # (Assume b_bc is a 1D tensor of length n_edges_bc*n_comp.)
#                 b_all_list.append(b_bc[local_row].item())
#         # Update the local counters.
#         if not bc_edge_mask[i]:
#             main_counter += 1
#         else:
#             bc_counter += 1
#
#
#     # Convert the lists into tensors.
#     indices = torch.tensor([global_rows, global_cols], dtype=torch.long, device=device)
#     values = torch.tensor(global_vals, dtype=A_main_values.dtype, device=device)
#
#     # The global operator acts on flattened cell fields of length n_cells*n_comp and produces
#     # an output of length n_edges*n_comp.
#     size_all = (n_edges * n_comp, n_cells * n_comp)
#     A_all = torch.sparse_coo_tensor(indices, values, size=size_all).coalesce()
#     b_all = torch.tensor(b_all_list, dtype=A_main_values.dtype, device=device)"""
#
#     # Assume the following inputs are given:
#     # A_main: sparse COO tensor of shape (n_edges_m*n_comp, n_cells*n_comp)
#     # A_bc: sparse COO tensor of shape (n_edges_bc*n_comp, n_cells*n_comp)
#     # b_bc: tensor of shape (n_edges_bc*n_comp,)
#     # bc_edge_mask: Boolean tensor of shape (n_edges,), where True indicates a boundary edge.
#     # n_edges, n_cells, n_comp, device are given.
#
#     # First, compute the mapping for global main and boundary edges.
#     # The ordering of the local operators corresponds to the order of the global edges.
#     main_edge_global_indices = torch.where(~bc_edge_mask)[0]  # shape: (n_main_edges,)
#     bc_edge_global_indices = torch.where(bc_edge_mask)[0]  # shape: (n_bc_edges,)
#
#     A_main_indices = A_main._indices()  # shape: (2, L_main)
#     A_main_values = A_main._values()  # shape: (L_main,)
#     A_bc_indices = A_bc._indices()  # shape: (2, L_bc)
#     A_bc_values = A_bc._values()  # shape: (L_bc,)
#
#     # --- Process A_main ---
#     # For each local row in A_main, determine its main edge index and component:
#     local_rows_main = A_main_indices[0, :]  # local row indices in A_main (range: 0 to n_edges_m*n_comp - 1)
#     j_main = local_rows_main // n_comp  # index into main_edge_global_indices
#     c_main = local_rows_main % n_comp  # component index
#
#     # Map to global row index: for main edges the global row is (global_edge_index * n_comp + component)
#     global_rows_main = main_edge_global_indices[j_main] * n_comp + c_main
#     global_cols_main = A_main_indices[1, :]
#
#
#     # --- Process A_bc ---
#     local_rows_bc = A_bc_indices[0, :]  # local row indices in A_bc (range: 0 to n_edges_bc*n_comp - 1)
#     j_bc = local_rows_bc // n_comp  # index into bc_edge_global_indices
#     c_bc = local_rows_bc % n_comp  # component index
#
#     global_rows_bc = bc_edge_global_indices[j_bc] * n_comp + c_bc
#     global_cols_bc = A_bc_indices[1, :]
#
#     # --- Combine the main and boundary contributions ---
#     global_rows = torch.cat([global_rows_main, global_rows_bc], dim=0)
#     global_cols = torch.cat([global_cols_main, global_cols_bc], dim=0)
#     global_vals = torch.cat([A_main_values, A_bc_values], dim=0)
#     global_indices = torch.stack([global_rows, global_cols], dim=0)
#
#     # Build the global sparse operator of shape (n_edges*n_comp, n_cells*n_comp).
#     A_all = torch.sparse_coo_tensor(global_indices, global_vals,
#                                     size=(n_edges * n_comp, n_cells * n_comp),
#                                     device=device, dtype=A_main.dtype).coalesce()
#
#     # --- Build the global offset vector b_all ---
#     b_all = torch.zeros(n_edges * n_comp, device=device, dtype=b_bc.dtype)
#     # For the boundary rows, compute the global row indices similarly.
#     # Create a vector for the local rows in the boundary operator.
#     r_bc = torch.arange(b_bc.numel(), device=device)
#     j_bc_for_b = r_bc // n_comp  # which boundary edge block each entry belongs to
#     c_bc_for_b = r_bc % n_comp  # component within the block
#
#     global_b_rows = bc_edge_global_indices[j_bc_for_b] * n_comp + c_bc_for_b
#     b_all[global_b_rows] = b_bc
#
#
#     return A_all.to_sparse_csr(), b_all
#
#
# def lift_sparse_matrix(A_old, n_comp):
#     """
#     Lift a sparse matrix so that it acts on a flattened multi-component vector.
#         U_out = torch.sparse.mm(A_old, U)   # U_out has shape (M, n_comp)
#         U_out = torch.sparse.mm(A_new, U.flatten()).reshape(M, n_comp)
#     A_old : torch.sparse.Tensor
#         A sparse matrix in COO format of shape (M, N).
#     n_comp : int
#         The number of components (i.e. the second dimension of U).
#     A_new : torch.sparse.Tensor
#         The "lifted" sparse matrix of shape (M*n_comp, N*n_comp) that operates on a flattened U.
#     """
#     # Get the original indices and values.
#     # indices_old is a tensor of shape (2, nnz), where nnz is the number of nonzero entries.
#     A_old = A_old.coalesce()
#
#     indices_old = A_old.indices()  # shape: (2, nnz)
#     values_old = A_old.values()  # shape: (nnz,)
#     M, N = A_old.size()
#     nnz = values_old.size(0)
#     device = A_old.device
#
#     # Create a vector for the component indices: 0, 1, ..., n_comp - 1.
#     comp = torch.arange(n_comp, device=device)  # shape: (n_comp,)
#
#     # For each nonzero entry in A_old, we replicate the index for each component.
#     # The new row index for an entry originally at row i becomes:
#     #    i_new = i * n_comp + c   for c in 0,..., n_comp-1.
#     new_rows = indices_old[0].unsqueeze(1) * n_comp + comp.unsqueeze(0)  # shape: (nnz, n_comp)
#     new_cols = indices_old[1].unsqueeze(1) * n_comp + comp.unsqueeze(0)  # shape: (nnz, n_comp)
#     new_vals = values_old.unsqueeze(1).expand(nnz, n_comp)  # shape: (nnz, n_comp)
#
#     # Flatten these arrays to create the COO indices for A_new.
#     new_rows = new_rows.reshape(-1)  # shape: (nnz * n_comp,)
#     new_cols = new_cols.reshape(-1)
#     new_vals = new_vals.reshape(-1)
#
#     new_indices = torch.stack([new_rows, new_cols], dim=0)  # shape: (2, nnz * n_comp)
#     new_size = (M * n_comp, N * n_comp)
#
#     # # For now, we don't need rho entries.
#     # rho_mask = (new_indices[0] % 3 == 2)
#     # new_indices = new_indices[:, ~rho_mask]
#     # new_vals = new_vals[~rho_mask]
#
#     A_new = torch.sparse_coo_tensor(new_indices, new_vals, size=new_size, device=device).coalesce()
#
#     return A_new


class FarfieldBC:
    def __init__(self, cfg: ConfigFVM, farfield_mask):
        # TODO: Don't assume boundary is in +X direction, use phi for general boundary.

        self.cfg = cfg
        self.exit_cfg = cfg.exit_cfg

        self.farfield_mask = farfield_mask
        self.v_far = self.exit_cfg.v_far
        self.rho_far = self.exit_cfg.rho_far

        self.factor = 1
        self.c = 300

        if self.exit_cfg.mode == "decay":
            self.tau = 1 / self.exit_cfg.decay_tau
            self.decay_beta = self.exit_cfg.decay_beta
            self.set_bc_U_face = self.__decay
        elif self.exit_cfg.mode == "farfield":
            self.set_bc_U_face = self.__farfield
        elif self.exit_cfg.mode == "interior":
            self.beta = 1 - 1 / self.exit_cfg.beta_tau
            self.set_bc_U_face = self.__interior


    def __decay(self, U_face, Us_bc_cells, dt):
        """" Combine two methods to decay boundary:
                U_charachteristic = f * rho_interior * torch.exp(vx_interior - self.v_far)

                f estimates the natural farfield value rho* exp(-v_far)

            Interpolate using moving state f(S):
                S = (1-tau) * S + tau  * dUdt
                U_face = f(S) * U_charachteristic + (1 - f(S)) * U_decay

        """
        vx_interior = Us_bc_cells[:, 0]
        tau = dt * self.tau

        rho_bc = self.factor * self.rho_far * torch.exp((vx_interior - self.v_far)/self.c)

        d_factor = (self.rho_far - rho_bc) + self.decay_beta * (1 - self.factor)
        self.factor = self.factor + tau * d_factor

        # rho_bc = rho_bc.clamp(min=0.3, max=1.4)
        U_face[self.farfield_mask] =  rho_bc


    def __farfield(self, U_face, Us_bc_cells, dt):
        # V = Us_bc_cells[:, :2]                                          # shape = [n_ff_edge, 2]
        # V_n = (V * self.farfield_normals).sum(dim=1, keepdim=True)      # shape = [n_ff_edge, 1]

        # rho_int = Us_bc_cells[:, 2]                                   # shape = [n_ff_edge]
        # T_int = Us_bc_cells[:, 3]                                     # shape = [n_ff_edge]
        # P_int = self.R * rho_int * T_int
        #
        # rho_bc = rho_int * (self.P_far / P_int) ** (1 / self.cfg.gamma)
        # T_bc = self.P_far / (rho_bc * self.cfg.R)

        # U_face[self.farfield_mask, 2] = rho_bc
        # U_face[self.farfield_mask, 3] = T_bc

        vx_interior = Us_bc_cells[:, 0]
        rho_bc = self.rho_far * torch.exp(vx_interior - self.v_far)
        U_face[self.farfield_mask] = rho_bc

    def __interior(self, U_face, Us_bc_cells, dt):
        vx_interior = Us_bc_cells[:, 0]
        rho_interior = Us_bc_cells[:, 2]
        beta = dt * self.beta

        U_face[self.farfield_mask] = torch.where((vx_interior < 0),
                             rho_interior * torch.exp(vx_interior),
                             (beta * rho_interior + (1-beta) * self.rho_far),
                             )


    def set_bc_U_face(self, U_face, Us_bc_cells):
        raise NotImplementedError


class SlopeLimiter:
    def __init__(self, areas, cfg: ConfigFVM):
        lim_p = cfg.lim_p
        K = cfg.lim_K
        areas = areas.view(-1, 1, 1)

        self.eps_p = (K * areas ** 0.5) ** (lim_p+1)

        if lim_p == 1:
            self._limit = self.p1
        elif lim_p == 2:
            self._limit = self.p2
        elif lim_p == 3:
            self._limit = self.p3
        elif lim_p == 4:
            self._limit = self.p4
        else:
            raise NotImplementedError(f"Limiter of order {lim_p} is not implemented.")

    def p1(self, delta, dU):
        """ BJ limiter"""
        dU = 2 * ((dU > 0).float() - 0.5) * (dU.abs() + 1e-8)
        r = delta / dU  # shape = [n_cells, neigh=3, n_comp]
        phi = torch.clamp(r, min=0., max=1)  # shape = [n_cells, neigh=3, n_comp]
        return phi

    def p2(self, delta, dU):
        """ Venkatakrishnan limiter"""
        phi = (delta ** 2 + self.eps_p + 2 * delta * dU) / (delta ** 2 + 2 * dU ** 2 + delta * dU + self.eps_p)
        return phi

    def p3(self, delta, dU):
        """ 3rd order limiter - https://arc.aiaa.org/doi/epdf/10.2514/6.2022-1374"""
        a = delta.abs()
        b = dU.abs()

        a_eps = a ** 3 + self.eps_p
        S = 4 * b ** 2
        phi = (a_eps + a * S) / (a_eps + b * (delta ** 2 + S))
        phi = torch.where(a < 2 * b, phi, 1)
        return phi

    def p4(self, delta, dU):
        """ 4th order limiter """
        a = delta.abs()
        b = dU.abs()
        a_eps = a ** 4 + self.eps_p
        S = 2 * b * (a ** 2 - 2 * b * (a - 2 * b))
        phi = (a_eps + a * S) / (a_eps + b * (delta ** 3 + S))
        phi = torch.where(a < 2 * b, phi, 1)
        return phi

    def limit(self, delta, dU):
        """ delta: maximum allowed values
            dU: Predicted value from lstsq gradient
        """
        phi = self._limit(delta, dU)    # shape = [n_cells, neigh=3, n_comp]

        # print(f'{phi[4508, :, 0] = }')
        # Cell wide clamping
        phi = torch.min(phi, dim=1, keepdim=True).values  # shape = [n_cells, neigh=1, n_comp]

        return phi


class FVMEdgeInfo:
    device: str
    n_edges: int
    n_cells: int
    n_comp: int
    slope_limiter: SlopeLimiter

    # Shared
    edge_len: torch.Tensor  # shape = (n_edges, 1)
    normals: torch.Tensor  # shape = (n_edges, 2)
    normals_hat: torch.Tensor  # shape = (n_edges, 2)
    #V_insertion_matrix: torch.Tensor  # shape = (n_edges*n_component, n_edges*2)
    # edge_to_tri_comb: torch.Tensor  # shape = (n_edges, 2)

    # Main mesh
    edge_to_tri_main: torch.Tensor  # shape = [n_edges_m, 2], ordered so triangle parallel to edge normal comes last, antiparallel first.
    cell_dist_proj: torch.Tensor  # shape = (n_edges_m)
    tri_edge_signs: torch.Tensor  # shape = (3 * n_cells)
    tri_to_edge: torch.Tensor  # shape = (3 * n_cells)
    cent_to_edge_disp: torch.Tensor  # shape = (n_cells, 3, 2)  # Displacement vector between cell centroids, for every edge

    # Boundary condition
    n_edges_bc: int             # Number of boundary edges
    bc_edge_mask: torch.Tensor  # shape = (n_edges)
    edge_to_tri_bc: torch.Tensor  # shape = (n_edges_bc)
    exit_cell2edge: torch.Tensor  # shape = (n_cells, 2)  # Exit edge for each cell
    bc_edge_side: torch.Tensor  # shape = (n_edges_bc, 2)  # Side of the edge for each boundary edge
    use_farfield: bool
    boundary_setter: any

    # Gradients
    G_mats: list[torch.Tensor]  # shape = [2](n_cells, n_cells)  Gradient matrix for every cell
    edge_dists_bc: torch.Tensor  # shape = (n_bc_edges, n_comp)     Distance between cell centroids, for every edge_bc
    neigh_combine: torch.Tensor # shape = (n_cell, 2). Used for masking neighbors of cell incl boundary edges, in format [Us, U_face_bc]

    # Temporary Variables
    dUf_dUc: tuple[torch.Tensor, torch.Tensor] # shape = [n_comp * n_edges, n_comp * n_cells]. Jacobian of face w.r.t. cell
    grad_faces_n: torch.Tensor  # shape = (n_edges, n_comp)  n . grad(u) on faces
    # Primitive face variables
    grad_V: torch.Tensor        # shape = (n_edges, {dx,dy}, {vx,vy})  Gradient at V_faces
    Vs_faces: torch.Tensor  # shape = (n_edges, 2, 2)  Face values
    rho_faces: torch.Tensor  # shape = (n_edges, 2, 1)  Face values
    T_faces: torch.Tensor # shape = (n_edges, 2, 1)  Face values
    grad_T_n: torch.Tensor  # shape = (n_edges)         Face temperature gradient
    div_V_faces: torch.Tensor  # shape = (n_edges, 2)  Divergence of V_faces
    mom_faces: torch.Tensor     # shape = (n_edges, 2, 2)  Face values
    Q_faces: torch.Tensor       # shape = (n_edges, 2, 1)  Face energy values
    phi: torch.Tensor  # shape = (n_edges, 1)  Face values = V_faces dot normals. After averaging over faces.


    def __init__(self, cfg: ConfigFVM, mesh: FVMMesh, n_comp, bc_tags, device="cpu"):
        self.device = device
        self.cfg = cfg
        self.mesh = mesh
        self.n_edges = mesh.n_edges
        self.n_cells = mesh.n_cells
        self.n_comp = n_comp
        self.slope_limiter = SlopeLimiter(mesh.areas.to(device), cfg)


        self.C_v = cfg.C_v

        self.edge_to_tri_main = mesh.edge_to_tri_main.to(device)
        self.cent_to_edge_disp = mesh.cent_to_edge_disp.to(device).unsqueeze(-1)
        self.tri_edge_signs = (-self.mesh.tri_edge_signs + 1 / 2).to(torch.int32).view(3*self.n_cells).to(device)
        self.tri_to_edge = mesh.tri_to_edge.view(3*self.n_cells).to(device)

        self.edge_to_tri_bc = mesh.edge_to_tri_bc.to(device)
        self.bc_edge_mask = mesh.bc_edge_mask.to(device)

        (cell_disps, edge_dists_bc, G_mats, neigh_combine, edge_to_tri_comb) = mesh.cell_grad_stuff
        self.edge_dists_bc = edge_dists_bc.to(device).unsqueeze(-1).expand(-1, self.n_comp)
        self.G_mats = torch.cat([G_mats[0], G_mats[1]], dim=0).to_sparse_csr().to(device)

        self.neigh_combine = neigh_combine.to(device)
        self.cell_disps = cell_disps.to(device)
        self.normals = mesh.normals.to(device)
        normal_hat = self.normals / torch.norm(self.normals, dim=1, keepdim=True)
        self.normals_hat = normal_hat
        cell_disps = torch.full((self.n_edges, 2), float("nan"), device=device)
        cell_disps[~self.bc_edge_mask] = self.cell_disps
        d_cos_theta = (normal_hat * cell_disps).sum(dim=1)
        self.cell_dist_proj = d_cos_theta
        # Non-orthogonal correction: du/dn = du/dn_face - grad(U) (d/(n_hat dot d) - n_hat)
        X_orthog = cell_disps / d_cos_theta.unsqueeze(-1) - normal_hat
        X_orthog[self.bc_edge_mask] = 0
        self.X_orthog = X_orthog
        self.edge_len = torch.norm(self.normals, dim=1).to(device).unsqueeze(-1)

        self._init_bc(bc_tags)
        c_print(f'_init_bc done', color="magenta")

        self._build_spm_face_grads()
        c_print(f'_build_spm_face_grads done', color="magenta")

        #self.V_insertion_matrix = create_insertion_matrix(self.n_edges, self.n_comp, [0, 1], device=device).to_sparse_csr()
        # c_print(f'V_insertion_matrix done', color="magenta")

        # Create indexing masks between main and boundary edges
        # Step 1: Create a boolean tensor tracking which face of each edge is assigned.
        face_assigned = torch.zeros((self.n_edges, 2), dtype=torch.bool, device=self.device)
        face_assigned[self.tri_to_edge, self.tri_edge_signs] = True
        # Step 2: For each boundary edge, find which side (face) is not assigned.
        # This returns a tensor of shape (n_boundary_edges,) with the index (0 or 1) of the unset face.
        assigned_boundary = face_assigned[self.bc_edge_mask]  # shape: (n_boundary_edges, 2)
        self.bc_edge_side = (~assigned_boundary).float().argmax(dim=1)
        c_print(f'Complete init FVMEdgeInfo', color="magenta")

        self.cell_grads = None


    def clear_temp(self):
        # del self.edge_dists_bc, self.cell_dist_proj, self.edge_to_tri_main, self.dirich_val, self.neumann_val
        # del self.dirich_mask, self.neumann_mask
        # del self.dUf_dUc

        torch.cuda.empty_cache()
        c_print(f'Deleted temp variables', color="magenta")


    def _init_bc(self, bc_tags: dict[int, Edge]):
        #self.n_edges_m = self.n_edges - self.bc_edge_mask.sum().item()
        self.n_edges_bc = self.bc_edge_mask.sum().item()

        dirich_mask, neumann_mask = [], []
        dirich_val, neumann_val = [], []
        farfield_mask = []
        euler_w_mask = []
        self.use_farfield = False
        for bc_idx, e_type in bc_tags.items():
            dirich_mask.append(e_type.dirichlet())
            neumann_mask.append(e_type.neumann())
            dirich_val.append(e_type.U)
            neumann_val.append(e_type.dUdn)

            euler_w_mask.append(e_type.euler_wall)

            farfield_mask.append(e_type.farfield())

        self.dirich_mask, self.neumann_mask = torch.tensor(dirich_mask, device=self.device), torch.tensor(neumann_mask, device=self.device)
        self.farfield_mask = torch.tensor(farfield_mask, device=self.device)  # All farfield must be the same
        self.use_farfield = torch.any(self.farfield_mask).item()

        dirich_val, neumann_val = torch.tensor(dirich_val, dtype=torch.float32, device=self.device), torch.tensor(neumann_val, dtype=torch.float32, device=self.device)
        self.dirich_val = dirich_val[self.dirich_mask]
        self.neumann_val = neumann_val[self.neumann_mask]

        assert self.dirich_mask.shape[0] == self.bc_edge_mask.sum(), f'Wrong mask shape'
        assert self.neumann_mask.shape[0] == self.bc_edge_mask.sum(), f'Wrong mask shape'
        assert self.farfield_mask.shape[0] == self.bc_edge_mask.sum(), f'Wrong mask shape'

        self.boundary_setter = BoundarySetter(self)

        if self.use_farfield:
            exit_cell2edge = self.edge_to_tri_bc[self.farfield_mask[:, 2]]
            # self.farfield_calc = FarfieldBC(self.cfg, self.farfield_mask)
            self.boundary_setter.init_farfield(self.cfg, self.farfield_mask, exit_cell2edge)


    @torch.compile()
    def precompute_shared(self, Us, dt):
        """ Precompute shared values that are used multiple times later.
            Us.shape = [n_cells, n_component] """
        U_face_bc = self._bc_face_vals(Us, dt)      # shape = [n_edges_bc, n_comp]
        Us_cell_face = torch.cat([Us, U_face_bc])        # shape = [n_cells + n_edges_bc, n_comp]
        cell_grads = self._cell_grads(Us_cell_face) # shape = [n_cells, 2, n_comp]
        grad_faces_n = self._face_grads(Us)        # shape = [n_faces, n_comp]

        self.cell_grads = cell_grads

        # Compute limited face values,
        Us_face, phi_lim = self._limit_face_vals(Us, U_face_bc, cell_grads)   # Us_face.shape = [n_cells, 3, n_comp], phi_lim.shape =  [n_cells, 1, n_comp]
        Us_face = Us_face.view(3 * self.n_cells, self.n_comp)
        cell_grads = cell_grads * phi_lim

        self.phi_lim = phi_lim

        # Face gradients of velocity and temperature
        grad_F_dn = grad_faces_n[:, [0, 1, 3]]   # shape = [n_faces, 3]
        grad_F = cell_grads[:, :, [0, 1, 3]]       # shape = [n_cells, {x, y} , {vx, vy, T}]
        grad_F_flat = grad_F.repeat_interleave(3, dim=0).view(3*self.n_cells, 6) # [dvx/dx, dvy/dx, dvx/dy, dvy/dy, dT/dx, dT/dy]
        grad_F_bc = grad_F_flat[self.edge_to_tri_bc]

        # Limited cell divergence
        div_V = cell_grads[:, 0, 0] + cell_grads[:, 1, 1]        # shape = [n_cells]
        div_V_bc = div_V[self.edge_to_tri_bc].unsqueeze(-1)
        div_V = div_V.repeat_interleave(3).unsqueeze(-1)            # shape = [3*n_cells, 1]


        # Prepare projection from cell to edges - (slow step so vectorise over all components)
        cell_values = torch.cat([Us_face, div_V, grad_F_flat], dim=-1)        # shape = [3*n_cells, n_comp+1]
        cell_values_bc = torch.cat([U_face_bc, div_V_bc, grad_F_bc], dim=1)      # shape = [n_edges_bc, n_comp+1]

        # Project to left and right face values
        U_face_all = torch.empty((self.n_edges, 2, self.n_comp + 7), device=self.device)    # [momx, momy, rho, Q, div_V, face_grad X 4]
        U_face_all[self.tri_to_edge, self.tri_edge_signs] = cell_values
        U_face_all[self.bc_edge_mask, self.bc_edge_side] = cell_values_bc
        # U_face_all[self.bc_edge_mask, ~self.bc_edge_side] = cell_values_bc

        # """ TEMP TEST """
        # neum_mask_all = torch.zeros_like(self.bc_edge_mask)
        # neum_mask_all = neum_mask_all.unsqueeze(-1).repeat(1, 4)
        # neum_mask_all[self.bc_edge_mask] = self.neumann_mask
        # where_neum_all = torch.where(neum_mask_all)
        #
        # where_neum = torch.where(self.neumann_mask)
        # neum_vals = U_face_bc[where_neum[0], where_neum[1]]
        # U_face_all[where_neum_all[0], :, where_neum_all[1]] = neum_vals.unsqueeze(-1)


        # Decompose components back
        self.Vs_faces = U_face_all[:, :, [0, 1]]  # shape = [n_edges, edges=2, n_comp=2]
        self.rho_faces = U_face_all[:, :, [2]]  # shape = [n_edges, edges=2, dims=1]
        self.T_faces = U_face_all[:, :, [3]]    # shape = [n_edges, edges=2, dims=1]
        self.div_V_faces = U_face_all[:, :, 4]  # shape = [n_edges, edges=2]

        # Conserved quantities
        self.mom_faces = self.Vs_faces * self.rho_faces  # shape = [n_edges, edges=2, dims=2]
        self.Q_faces = self.rho_faces * (1/2  * self.Vs_faces.norm(dim=-1, keepdim=True) ** 2 + self.C_v * self.T_faces)
        self.phi = (self.Vs_faces * self.normals.unsqueeze(1)).sum(dim=-1) # shape = [n_edges, edges=2, ]

        # Non-orthogonal correction for face gradient. Assume lstsq gradient is mean of left and right cell.
        grad_F_lstsq = U_face_all[:, :, 5:11].view(self.n_edges, 2, 6)   # shape = [n_edges, edges=2, components=6]
        grad_F_lstsq = grad_F_lstsq.mean(dim=1).view(self.n_edges, 2, 3)   # shape = [n_edges, {x, y}, {vx, vy, T}]
        dFdn_correct = grad_F_dn - (grad_F_lstsq * self.X_orthog.unsqueeze(-1)).sum(dim=1)      # shape = [n_edges, 3]
        # Replace normal part of gradient with face gradient
        grad_F_dot_n = (grad_F_lstsq * self.normals_hat.unsqueeze(-1)).sum(dim=1, keepdim=True)       # shape = [n_edges, 1, 3]
        grad_F_para = (dFdn_correct.unsqueeze(dim=1) - grad_F_dot_n) * self.normals_hat.unsqueeze(dim=-1)       # shape = [n_edges, 2, 3]
        grad_F = grad_F_lstsq + grad_F_para             # shape = [n_edges, 2, 3]
        self.grad_V = grad_F[:, :, :2]
        self.grad_T_n = dFdn_correct[:, 2]      # shape = [n_edges]
        # self.grad_T_n = grad_faces_n[:, 3]


    def _limit_face_vals(self, Us, U_face_bc, cell_grads):
        """ Limited B-J scheme for cell to face interpolation.

        """
        U_cent = Us.unsqueeze(1)        # shape = [n_cells, 1, n_comp]
        Us_cell_face = torch.cat([Us, U_face_bc])        # shape = [n_cells + n_edges_bc, n_comp]
        Us_neigh = Us_cell_face[self.neigh_combine]  # shape = [n_cells, neigh=3, n_comp]

        # Uncorrected update
        grads = cell_grads.unsqueeze(1)     # shape = [n_cells, 1, dims=2, n_comp]
        dU = (grads * self.cent_to_edge_disp).sum(dim=2)  # shape = [n_cells, neigh=3, n_comp]

        # Select limiting neighbor values and compute gradient limiter
        U_cent_neigh = torch.cat([U_cent, Us_neigh], dim=1)             # shape = [n_cells, 4, n_comp]
        U_upper = torch.max(U_cent_neigh,dim=1, keepdim=True)[0] - U_cent      # shape = [n_cells, 1, n_comp]
        U_lower = torch.min(U_cent_neigh,dim=1, keepdim=True)[0] - U_cent

        numerator = torch.where(dU > 0, U_upper, U_lower) # shape = [n_cells, neigh=3, n_comp]
        phi_lim = self.slope_limiter.limit(numerator, dU)           # shape = [n_cells, neigh=3, n_comp]
        Us_face = U_cent + phi_lim * dU      # shape = [n_cells, neigh=3, n_comp]

        return Us_face, phi_lim


    def _bc_face_vals(self, Us, dt):
        """ Boundary face values.
            Us.shape = [n_cells, n_comp]
            return.shape: [n_edges_bc, n_comp]

         """
        # U_face = torch.empty((self.n_edges_bc, self.n_comp), device=self.device)
        # # Boundary edges
        # u_centroid_bc = Us[self.edge_to_tri_bc] # shape = [n_bc_edges, n_comp]
        # # Dirichlet
        # U_face[self.dirich_mask] = self.dirich_val
        # # Neumann
        # U_cent_bc_neum = u_centroid_bc[self.neumann_mask]        # shape = [n_neum_edges]
        # U_face_neum = U_cent_bc_neum + self.neumann_val * self.edge_dists_bc[self.neumann_mask]
        # U_face[self.neumann_mask] = U_face_neum

        U_face = self.boundary_setter.set_face_values(Us, self.cell_grads, dt)

        # if self.use_farfield :
        #     self.farfield_calc.set_bc_U_face(U_face, Us[self.exit_cell2edge], dt)

        return  U_face


    def _cell_grads(self, Us_cell_face):
        """ Vectorised gradient computation
            Gradient = G @ (u_neigh - u_cell)
            Us_cell_face.shape = (n_cells+n_edges_bc, N_component)
            Returns: Gradient matrix of shape (n_cells, 2, N_component)
        """
        combined_grad = torch.sparse.mm(self.G_mats, Us_cell_face)  # combined_grad.shape == [2 * n_cells, N_component]
        cell_grads = combined_grad.view(2, self.n_cells, -1).permute(1, 0, 2)    # shape = [n_cells, 2, N_component]
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
        """ NEW, non-orthogonal correction """
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
        # dUdn_face = torch.empty((self.n_edges, self.n_comp), device=self.device)
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

        # """ SPARSE"""
        Us_flat = Us.flatten()
        dUdn_face_flat = torch.mv(self.A_face_grad, Us_flat) + self.b_face_grad
        dUdn_face = dUdn_face_flat.view(self.n_edges, self.n_comp)

        return dUdn_face


    def _build_spm_face_grads(self):
        n_edges = self.edge_to_tri_main.shape[0]  # number of faces (edges)
        n_cells = self.n_cells
        n_bc = self.n_edges_bc  # number of boundary edges
        n_comp = self.n_comp  # number of components

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
        cell_dist_rep = self.cell_dist_proj[~self.bc_edge_mask].repeat_interleave(2)  # shape (2*n_edges,)
        # Create a vector with the appropriate signs: first -1 then +1 for each face.
        face_signs = torch.tensor([-1, 1], device=self.device, dtype=torch.float32).repeat(n_edges)
        # Now compute the nonzero values.
        vals = face_signs / cell_dist_rep

        # Build the sparse matrix A_face.
        A_face = torch.sparse_coo_tensor(
            torch.stack([rows, cols]),
            vals,
            size=(n_edges, n_cells))

        A_face_grad_main = lift_sparse_matrix(A_face, self.n_comp)

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

        # print("Starting combine_edge_operators")
        self.A_face_grad, self.b_face_grad = combine_edge_operators(A_face_grad_main, A_grad_bc, b_grad, self.bc_edge_mask, self.n_edges, self.n_cells, self.n_comp, self.device)


    def _build_dUf_dUc(self):
        """ Build sparse matrix for dU_f/dU_c - gradient of face val w.r.t. cell values.
            U_cell = [interleave(mom_x | mom_y | rho)], shape = [3*n_cell]
            U_face = A(U_cell)              shape = [3*n_edges, 2], second dim is Left / Right side of face.
            d(U_f_i, L/R)/d(U_c_j) = :
                                0 if cell_j doesnt have face_i  - Or computing wrong mom_x, mom_y, rho component.
                                1 if face_i is on cell_j and cell_j is on L/R side.
            Returns.shape = 2 * [3*n_edges, 3*n_cells]
        """

        n_edges, n_cells = self.n_edges, self.n_cells
        rows_left = []
        cols_left = []
        rows_right = []
        cols_right = []
        # Loop over each edge and each component
        for e, adj_cell in self.mesh.edge_to_tri.items():
            if adj_cell.numel() == 2:  # Only interior edges have gradient
                for comp in range(self.n_comp):
                    i = 3 * e + comp
                    # Left side: contribution from the left cell.
                    rows_left.append(i)
                    cols_left.append(3 * int(adj_cell[0].item()) + comp)
                    # Right side: contribution from the right cell.
                    rows_right.append(i)
                    cols_right.append(3 * int(adj_cell[1].item()) + comp)

        # Convert indices to tensors.
        rows_left = torch.tensor(rows_left, dtype=torch.long)
        cols_left = torch.tensor(cols_left, dtype=torch.long)
        rows_right = torch.tensor(rows_right, dtype=torch.long)
        cols_right = torch.tensor(cols_right, dtype=torch.long)

        # Create the constant skeleton sparse matrices (values are 1).
        indices_left = torch.stack([rows_left, cols_left], dim=0)
        indices_right = torch.stack([rows_right, cols_right], dim=0)
        ones_left = torch.ones(rows_left.size(0))
        ones_right = torch.ones(rows_right.size(0))

        dUfL_dUc = torch.sparse_coo_tensor(indices_left, ones_left, size=(3 * n_edges, 3 * n_cells), dtype=torch.int16)
        dUfR_dUc = torch.sparse_coo_tensor(indices_right, ones_right, size=(3 * n_edges, 3 * n_cells), dtype=torch.int16)

        #dUfL_dUc, dUfR_dUc = dUfL_dUc.to_sparse_csr(), dUfR_dUc.to_sparse_csr()
        dUfL_dUc, dUfR_dUc = dUfL_dUc.cuda(non_blocking=True), dUfR_dUc.cuda(non_blocking=True)
        return dUfL_dUc, dUfR_dUc


class BoundarySetter:
    """ Non-orthogonal correction for Neumann BCs."""
    n_comp: int
    n_edges_bc: int

    tri_to_bc_edge: torch.Tensor    # shape = [n_edges, 3]
    grad_comps: torch.Tensor          # shape = [n_neum_edges, 2, 1]
    where_neum: tuple[torch.Tensor]      # shape = [2][n_neum_edges, 2]

    # Matrices for general face values
    A_bc: torch.Tensor
    b_bc: torch.Tensor

    # Farfield boundary condition
    use_farfield: bool
    farfield_calc: FarfieldBC
    exit_cell2edge: torch.tensor    # shape = (n_cells, 2)  # Exit edge for each cell

    def __init__(self, E_props: FVMEdgeInfo):
        self.use_farfield = False

        self.n_comp = E_props.n_comp
        self.n_edges_bc = E_props.n_edges_bc
        tri_to_edge = E_props.tri_to_edge.view(-1, 3)

        # Non orthogonal correction is only done for cells with 1 edge.
        tri_to_bc_edge = E_props.bc_edge_mask[tri_to_edge]

        # Flatten out all Neumann BCs and index according to order where_neum_all[0]
        neum_mask_all = torch.zeros_like(E_props.bc_edge_mask)
        neum_mask_all = neum_mask_all.unsqueeze(-1).repeat(1, 4)
        neum_mask_all[E_props.bc_edge_mask] = E_props.neumann_mask
        where_neum_all = torch.where(neum_mask_all)
        where_neum = {'edge': where_neum_all[0], 'comp': where_neum_all[1]}  # shape = [n_neum_edges, 2]
        # Mapping from boundary id to boundary edge id
        self.where_neum = torch.where(E_props.neumann_mask)

        # Mapping from boundary edge to cell
        bc_edge_to_tri = torch.zeros_like(E_props.bc_edge_mask).long()
        bc_edge_to_tri[E_props.bc_edge_mask] = E_props.edge_to_tri_bc

        # Cells corresponding to Neumann BC
        self.neum_cells = bc_edge_to_tri[where_neum_all[0]]  # shape = [n_neum_edges], which cells have neuman BCs
        where_neum['cells'] = self.neum_cells
        # Edge within cell corresponding to Neumann BC
        tri_edge_num = (where_neum['edge'].unsqueeze(-1).repeat(1, 3) == tri_to_edge[where_neum['cells']])
        tri_edge_id = torch.where(tri_edge_num)[1]
        where_neum['tri_edge_id'] = tri_edge_id

        # Which component of gradient is needed for Neumann BC
        self.grad_comps = where_neum['comp'].unsqueeze(1).repeat(1, 2).unsqueeze(2)     # shape = [n_neum_edges, 2, 1]
        # Normal vector of edges
        n_hats = E_props.normals_hat[where_neum['edge']]  # shape = [n_neum_edges, 2]
        # Displacement from centroid to edge
        cent_to_edge = E_props.cent_to_edge_disp[where_neum['cells']].squeeze()  # shape = [n_neum_edge, 3, 2]
        r = cent_to_edge[torch.arange(cent_to_edge.shape[0]), where_neum['tri_edge_id']]  # shape = [n_neum_edge, 2]
        # Normal component of r
        d = n_hats * (r * n_hats).sum(dim=1, keepdim=True)  # shape = [n_neum_edge, 2]
        # Parallel component of r
        self.l = r - d

        A_bc, b_bc = self._build_spm_face_vals(E_props)
        self.A_bc, self.b_bc = A_bc, b_bc

    def set_face_values(self, Us, cell_grads=None, dt=None):
        Us_flat = Us.flatten()
        # Final U_face in flattened form.a
        U_face_flat = torch.mv(self.A_bc, Us_flat) + self.b_bc      # shape = [n_edges_bc * n_comp]
        # Reshape back to (n_edges_bc, n_comp)
        U_face = U_face_flat.view(self.n_edges_bc, self.n_comp)

        if cell_grads is not None:
            self._non_orthogonal_correction(U_face, cell_grads)

        if self.use_farfield:
            self.farfield_calc.set_bc_U_face(U_face, Us[self.exit_cell2edge], dt)

        return U_face

    def _non_orthogonal_correction(self, U_face, cell_grads):
        """
        Use previous gradient for non-orthogonal correction.

        U_face.shape = [n_bc_faces, n_comp]
        cell_grads.shape = [n_cells, 2, n_comp]

        r = centroid to midpoint.
        d = normal component of r
        U_f = U_0 + d * dUdn + (r-d) grad(U)
        """
        grads = torch.gather(cell_grads[self.neum_cells], 2, self.grad_comps).squeeze()  # shape = [n_neum_edge, 2]

        dU = (grads * self.l).sum(dim=1)  # shape = [n_neum_edge]
        U_face[self.where_neum[0], self.where_neum[1]] += dU

    def _build_spm_face_vals(self, E_props):
        """ Compute bc edge values using sparse matrix multiplication. """

        device = E_props.device
        n_bc = E_props.n_edges_bc  # number of boundary edges
        n_comp = E_props.n_comp
        n_cells = E_props.n_cells

        # Total number of flattened BC rows.
        N = n_bc * n_comp

        # Create flattened indices for the boundary rows and the corresponding component.
        # Each boundary edge gives rise to n_comp rows.
        bc_rows = torch.arange(n_bc, device=device).unsqueeze(1).expand(n_bc, n_comp).reshape(-1)
        comp_idx = torch.arange(n_comp, device=device).unsqueeze(0).expand(n_bc, n_comp).reshape(-1)

        # Reshape the condition masks to a flat vector of length N.
        dirich_mask = E_props.dirich_mask.reshape(-1)  # For Dirichlet conditions.
        neum_mask = E_props.neumann_mask.reshape(-1)  # For Neumann conditions.

        # --- Build sparse matrix A ---
        # For Neumann entries, we want to extract the cell value from Us.
        # For each Neumann row, the corresponding column in Us (flattened) is given by:
        #   col = self.edge_to_tri_bc[ edge_index ] * n_comp + component
        neum_indices = torch.nonzero(neum_mask, as_tuple=False).squeeze(1)  # indices where Neumann is True.
        A_rows = neum_indices
        # bc_rows[neum_indices] gives the corresponding boundary edge for each flattened row.
        A_cols = E_props.edge_to_tri_bc[bc_rows[neum_indices]] * n_comp + comp_idx[neum_indices]
        A_vals = torch.ones_like(A_rows, dtype=torch.float32, device=device)

        size_A = (N, n_cells * n_comp)
        A_bc = torch.sparse_coo_tensor(torch.stack([A_rows, A_cols], dim=0), A_vals, size=size_A).coalesce().to_sparse_csr()

        # Build the offset vector b.
        b_bc = torch.empty(N, device=device, dtype=torch.float32)
        # For Dirichlet entries, the prescribed value should override any extracted value.
        b_bc[dirich_mask] = E_props.dirich_val
        # For Neumann entries, add the offset computed from the edge distance.
        # Here, we select the proper component value from self.neumann_val using comp_idx.
        b_bc[neum_mask] = E_props.neumann_val[comp_idx[neum_mask]] * E_props.edge_dists_bc.flatten()[neum_mask]

        return A_bc, b_bc

    def init_farfield(self, cfg, farfield_mask, exit_cell2edge):
        self.use_farfield = True
        self.farfield_calc = FarfieldBC(cfg, farfield_mask)
        self.exit_cell2edge = exit_cell2edge