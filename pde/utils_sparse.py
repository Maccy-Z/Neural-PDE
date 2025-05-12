import torch
import cupy as cp
import numpy as np
from matplotlib import pyplot as plt

def gen_rand_sp_matrix(rows, cols, density, device="cpu"):
    num_nonzeros = int(rows * cols * density)
    row_indices = torch.randint(0, rows, (num_nonzeros,))
    col_indices = torch.randint(0, cols, (num_nonzeros,))
    values = torch.randn(num_nonzeros)  # Random values for the non-zero entries

    edge_index = torch.stack([row_indices, col_indices], dim=0)
    return torch.sparse_coo_tensor(edge_index, values, (rows, cols)).to(device).to_sparse_csr()

def plot_sparsity(A):
    A = A.to_dense()[:250, :250]
    sparse_coo = A.to_sparse_coo().coalesce()
    indices = sparse_coo.indices()
    rows = indices[0].cpu().numpy()
    cols = indices[1].cpu().numpy()
    size = sparse_coo.size()
    # Create dense binary matrix

    dense_binary = np.zeros(size, dtype=np.int32)
    dense_binary[rows, cols] = 1

    # Plot using imshow
    plt.figure(figsize=(15, 15))
    plt.imshow(dense_binary, cmap='Greys', interpolation='none', aspect='auto')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.title(f'Sparsity Pattern, nnz={sparse_coo._nnz()}')
    plt.gca().invert_yaxis()
    plt.show()

def permutation_to_csr(perm, dtype=torch.float32, device="cpu"):
    """
    Convert a permutation tensor to a sparse permutation matrix in CSR format.

    Args:
        perm (torch.Tensor): 1D integer tensor representing the permutation.
                             It should contain each integer from 0 to n-1 exactly once.
        dtype (torch.dtype, optional): Data type of the sparse matrix values. Defaults to torch.float32.

    Returns:
        torch.Tensor: A sparse CSR tensor representing the permutation matrix.
    """
    n = perm.size(0)

    # Validate that perm is a valid permutation
    if not torch.all((perm >= 0) & (perm < n)):
        raise ValueError("Permutation tensor contains invalid indices.")
    if torch.unique(perm).numel() != n:
        raise ValueError("Permutation tensor must contain each index exactly once.")

    # Data is all ones
    data = torch.ones(n, dtype=dtype, device=device)

    # Indices are the permutation itself
    col_indices = perm.clone().to(torch.int32)

    # Indptr for CSR: [0, 1, 2, ..., n]
    row_ptr = torch.arange(n + 1, dtype=torch.int32, device=device)

    # Create sparse CSR tensor
    sparse_matrix = torch.sparse_csr_tensor(
        crow_indices=row_ptr,
        col_indices=col_indices,
        values=data,
        size=(n, n),
        device=device
    )
    return sparse_matrix

class CsrBuilder:
    """ Incrementally build a sparse CSR tensor from dense blocks. """
    def __init__(self, total_rows, total_cols, device=None):
        """
        Initializes the builder for a CSR sparse tensor.
        Parameters:
        - total_rows: int, total number of rows in the matrix.
        - total_cols: int, total number of columns in the matrix.
        - device: torch device (optional).
        """
        self.dtype = torch.int64
        self.total_rows = total_rows
        self.total_cols = total_cols
        self.device = device

        self.zero_ten = torch.tensor([0], dtype=self.dtype, device=self.device)

        # Internal storage for CSR components
        self.nnz_per_row = torch.tensor([0] * self.total_rows, device=self.device, dtype=self.dtype)  # Number of non-zero elements per row
        self.col_indices = []                # Column indices of non-zero elements
        self.values = []                     # Non-zero values

    def add_block(self, block_dense_values, block_row_offset, block_col_offset):
        """
        Adds a dense block to the CSR components using efficient tensor operations.

        Parameters:
        - block_dense_values: 2D tensor (n x m), dense block of values.
        - block_row_offset: int, the starting row index of the block in the overall matrix.
        - block_col_offset: int, the starting column index of the block in the overall matrix.
        """
        n, m = block_dense_values.shape
        crow_idxs, col_idxs, values = self.to_csr(block_dense_values)

        # Count non-zero elements per row in the block
        counts = crow_idxs[1:] - crow_idxs[:-1]

        # Update nnz_per_row for the corresponding global rows
        # Using scatter_add for efficient batch updates
        self.nnz_per_row[block_row_offset:block_row_offset + n] += counts

        # Calculate global column indices
        global_cols = col_idxs + block_col_offset
        self.col_indices.append(global_cols)

        # Extract the non-zero values
        non_zero_values = values
        self.values.append(non_zero_values)

    def build(self):
        """
        Builds and returns the sparse CSR tensor from the accumulated components.

        Returns:
        - csr_tensor: torch.sparse_csr_tensor, the constructed sparse CSR tensor.
        """
        # Compute crow_indices by cumulatively summing nnz_per_row
        crow_indices = torch.cat([
            self.zero_ten,
            torch.cumsum(self.nnz_per_row, dim=0, dtype=self.dtype)
        ])

        # Convert col_indices and values to tensors
        col_indices_tensor = torch.cat(self.col_indices).to(self.dtype)
        values_tensor = torch.cat(self.values)

        # Create the sparse CSR tensor
        csr_tensor = torch.sparse_csr_tensor(
            crow_indices,
            col_indices_tensor,
            values_tensor,
            size=(self.total_rows, self.total_cols),
        )
        return csr_tensor

    def reset(self):
        self.nnz_per_row = torch.tensor([0] * self.total_rows, device=self.device, dtype=torch.int32)  # Number of non-zero elements per row
        self.col_indices = []                # Column indices of non-zero elements
        self.values = []                     # Non-zero values

    def to_csr(self, A_torch):
        """ Cupy is faster than torch """
        A_cp = cp.asarray(A_torch)
        A_csr_cp = cp.sparse.csr_matrix(A_cp)

        crow_indices = torch.from_dlpack(A_csr_cp.indptr)
        col_indices = torch.from_dlpack(A_csr_cp.indices)
        values = torch.from_dlpack(A_csr_cp.data)
        return crow_indices, col_indices, values


class CSRTransposer:
    def __init__(self, csr_matrix, check_sparsity=False):
        """
        Transposer that transposes CSR matrices efficiently using a precomputed template.
        Strategy: Create CSR matrix with same sparsity, but entries are permutation index.
        Use COO to transpose the matrix and extract the row indices, and new permutation index.
        Finally, use the permutation index and new indices to construct the transposed matrix.

        Args:
            csr_matrix (torch.sparse_csr_tensor): A CSR matrix to extract the sparsity pattern.
            check_sparsity (bool): Whether to check if the input matrix has the same sparsity pattern. False saves memory.
        """
        self.check_sparsity = check_sparsity

        device = csr_matrix.device
        crow_indices = csr_matrix.crow_indices()
        col_indices = csr_matrix.col_indices()
        numel = len(col_indices)

        # Construct a second csr_matrix with same sparsity, but
        csr_temp = torch.sparse_csr_tensor(crow_indices, col_indices, torch.arange(numel, device=device) + 1, csr_matrix.size())
        csr_matrix_T = csr_temp.t().to_sparse_csr()

        self.crow_indices_T = csr_matrix_T.crow_indices().to(torch.int32)
        self.col_indices_T = csr_matrix_T.col_indices().to(torch.int32)
        self.perm_idx_T = csr_matrix_T.values() - 1
        self.size_T = (csr_matrix.size(1), csr_matrix.size(0))  # Transposed size

        if check_sparsity:
            self.crow_indices = crow_indices
            self.col_indices = col_indices
            self.numel = numel

    def transpose(self, csr_matrix):
        """
        Transpose a single CSR matrix using the precomputed template.
        """
        if self.check_sparsity:
            # Ensure the matrix has the same sparsity pattern
            crow_indices = csr_matrix.crow_indices()
            col_indices = csr_matrix.col_indices()
            numel = len(col_indices)
            assert numel == self.numel, f"Matrix has different number of non-zero elements, {numel = } vs {self.numel = }"
            assert torch.equal(crow_indices, self.crow_indices) and torch.equal(col_indices, self.col_indices), "Matrix has different sparsity pattern"

        # Permute values to transposed positions
        values = csr_matrix.values()
        values_T = values[self.perm_idx_T]

        # Create the transposed CSR tensor using the template
        A_T_csr = torch.sparse_csr_tensor(self.crow_indices_T, self.col_indices_T, values_T, size=self.size_T)

        return A_T_csr


class CSRSummer:
    """ Sum together multiple sparse CSR tensors with the same sparsity pattern. """
    def __init__(self, B_list: list[torch.Tensor], check_sparsity=False):
        """
        Precompute the output CSR structure and mappings for efficient summation.
            B_list: List of K initial sparse CSR tensors (torch.sparse_csr_tensor).
        """
        self.check_sparsity = check_sparsity
        self.size = B_list[0].size()
        self.device = B_list[0].device
        self.dtype = B_list[0].dtype

        # Store the initial crow_indices and col_indices of each B_k
        self.initial_crow_indices_list = [B.crow_indices() for B in B_list]
        self.initial_col_indices_list = [B.col_indices() for B in B_list]

        # Precompute the CSR structure and mappings
        self.output_crow_indices, self.output_col_indices, self.index_mapping_list = self.precompute_output_csr_structure(B_list)

    def precompute_output_csr_structure(self, B_list):
        """
        Precompute the CSR structure (crow_indices, col_indices) of the output tensor
        and the mappings from input tensors to output positions.

        Parameters:
        - B_list: List of K sparse CSR tensors.

        Returns:
        - output_crow_indices: crow_indices for the output CSR tensor.
        - output_col_indices: col_indices for the output CSR tensor.
        - index_mapping_list: List of mappings from each B_k's values to output values.
        """
        # Collect all non-zero indices from B_list
        all_row_indices = []
        all_col_indices = []
        nnz_per_tensor = []

        for B in B_list:
            crow_indices = B.crow_indices()
            col_indices = B.col_indices()
            num_rows = crow_indices.size(0) - 1
            row_indices = torch.repeat_interleave(
                torch.arange(num_rows, device=self.device),
                crow_indices[1:] - crow_indices[:-1]
            )
            all_row_indices.append(row_indices)
            all_col_indices.append(col_indices)
            nnz_per_tensor.append(col_indices.size(0))

        # Stack and get unique indices
        all_indices = torch.cat(
            [torch.stack([r, c], dim=1) for r, c in zip(all_row_indices, all_col_indices)],
            dim=0
        )
        unique_indices, inverse_indices = torch.unique(
            all_indices, dim=0, return_inverse=True
        )

        # Sort the unique indices to build CSR structure
        num_cols = self.size[1]
        sorted_order = torch.argsort(unique_indices[:, 0] * num_cols + unique_indices[:, 1])
        sorted_unique_indices = unique_indices[sorted_order]
        row_indices = sorted_unique_indices[:, 0]
        col_indices = sorted_unique_indices[:, 1]

        # Build output_crow_indices
        num_rows = self.size[0]
        row_counts = torch.bincount(row_indices, minlength=num_rows)
        output_crow_indices = torch.zeros(num_rows + 1, dtype=torch.long, device=self.device)
        output_crow_indices[1:] = torch.cumsum(row_counts, dim=0)
        output_col_indices = col_indices

        # Map each input tensor's indices to the output positions
        num_unique = unique_indices.size(0)
        unique_indices_to_sorted = torch.empty(
            num_unique, dtype=torch.long, device=self.device
        )
        unique_indices_to_sorted[sorted_order] = torch.arange(
            num_unique, device=self.device
        )

        cumulative_nnz = [0] + list(
            torch.cumsum(torch.tensor(nnz_per_tensor, device=self.device), dim=0).cpu().numpy()
        )
        index_mapping_list = []

        for k in range(len(B_list)):
            start = cumulative_nnz[k]
            end = cumulative_nnz[k+1]
            indices_k = inverse_indices[start:end]
            positions_k = unique_indices_to_sorted[indices_k]
            index_mapping_list.append(positions_k)

        return output_crow_indices, output_col_indices, index_mapping_list

    def sum(self, B_list_new: list[torch.Tensor]) -> torch.Tensor:
        """
        Sum the values from B_list_new into the output CSR tensor using precomputed mappings.
        Parameters:
        - B_list_new: List of K sparse CSR tensors with new values but same sparsity patterns.
        Returns:
        - J: The output sparse CSR tensor representing J_{ij} = sum_k B_{ijk}.
        """
        if self.check_sparsity:
            # Check that the number of tensors matches
            assert len(B_list_new) == len(self.index_mapping_list), (
                "The number of tensors in B_list_new must match the initial B_list."
            )

            for k, B in enumerate(B_list_new):
                # Ensure the sparsity pattern matches the initial B_k
                if not torch.equal(B.crow_indices(), self.initial_crow_indices_list[k]):
                    raise ValueError(f"Sparsity pattern of B_list_new[{k}] does not match the initial B_list.")
                if not torch.equal(B.col_indices(), self.initial_col_indices_list[k]):
                    raise ValueError(f"Sparsity pattern of B_list_new[{k}] does not match the initial B_list.")

        # Initialize the output values tensor
        nnz_total = self.output_col_indices.size(0)
        output_values = torch.zeros(nnz_total, dtype=self.dtype, device=self.device)

        for k, B in enumerate(B_list_new):
            B_values = B.values()
            positions = self.index_mapping_list[k]
            output_values.index_add_(0, positions, B_values)

        # Create the output CSR tensor
        J = torch.sparse_csr_tensor(
            self.output_crow_indices, self.output_col_indices, output_values, size=self.size
        )
        return J

    def blank_csr(self):
        J = torch.sparse_csr_tensor(self.output_crow_indices, self.output_col_indices, torch.ones_like(self.output_col_indices), size=self.size)
        return J


class CSRRowMultiplier:
    def __init__(self, A_csr: torch.Tensor, check_sparsity=False):
        """
        Initialize the CSRRowMultiplier with a CSR tensor.
        Args:
            A_csr (torch.Tensor): A sparse CSR tensor with fixed sparsity pattern.
            check_sparsity (bool): Whether to check if the input matrix has the same sparsity pattern. False saves memory.
        """
        self.check_sparsity = check_sparsity
        self.A_csr = A_csr
        self.crow_indices = A_csr.crow_indices()
        self.col_indices = A_csr.col_indices()
        self.size = A_csr.size()

        # Precompute row indices for each non-zero element
        row_lengths = self.crow_indices[1:] - self.crow_indices[:-1]
        self.row_indices = torch.arange(self.A_csr.size(0), device=self.A_csr.device).repeat_interleave(row_lengths)

    def mul(self, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Multiply the CSR tensor A row-wise by vector b.
        """
        if self.check_sparsity:
            # Ensure the matrix has the same sparsity pattern
            crow_indices = A.crow_indices()
            col_indices = A.col_indices()
            assert torch.equal(crow_indices, self.crow_indices) and torch.equal(col_indices, self.col_indices), "Matrix has different sparsity pattern"

        # Scale the values by the corresponding row elements
        scaled_values = A.values() * b[self.row_indices]

        # Return a new CSR tensor with the scaled values
        return torch.sparse_csr_tensor(
            self.crow_indices,
            self.col_indices,
            scaled_values,
            size=self.size,
            device=self.A_csr.device
        )


class CSRConcatenator:
    def __init__(self, csr_tensor_A, csr_tensor_B):
        """ Concatenate two CSR tensors with fixed sparsity patterns."""
        device = csr_tensor_A.device

        # Total number of rows and columns
        num_rows_A = csr_tensor_A.shape[0]
        num_rows_B = csr_tensor_B.shape[0]
        total_rows = num_rows_A + num_rows_B
        num_cols = csr_tensor_A.shape[1]

        # Precompute the output crow_indices
        self.output_crow_indices = torch.zeros(total_rows + 1, dtype=torch.int32, device=device)
        self.output_crow_indices[1:num_rows_A + 1] = csr_tensor_A.crow_indices()[1:]
        total_nnz_A = csr_tensor_A.crow_indices()[-1]
        self.output_crow_indices[num_rows_A + 1:] = total_nnz_A + csr_tensor_B.crow_indices()[1:]

        # Precompute the output col_indices
        self.output_col_indices = torch.cat([csr_tensor_A.col_indices(), csr_tensor_B.col_indices()]).to(torch.int32)

        # Store the output shape
        self.output_shape = (total_rows, num_cols)

    def cat(self, csr_A, csr_B):
        values_A = csr_A.values()
        values_B = csr_B.values()

        # Concatenate the values efficiently
        output_values = torch.cat([values_A, values_B])

        # Build and return the output CSR tensor
        output_csr = torch.sparse_csr_tensor(
            self.output_crow_indices,
            self.output_col_indices,
            output_values,
            size=self.output_shape
        )
        return output_csr

    def blank_csr(self):
        J = torch.sparse_csr_tensor(self.output_crow_indices, self.output_col_indices, torch.ones_like(self.output_col_indices), size=self.output_shape)
        return J


class CSRPermuter:
    def __init__(self, perm_to, A_csr):
        """ Precompute matrix permutation.
            perm_to: perm_to[IDX] = VALUE means row VALUE should be moved to IDX / where rows should be moved to.
       """
        self.perm_from = reverse_permutation(perm_to)

        device = A_csr.device
        A_crow_indices = A_csr.crow_indices()
        A_col_indices = A_csr.col_indices()
        A_values = A_csr.values()

        A_csr_permute = torch.sparse_csr_tensor(A_crow_indices, A_col_indices, torch.arange(len(A_values), dtype=torch.float32) + 1, A_csr.size(), device=device)
        perm_mat = permutation_to_csr(self.perm_from, device=device, dtype=torch.float32)

        out_mat = torch.sparse.mm(perm_mat, A_csr_permute)

        self.crow_indices = out_mat.crow_indices().to(torch.int32)
        self.col_indices = out_mat.col_indices().to(torch.int32)
        self.val_permutation = (out_mat.values() - 1).to(torch.int32)
        self.size = out_mat.size()


    def matrix_permute(self, A_csr):
        """ Precomputed permutation of a matrix."""
        A_values = A_csr.values()
        A_values_perm = A_values[self.val_permutation]

        A_permute = torch.sparse_csr_tensor(self.crow_indices, self.col_indices, A_values_perm, self.size)
        return A_permute

    def vector_permute(self, b):
        """ Precomputed permutation of a vector."""
        return b[self.perm_from]


def csr_col_shift(csr_mat, n_cols):
    crow_indices, col_indices, values = csr_mat.crow_indices(), csr_mat.col_indices(), csr_mat.values()
    col_indices = col_indices + n_cols
    deriv_mat_new = torch.sparse_csr_tensor(crow_indices, col_indices, values, size=csr_mat.size(), device=csr_mat.device)

    return deriv_mat_new

def coo_row_select(sparse_coo: torch.sparse_coo_tensor, row_mask) -> torch.sparse_coo_tensor:
    """
    Selects rows from a COO sparse tensor based on a row-wise mask.
    Args:
        sparse_coo (torch.sparse_coo_tensor): The input sparse COO tensor.
        row_mask (torch.Tensor): A boolean mask for selecting rows.
    Returns:
        torch.sparse_coo_tensor: A new sparse COO tensor with only the selected rows.
    """
    # Extract indices and values from the sparse tensor
    indices = sparse_coo.coalesce().indices()  # Shape: [ndim, nnz]
    values = sparse_coo.coalesce().values()  # Shape: [nnz]

    # Assume the first dimension corresponds to rows
    row_indices = indices[0]

    # Create a mask for non-zero elements in the selected rows
    mask = row_mask[row_indices]

    # Apply the mask to filter indices and values
    selected_indices = indices[:, mask]
    selected_values = values[mask]

    # Get the selected row numbers in sorted order
    selected_rows = row_mask.nonzero(as_tuple=False).squeeze()

    # Ensure selected_rows is 1D
    if selected_rows.dim() == 0:
        selected_rows = selected_rows.unsqueeze(0)

    # Create a mapping from old row indices to new row indices
    # This ensures that the new tensor has contiguous row indices starting from 0
    # Example: If rows 1 and 3 are selected, row 1 -> 0 and row 3 -> 1 in the new tensor
    row_mapping = torch.arange(len(selected_rows), device=selected_rows.device)
    # Create a dictionary-like mapping using scatter
    mapping = torch.full((sparse_coo.size(0),), -1, dtype=torch.long, device=selected_rows.device)
    mapping[selected_rows] = row_mapping
    # Map the selected row indices
    new_row_indices = mapping[selected_indices[0]]

    if (new_row_indices == -1).any():
        raise RuntimeError("Some row indices were not mapped correctly.")

    # Replace the row indices with the new row indices
    new_indices = selected_indices.clone()
    new_indices[0] = new_row_indices

    # Define the new size: number of selected rows and the remaining dimensions
    new_size = [row_mask.sum().item()] + list(sparse_coo.size())[1:]

    # Create the new sparse COO tensor
    new_sparse_coo = torch.sparse_coo_tensor(new_indices, selected_values, size=new_size)

    return new_sparse_coo


def coo_col_select(sparse_coo: torch.sparse_coo_tensor, col_mask) -> torch.sparse_coo_tensor:
    """
    Selects columns from a COO sparse tensor based on a column-wise mask.
    Args:
        sparse_coo (torch.sparse_coo_tensor): The input sparse COO tensor.
        col_mask (torch.Tensor): A boolean mask for selecting
    Returns:
        torch.sparse_coo_tensor: A new sparse COO tensor with only the selected columns.
    """
    # Extract indices and values from the sparse tensor
    sparse_coo = sparse_coo.coalesce()  # Ensure indices are coalesced
    indices = sparse_coo.indices()      # Shape: [ndim, nnz]
    values = sparse_coo.values()        # Shape: [nnz]

    # Assume the second dimension corresponds to columns
    col_indices = indices[1]

    # Create a mask for non-zero elements in the selected columns
    mask = col_mask[col_indices]

    # Apply the mask to filter indices and values
    selected_indices = indices[:, mask]
    selected_values = values[mask]

    # Get the selected column numbers in sorted order
    selected_cols = col_mask.nonzero(as_tuple=False).squeeze()

    # Ensure selected_cols is 1D
    if selected_cols.dim() == 0:
        selected_cols = selected_cols.unsqueeze(0)

    # Create a mapping from old column indices to new column indices
    # This ensures that the new tensor has contiguous column indices starting from 0
    row_mapping = torch.arange(len(selected_cols), device=selected_cols.device)
    # Initialize a mapping tensor with -1 (invalid)
    mapping = torch.full((sparse_coo.size(1),), -1, dtype=torch.long, device=selected_cols.device)
    # Assign new indices to the selected columns
    mapping[selected_cols] = row_mapping
    # Map the selected column indices
    new_col_indices = mapping[selected_indices[1]]

    if (new_col_indices == -1).any():
        raise RuntimeError("Some column indices were not mapped correctly.")

    # Replace the column indices with the new column indices
    new_indices = selected_indices.clone()
    new_indices[1] = new_col_indices

    # Define the new size: number of rows remains the same, number of selected columns
    new_size = list(sparse_coo.size())
    new_size[1] = col_mask.sum().item()

    # Create the new sparse COO tensor
    new_sparse_coo = torch.sparse_coo_tensor(new_indices, selected_values, size=new_size)

    return new_sparse_coo


def CSRToInt32(sparse_csr: torch.Tensor) -> torch.Tensor:
    """
    Converts the crow_indices and col_indices of a sparse CSR tensor from int64 to int32.

    Args:
        sparse_csr (torch.Tensor): A PyTorch sparse CSR tensor.

    Returns:
        torch.Tensor: A new sparse CSR tensor with int32 indices and the same values.
    """
    # Extract CSR components
    crow_indices = sparse_csr.crow_indices()
    col_indices = sparse_csr.col_indices()
    values = sparse_csr.values()
    size = sparse_csr.size()
    dtype = values.dtype
    device = sparse_csr.device

    # Convert indices to int32
    crow_indices_int32 = crow_indices.to(torch.int32)
    col_indices_int32 = col_indices.to(torch.int32)

    # Reconstruct the sparse CSR tensor with int32 indices
    sparse_csr_int32 = torch.sparse_csr_tensor(
        crow_indices_int32,
        col_indices_int32,
        values,
        size=size,
        dtype=dtype,
        device=device
    )

    return sparse_csr_int32


def reverse_permutation(indices):
    # Create an empty tensor for the reversed permutation with the same length as indices
    reversed_indices = torch.empty_like(indices)

    # Populate the reversed indices
    for i, target_position in enumerate(indices):
        reversed_indices[target_position] = i
    return reversed_indices


def block_repeat_csr(A: torch.Tensor, k: int) -> torch.Tensor:
    """
    Create a block-diagonal CSR matrix B by repeating a CSR matrix A 'k' times along the diagonal.

    Args:
        A (torch.sparse_csr_tensor): A sparse CSR matrix of shape (m, n).
        k (int): Number of times to repeat A along the diagonal.

    Returns:
        torch.sparse_csr_tensor: The block-diagonal matrix B of shape (k*m, k*n).
    """
    if A.layout != torch.sparse_csr:
        raise ValueError("Input A must be a torch.sparse_csr_tensor.")

    m, n = A.shape
    A_indptr = A.crow_indices()
    A_indices = A.col_indices()
    A_data = A.values()

    # Number of nonzeros in A
    nnz = A_indptr[-1].item()

    # Prepare the arrays for B
    # For B, we'll have k*m rows and k*n cols, and total nnz = k * nnz(A).
    B_m = k * m
    B_n = k * n

    # Allocate storage
    # - B_indptr: length B_m+1
    # - B_indices, B_data: length k * nnz
    B_indptr = torch.empty(B_m + 1, dtype=A_indptr.dtype, device=A.device)
    B_indices = torch.empty(k * nnz, dtype=A_indices.dtype, device=A.device)
    B_data = torch.empty(k * nnz, dtype=A_data.dtype, device=A.device)

    # Fill the B_indptr, B_indices, and B_data
    # Each block i (0-based) corresponds to rows [i*m : (i+1)*m] and columns [i*n : (i+1)*n].
    for i in range(k):
        start_row = i * m
        start_nnz = i * nnz

        # Indptr:
        # For the first block, B_indptr[0:m+1] = A_indptr[0:m+1]
        # For subsequent blocks, shift by i*nnz
        if i == 0:
            B_indptr[0: m + 1] = A_indptr
        else:
            # For block i, row pointers start at B_indptr[i*m]
            # We want B_indptr[i*m+1 : (i+1)*m+1] = A_indptr[1:] + i*nnz
            B_indptr[start_row + 1: start_row + m + 1] = A_indptr[1:] + start_nnz

        # Indices & data:
        # For the indices in block i, we add i*n to the original indices
        # to shift them into the correct column block.
        B_indices[start_nnz: start_nnz + nnz] = A_indices + i * n
        B_data[start_nnz: start_nnz + nnz] = A_data

    # Construct the final CSR matrix B
    B = torch.sparse_csr_tensor(B_indptr, B_indices, B_data, size=(B_m, B_n))
    return B


def stack_coo(X: torch.Tensor, n: int) -> torch.Tensor:
    """
    Vertically stack a sparse COO tensor X with itself n times, and extend column size by n.

    Args:
        X: a torch.sparse_coo_tensor of shape (m, k)
        n: number of times to replicate X vertically

    Returns:
        a torch.sparse_coo_tensor of shape (m*n, k)
    """
    if not X.is_sparse:
        raise ValueError("Input must be a sparse COO tensor")
    if n <= 0:
        raise ValueError("n must be a positive integer")

    # Ensure coalesced for unique indices
    X = X.coalesce()
    indices = X.indices()    # shape (2, nnz)
    values = X.values()      # shape (nnz,)
    m, k = X.size()

    # Prepare lists to collect all replications
    all_indices = []
    all_values = []

    for i in range(n):
        # Copy the indices and shift the row indices by i*m
        idx = indices.clone()
        idx[0] += i * m      # shift row dimension
        all_indices.append(idx)
        all_values.append(values)

    # Concatenate across all replications
    new_indices = torch.cat(all_indices, dim=1)  # now shape (2, n*nnz)
    new_values  = torch.cat(all_values, dim=0)   # shape (n*nnz,)

    # Build the new sparse tensor and coalesce
    result = torch.sparse_coo_tensor(
        new_indices, new_values, size=(m * n, k * n)
    ).coalesce()

    return result

# Example Usage
if __name__ == "__main__":
    import time

    torch.set_printoptions(precision=2, sci_mode=False)
    rows, cols = 10, 10
    density = 0.1

    A_csr = gen_rand_sp_matrix(rows, cols, density).cuda().to_sparse_csr()
    B = block_diagonal_repeat_csr(A_csr, 2)



    plot_sparsity(B)