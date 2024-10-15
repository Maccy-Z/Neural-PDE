import torch
from torch.autograd import Function

class SparseCSRTransposer:
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
            assert numel == self.numel, "Matrix has different number of non-zero elements"
            assert torch.equal(crow_indices, self.crow_indices) and torch.equal(col_indices, self.col_indices), "Matrix has different sparsity pattern"

        # Permute values to transposed positions
        values = csr_matrix.values()
        values_T = values[self.perm_idx_T]

        # Create the transposed CSR tensor using the template
        A_T_csr = torch.sparse_csr_tensor(self.crow_indices_T, self.col_indices_T, values_T, size=self.size_T)

        return A_T_csr


def gen_rand_sp_matrix(rows, cols, density, device="cpu"):
    num_nonzeros = int(rows * cols * density)
    row_indices = torch.randint(0, rows, (num_nonzeros,))
    col_indices = torch.randint(0, cols, (num_nonzeros,))
    values = torch.randn(num_nonzeros)  # Random values for the non-zero entries

    edge_index = torch.stack([row_indices, col_indices], dim=0)
    return torch.sparse_coo_tensor(edge_index, values, (rows, cols)).to(device)


class SparseMatMul(Function):
    @staticmethod
    def forward(ctx, A, A_T, b):
        """
        Forward pass for sparse matrix-vector multiplication.
        Args:
            A (torch.sparse_csr_tensor): Sparse CSR matrix of shape (m, n).
            A_T (torch.sparse_csr_tensor): Transpose of A, shape (n, m).
            b (torch.Tensor): Dense vector of shape (n,).
        Returns:
            torch.Tensor: Resulting vector of shape (m,).
        """
        # Save the transposed sparse matrix for backward
        ctx.save_for_backward(A_T)

        # Perform sparse-dense matrix multiplication
        output = torch.mv(A, b)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass to compute gradient with respect to b.
        Args: grad_output (torch.Tensor): Gradient of the loss with respect to the output, shape (m,).
        Returns:
            Tuple[None, None, torch.Tensor]: Gradients with respect to inputs.
                                             None for A and A_T (since they are constants), and gradient with respect to b.
        """
        (A_T,) = ctx.saved_tensors

        # Compute gradient with respect to b: A^T * grad_output
        grad_b = torch.mv(A_T, grad_output) # torch.sparse.mm(A_T, grad_output.unsqueeze(1)).squeeze(1)

        return None, None, grad_b


class SparseMatMulOperator:
    def __init__(self, A):
        """
        Initialize the operator with a sparse CSR matrix A.
        Args: A (torch.sparse_csr_tensor): Sparse CSR matrix of shape (m, n).
        """
        # self.A = A
        self.A = torch.sparse_csr_tensor(A.crow_indices().to(torch.int32), A.col_indices().to(torch.int32), A.values(), A.size())
        # Compute the transpose once and cache it
        self.A_T = self.A.t().to_sparse_csr() # .coalesce_csr()


    def matmul(self, b):
        """
        Perform the multiplication A * b.
        Args: b (torch.Tensor): Dense vector of shape (n,).
        Returns: torch.Tensor: Resulting vector of shape (m,).
        """
        return SparseMatMul.apply(self.A, self.A_T, b)

    def __repr__(self):
        return f"SparseMatMulOperator with tensor: {self.A}"


class SparseTensorSummer:
    """ Sum together multiple sparse CSR tensors with the same sparsity pattern. """
    def __init__(self, B_list):
        """
        Precompute the output CSR structure and mappings for efficient summation.
        Parameters:
        - B_list: List of K initial sparse CSR tensors (torch.sparse_csr_tensor).
        """
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

    def sum_tensors(self, B_list_new) -> torch.Tensor:
        """
        Sum the values from B_list_new into the output CSR tensor using precomputed mappings.
        Parameters:
        - B_list_new: List of K sparse CSR tensors with new values but same sparsity patterns.
        Returns:
        - J: The output sparse CSR tensor representing J_{ij} = sum_k B_{ijk}.
        """
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

# Example Usage
if __name__ == "__main__":
    import time

    torch.set_printoptions(precision=2, sci_mode=False)
    rows, cols = 100_000, 100_000
    density = 0.001

    A_coo = generate_random_sparse_matrix(rows, cols, density).cuda()
    A_csr = A_coo.to_sparse_csr()
    print("Original CSR Matrix A:")
    print()

    # Initialize the transposer with the first matrix
    print("Precomputing Transposer...")
    transposer = SparseCSRTransposer(A_csr, check_sparsity=False)
    print("Transposer Ready")

    # Create multiple CSR matrices with the same sparsity pattern but different values
    for i in range(5):
        print(i)
        # Generate different values for each matrix
        new_vals = torch.randn_like(A_csr.values())
        A_csr_new = torch.sparse_csr_tensor(A_csr.crow_indices(), A_csr.col_indices(), new_vals, A_csr.size()).cuda()

        torch.cuda.synchronize()
        st = time.time()
        A_t = transposer.transpose(A_csr_new)
        torch.cuda.synchronize()
        print(f"Time taken = {time.time() - st}")

        # A_t_coo = A_csr_new.to_sparse_coo().t()

        # print(coo_equal(A_t_coo, A_t.to_sparse_coo()))

