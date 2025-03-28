import torch

class SparseReshapeMM:
    def __init__(self, M, n, diag, device, zero_rows=None):
        """
        (M_reduced @ A).view(n, n) = (M @ A).view(n, n)

        Initializes the object with a fixed sparse matrix M and the desired output dimension.
        Precomputes a reduced version of M that only contains rows with nonzero entries,
        and computes the corresponding 2D indices (row, col) for the final (n x n) output.

        Parameters:
          M (torch.sparse_coo_tensor): Sparse matrix of shape (n*n, k)
          n (int): such that the full output is reshaped to (n, n)
        """

        self.device = device
        self.only_diagonal = diag
        self.n = n
        self.M_reduced, self.precomputed_indices = self._precompute_reduced_matrix(M, zero_rows)

    def _precompute_reduced_matrix(self, M, zero_rows=None):
        """
        Precomputes a reduced version of the sparse matrix M (in COO format)
        that only keeps rows that contain nonzero elements.

        Parameters:
          M (torch.sparse_coo_tensor): Sparse matrix of shape (n*n, k)

        Returns:
          M_reduced (torch.sparse_coo_tensor): Reduced sparse matrix of shape (num_nonzero_rows, k)
          unique_rows (torch.Tensor): Tensor of original row indices corresponding to M_reduced.
        """
        # Get row indices from M.
        indices = M._indices()
        values = M._values()
        row_indices = indices[0, :]  # all row indices with nonzeros

        # If only_diagonal is True, filter to only keep entries whose flattened row
        # corresponds to a diagonal element in the (n x n) output.
        if self.only_diagonal:
            flat_rows = indices[0, :]
            diag_mask = ((flat_rows // self.n) == (flat_rows % self.n))
            indices = indices[:, diag_mask]
            values = values[diag_mask]
            M = torch.sparse_coo_tensor(indices, values, size=M.shape).coalesce()

            # Update indices after potential filtering.
            indices = M._indices()
            row_indices = indices[0, :]  # flattened row indices with nonzeros

        # Identify unique rows and compute mapping.
        unique_rows, inv_map = torch.unique(row_indices, sorted=True, return_inverse=True)
        new_row_indices = inv_map  # remap each nonzero's row index to the reduced space.

        # Build new indices for the reduced matrix.
        new_indices = torch.stack([new_row_indices, indices[1, :]])
        num_nonzero_rows = unique_rows.numel()

        # Construct the reduced sparse matrix.
        M_reduced = torch.sparse_coo_tensor(new_indices, M._values(), size=(num_nonzero_rows, M.shape[1])).coalesce().to_sparse_csr()

        # Precompute the 2D indices for the output sparse matrix.
        # Each entry in unique_rows (which is an index in [0, n*n)) maps to (row, col) in the n x n matrix.
        rows = unique_rows // self.n
        cols = unique_rows % self.n
        out_diag_mask = (rows == cols)
        precomputed_coo = torch.stack([rows, cols])

        if zero_rows is not None:
            self.bc_mask = torch.isin(rows, zero_rows)
        else:
            self.bc_mask = None

        # --- Precompute the final CSR structure for the output matrix ---
        # Build a dummy COO tensor with these indices and dummy values indicating original ordering.
        nnz = precomputed_coo.size(1)
        dummy_values = torch.arange(nnz, device=self.device)
        dummy_coo = torch.sparse_coo_tensor(precomputed_coo, dummy_values, size=(self.n, self.n)).coalesce()
        # Convert to CSR to get the proper ordering.
        dummy_csr = dummy_coo.to_sparse_csr()
        # Extract the CSR indices.
        csr_crow = dummy_csr.crow_indices()
        csr_col = dummy_csr.col_indices()
        # The dummy values in CSR (dummy_csr.values()) now give the permutation mapping:
        # For each element in the CSR ordering, its value is the original index from dummy_values.
        csr_perm = dummy_csr.values()
        if torch.allclose(csr_perm, dummy_values):
            csr_perm = None
        else:
            out_diag_mask = out_diag_mask[csr_perm]

        return M_reduced, (csr_crow, csr_col, csr_perm, out_diag_mask)

    def multiply(self, phi):
        """
        Computes the product J_flat = M @ phi using the precomputed reduced matrix,
        then reconstructs the full sparse matrix (reshaped to (n, n)) by directly
        setting the values at precomputed positions.

        Parameters:
          phi (torch.Tensor): Dense tensor of shape (k,) or (k, 1)
          tol (float): Tolerance below which values are considered zero (optional).

        Returns:
          torch.sparse_coo_tensor: The result of (M @ phi).view(n, n) in sparse format.
        """
        # Compute the product using the reduced matrix.
        # J_reduced is a dense tensor of shape (num_nonzero_rows, 1).
        J_reduced = torch.mv(self.M_reduced, phi)

        if self.precomputed_indices[2] is not None:
            # Use the precomputed permutation to reorder J_reduced.
            J_reduced = J_reduced[self.precomputed_indices[2]]
        if self.bc_mask is not None:
            J_reduced[self.bc_mask] = 0.
        # Directly create the final sparse tensor using precomputed 2D indices.
        M_phi = torch.sparse_csr_tensor(self.precomputed_indices[0], self.precomputed_indices[1],
                                             J_reduced, size=(self.n, self.n))



        return M_phi, self.precomputed_indices[3]