import torch

import torch


class CSRSystemSimplifier:
    """
    Remove a fixed set of rows/columns from a CSR system and later rebuild the
    full solution vector — *even if the removable rows have multiple candidate
    columns in the pattern*.

    Parameters
    ----------
    A_sample_csr : torch.sparse_csr_tensor
        Any matrix whose crow/col indices match every system you will solve.
        Its *values* are ignored.
    rows_to_remove : 1-D Bool mask **or** 1-D Long index list
        The rows that are always removed.
    pivot_cols_for_rows : 1-D Long
        `pivot_cols_for_rows[i]` is the column that will be non-zero in
        `rows_to_remove[i]`.  Must be unique over the list.
    """

    # ─────────────────────────────── set-up ────────────────────────────────
    def __init__(
        self,
        A_sample_csr: torch.Tensor,
        rows_to_remove,
        pivot_cols_for_rows: torch.Tensor,
    ):
        if not A_sample_csr.is_sparse_csr:
            raise TypeError("A_sample_csr must be a torch.sparse_csr_tensor")

        self.device, self.dtype = A_sample_csr.device, A_sample_csr.dtype
        self.n_rows, self.n_cols = A_sample_csr.shape

        # ---------- canonicalise rows_to_remove → mask + index list -------------
        if isinstance(rows_to_remove, torch.Tensor) and rows_to_remove.dtype == torch.bool:
            if rows_to_remove.numel() != self.n_rows:
                raise ValueError("Boolean mask length must equal number of rows")
            self.rows_to_remove = torch.where(rows_to_remove.to(self.device))[0]
            self.single_nz_row_mask = rows_to_remove.to(self.device)
        else:  # treat as index tensor / list
            self.rows_to_remove = torch.as_tensor(
                rows_to_remove, dtype=torch.long, device=self.device
            )
            self.single_nz_row_mask = torch.zeros(
                self.n_rows, dtype=torch.bool, device=self.device
            )
            self.single_nz_row_mask[self.rows_to_remove] = True

        # ---------- pivot columns (user-supplied) -------------------------------
        self.pivot_cols = torch.as_tensor(
            pivot_cols_for_rows, dtype=torch.long, device=self.device
        )
        if self.pivot_cols.numel() != self.rows_to_remove.numel():
            raise ValueError("pivot_cols_for_rows must match rows_to_remove length")
        if self.pivot_cols.unique().numel() != self.pivot_cols.numel():
            raise ValueError("Pivot columns must be distinct")

        # ---------- find pivot *positions* in the value array -------------------
        # (done once; the indices are fixed by the pattern)
        row_ptr = A_sample_csr.crow_indices()          # (n_rows+1,)
        col_idx = A_sample_csr.col_indices()           # (nnz,)

        pivot_pos = []
        for r, c in zip(self.rows_to_remove.tolist(), self.pivot_cols.tolist()):
            start, end = row_ptr[r].item(), row_ptr[r + 1].item()
            match = (col_idx[start:end] == c).nonzero(as_tuple=False)
            if match.numel() == 0:
                raise ValueError(
                    f"Row {r}: pivot column {c} not present in sample pattern"
                )
            # take the first (only) match in that slice
            pivot_pos.append(start + match[0, 0].item())
        self.pivot_positions = torch.tensor(
            pivot_pos, dtype=torch.long, device=self.device
        )

        # ---------- masks & compact-index maps ----------------------------------
        self.row_mask = ~self.single_nz_row_mask
        self.col_mask = torch.ones(self.n_cols, dtype=torch.bool, device=self.device)
        self.col_mask[self.pivot_cols] = False

        self.n_rows_next = int(self.row_mask.sum())
        self.n_cols_next = int(self.col_mask.sum())

        self.col_compact = torch.full(
            (self.n_cols,), -1, dtype=torch.long, device=self.device
        )
        self.col_compact[self.col_mask] = torch.arange(
            self.n_cols_next, device=self.device
        )

        # ---------- fixed sparsity pattern of the reduced matrix ----------------
        A_coo = A_sample_csr.to_sparse_coo()
        r_all, c_all = A_coo.indices()
        keep_nz = self.row_mask[r_all] & self.col_mask[c_all]

        # compact rows: running count of kept rows, minus 1 (0-based)
        row_running = torch.cumsum(self.row_mask.to(torch.int), 0) - 1
        self._next_idx = torch.vstack(
            [row_running[r_all[keep_nz]], self.col_compact[c_all[keep_nz]]]
        )
        self._keep_nz = keep_nz

        # ---------- container for latest solved pivot values --------------------
        self._latest_solved_vals = None

    # ─────────────────────── build reduced system ─────────────────────────
    def __call__(self, A_csr: torch.Tensor, b: torch.Tensor):
        """
        Returns
        -------
        A_next : torch.sparse_csr_tensor
        b_next : torch.Tensor
        solved_vals : torch.Tensor   (pivot values in row order)
        """
        # -------- pivot variable values ----------------------------------------
        pivot_vals = A_csr.values()[self.pivot_positions]
        solved_vals = b[self.rows_to_remove] / pivot_vals
        self._latest_solved_vals = solved_vals  # cache for reconstruction

        # -------- update RHS ----------------------------------------------------
        solved_vec = torch.zeros(self.n_cols, dtype=self.dtype, device=self.device)
        solved_vec[self.pivot_cols] = solved_vals
        b_updated = b - (A_csr @ solved_vec)

        # -------- build reduced matrix -----------------------------------------
        new_vals = A_csr.values()[self._keep_nz]
        A_next = torch.sparse_coo_tensor(
            self._next_idx,
            new_vals,
            size=(self.n_rows_next, self.n_cols_next),
            dtype=self.dtype,
            device=self.device,
        ).to_sparse_csr()

        b_next = b_updated[self.row_mask]
        return A_next, b_next, solved_vals

    # ───────────────────── recover full solution ─────────────────────────
    def recover_full_solution(self, x_reduced: torch.Tensor, solved_vals=None):
        """
        Expand `x_reduced` (length = n_cols_next) into `x_full` (length = n_cols)
        by inserting the pivot values.
        """
        if solved_vals is None:
            if self._latest_solved_vals is None:
                raise RuntimeError(
                    "No cached pivot values. Supply `solved_vals`, or call "
                    "the eliminator first."
                )
            solved_vals = self._latest_solved_vals

        if x_reduced.numel() != self.n_cols_next:
            raise ValueError("x_reduced length mismatch")

        x_full = torch.zeros(self.n_cols, dtype=self.dtype, device=self.device)
        x_full[self.col_mask] = x_reduced          # surviving columns
        x_full[self.pivot_cols] = solved_vals      # eliminated columns
        return x_full



def test_vectorized_simplification():
    print("Running test_vectorized_simplification...\n")
    A_current = torch.tensor([
        [2.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 5.0, 0.0],  # Row 1: 5*x2 = 10  => x2 = 2
        [1.0, 1.0, 1.0, 1.0],
        [0.0, 8.0, 0.0, 0.0]  # Row 3: 3*x1 = 9   => x1 = 3
    ], dtype=torch.float32)

    b_current = torch.tensor([7.0, 10.0, 8.0, 9.0], dtype=torch.float32)

    # Mask identifying rows with a single non-zero entry
    # Rows 1 and 3 have single non-zero entries.

    print("Initial A:\n", A_current)
    print("Initial b:\n", b_current)

    # Expected results based on manual walkthrough:
    # Solved: x1=3 (from row 3), x2=2 (from row 1)
    # Remaining system for x0, x3:
    # 2*x0 + 1*x3 = 7 - (1*x1 + 0*x2) = 7 - (3) = 4


    simplifier = CSRSystemSimplifier(A_current.to_sparse_csr(), torch.tensor([1, 3]), torch.tensor([2, 1]))
    A_next, b_next, solved_vals = simplifier(A_current.to_sparse_csr(), b_current)

    x = torch.linalg.solve(A_current, b_current)
    print(x)

    x_reduced = torch.linalg.solve(A_next.to_dense(), b_next)
    print(x_reduced)
    x_reconstruct = simplifier.recover_full_solution(x_reduced, solved_vals)

    print(x_reconstruct)




if __name__ == '__main__':
    # To run this test, you'd need the definition of
    # simplify_square_system_vectorized_step in this file or imported.
    # For demonstration, I'll include a placeholder for the function.


    test_vectorized_simplification()