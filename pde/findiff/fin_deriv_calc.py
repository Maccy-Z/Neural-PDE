import torch

from pde.graph_grid.graph_store import DerivGraph, Deriv
from pde.utils_sparse import coo_row_select, coo_col_select, CSRToInt32, csr_block_repeat, plot_sparsity, coo_interleave, csr_col_shift
from pde.utils_sparse import coo_zero_elements, coo_col_interleave, coo_row_interleave


class FinDerivCalcSPMV:
    """ Using sparse matrix-vector multiplication to compute Grad^n(u) using finite differences. """
    def __init__(self, fd_graphs: dict[tuple, DerivGraph], N_comp, device="cpu"):
        """ Initialise sparse matrices from finite difference graphs.
            Compute d = A * u. Compile eq_mask into A.
            Jacobian is A.T

            pde_mask: PDE points
            bc_mask: Neumann BC points
            N_comp: Number of components in u.
        """
        self.device = device

        # self.N_us_grad = self.grad_mask.sum()
        self.N_deriv = len(fd_graphs)
        self.N_comp = N_comp

        N_Us = self._check(fd_graphs)
        # FD matrix for forward pass to compute derivatives
        self.fd_spms = {}       # shape = [N_deriv], [N_eqs, N_us]
        for order, graph in fd_graphs.items():
            sp_mat = graph.coo().T.coalesce()                   # shape = [N_Us=N_eqs_tot, N_Us]
            self.fd_spms[order] = CSRToInt32(sp_mat.to_sparse_csr())

        # Jacobians for backward pass
        eye_pde = torch.eye(N_Us, device=self.device).to_sparse_coo()
        jac_pde_spms = [eye_pde]        # shape = [N_deriv], [N_Us, N_Us]
        # Other derivatives
        for order, graph in fd_graphs.items():
            sp_mat = graph.coo().T.coalesce()   # shape = [N_Us, N_Us]
            jac_pde_spms.append(sp_mat)

        self.jac_spms = self._process_jacobians(jac_pde_spms)   # shape = [N_deriv_][N_pde_ N_us_]


    def _process_jacobians(self, spms: list[torch.Tensor]) -> list[torch.Tensor]:
        """ Expand Jacobian from 1 component to N_comp.

            spms.shape = [N_deriv][N_Us, N_Us]
            return.shape [N_deriv*N_comp][N_Us*N_comp, N_Us*N_comp]
        """
        jac_spms = []
        for i, jac_deriv in enumerate(spms):
            jac_deriv = jac_deriv.coalesce()
            spm = coo_row_interleave(jac_deriv, self.N_comp)     # shape = [N_us_, N_Us]
            for j in range(self.N_comp):
                # Repeat each row N_component times, and shift columns by j*N_comp
                spm_comp = coo_col_interleave(spm, self.N_comp, col_shifts=j)      # shape = [N_us_, N_Us_]
                jac_spms.append(CSRToInt32(spm_comp.to_sparse_csr()))

        return jac_spms


    def _check(self, fd_graphs):
        """ Check that all graphs have the same shape. """
        graphs = list(fd_graphs.values())
        for graph in graphs:
            assert graph.shape == graphs[0].shape, "All graphs must have the same shape."
            assert graph.shape[0] == graph.shape[1], "Graphs must be square."

        return graphs[0].shape[0]


    def derivative(self, Us, get_orders: list = None) -> dict[tuple, torch.Tensor]:
        """ Xs.shape = [N_points, N_components]
            spm.shape = [N_pde, N_points]
            return.shape = {N_deriv: [N_pde, N_components]}
        """
        Us = Us.contiguous()

        derivatives = {(0, 0): Us.clone()}
        if get_orders is None:
            for order, spm in self.fd_spms.items():
                derivatives[order] = torch.mm(spm, Us)
        else:
            for order in get_orders:
                spm = self.fd_spms[order]
                derivatives[order] = torch.mm(spm, Us)
        return derivatives

    def jacobian(self) -> list[torch.Tensor]:
        """ Linear transform, so jacobian is the same as the sparse matrix.
            return.shape: [N_deriv], [N_Us*N_comp, N_Us*N_comp]
         """
        return self.jac_spms


class BCCalc:
    def __init__(self, bc_specs: dict[int, list[Deriv]], dirich_bc_mask, N_comp: int, N_Us: int, dif_degrees: list[tuple],
                 device="cpu"):
        """ Boundary condition specified as sum_n w_n * dU_i/dX_{j} - C = r -> 0
            bc_specs: {U_idx: [Deriv(orders, comp, weights, value), ...], ...}. Dict of point idx to bc on that point.
            dirich_bc_mask: Mask of Dirichlet BCs in the flattened bc list. shape = [N_bc * N_comp]
            N_comp: Number of components in U.
            N_Us: Total number of U points.
            dif_degrees: List of derivative orders used in the PDE.
        """
        self.device = device
        self.dirich_bc_mask = dirich_bc_mask.flatten()      # shape = [N_bc * N_comp]
        self.N_comp = N_comp
        self.N_bc = len(bc_specs)
        self.N_Us = N_Us
        N_derivs = len(dif_degrees)

        C = []
        for p in bc_specs.values():
            for deriv in p:
                C.append(float(deriv.value))
        self.C = torch.tensor(C, device=device)

        # Convert from indexing per point to indexing per component (like Us_)
        order_to_j = {deg: i for i, deg in enumerate(dif_degrees)}
        idxes, orders, comps, weights = [], [], [], []      # idx.shape = [N_bc], {orders, comps, weights} = [N_bc][N_deriv_per_bc]
        for U_idx, p in bc_specs.items():
            for deriv in p:
                idxes.append(U_idx)
                diff_deg = [order_to_j[order] for order in deriv.orders]
                orders.append(torch.tensor(diff_deg))
                comps.append(torch.tensor(deriv.comp))
                weights.append(torch.tensor(deriv.weights, dtype=torch.float32))

        # Flatten indices into 1D list for vectorised computation.
        # lengths per batch item
        lens = torch.tensor([len(o) for o in orders], device=device, dtype=torch.long)      # shape = [N_idx]
        self.N_bc_ = len(lens)          # = N_bc * N_comp
        assert len(comps) == self.N_bc_ and len(weights) == self.N_bc_
        assert self.N_bc_ == self.N_bc * self.N_comp

        # N_idx = sum (N_deriv_per_bc)
        # concat ragged lists
        self.orders_cat = torch.as_tensor([x for r in orders for x in r], device=device, dtype=torch.long)      # shape = [N_idx]
        self.comps_cat = torch.as_tensor([x for r in comps for x in r], device=device, dtype=torch.long)        # shape = [N_idx]
        self.weights_cat = torch.as_tensor([x for r in weights for x in r], device=device)                      # shape = [N_idx]
        # which batch item each concatenated element belongs to
        self.batch_ids = torch.repeat_interleave(torch.arange(self.N_bc_, device=device), lens)  # shape = [N_idx]
        # map to the first dimension indices
        idxes_t = torch.as_tensor(idxes, device=device, dtype=torch.long)
        self.first_dim = idxes_t[self.batch_ids]                                                            # shape = [N_idx]

        # Construct Jacobian matrix for BCs
        # J = dR/dU_dUs. J.shape = [N_bc_, N_derivs_]
        rows = self.batch_ids
        cols = self.comps_cat + self.orders_cat * N_comp
        indices_2d = torch.stack([rows, cols], dim=0)
        size_2d = (self.N_bc * self.N_comp, N_derivs * self.N_comp)
        # Build directly as a 2D sparse matrix (COO), then coalesce to sum duplicates & sort
        J_coo = torch.sparse_coo_tensor(
            indices_2d,
            self.weights_cat,
            size_2d,
            dtype=torch.float32,
            device=self.device,
        )
        self.J = J_coo.to_dense()


    def forward(self, U_dUs: torch.Tensor):
        """ Returns residauals for BCs.
            U_dUs.shape = [N_Us, N_derivs, N_comp]
            Output = sum U_dUs[idxes, orders, comps] * weights[idxs] - C
            return.shape = [N_bc, N_comp]
        """

        vals = U_dUs[self.first_dim, self.orders_cat, self.comps_cat] * self.weights_cat
        out = torch.zeros(self.N_bc_, device=self.device)
        out.index_add_(0, self.batch_ids, vals)  # per-batch sum

        out = out - self.C
        return out

    def resid_jac(self, U_dUs: torch.Tensor):
        """ Return residauls and jacobian for BCs.
            U_dUs.shape = [N_Us, N_derivs, N_comp]
            return: [N_bc_, N_derivs_], [N_bc, N_comp]
        """

        residuals = self.forward(U_dUs)
        return self.J, residuals

    def dirich_bc_vals(self):
        """ Return Dirichlet BC values C. """
        return self.C[self.dirich_bc_mask]