import torch

from pde.graph_grid.graph_store import DerivGraph, Deriv
from pde.BaseDerivCalc import BaseDerivCalc
from pde.utils_sparse import coo_row_select, coo_col_select, CSRToInt32, csr_block_repeat, plot_sparsity, coo_interleave, csr_col_shift
from pde.utils_sparse import coo_zero_elements, coo_col_interleave, coo_row_interleave


class FinDerivCalcSPMV(BaseDerivCalc):
    """ Using sparse matrix-vector multiplication to compute Grad^n(u) using finite differences. """
    def __init__(self, fd_graphs: dict[tuple, DerivGraph], N_comp, device="cpu"):
        """ Initialise sparse matrices from finite difference graphs.
            Compute d = A * u [eq_mask]. Compile eq_mask into A.
            Jacobian is A[eq_mask][us_mask]

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
        derivatives = {(0, 0): Us}
        Us = Us.contiguous()

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
    def __init__(self, bc_specs: dict[int, list[Deriv]], N_comp: int, N_Us: int, dif_degrees: list[tuple],
                 device="cpu"):
        """ Boundary condition specified as sum_n w_n * dU_i/dX_{j} - C = r -> 0"""
        self.device = device

        self.N_comp = N_comp
        self.N_bc = len(bc_specs)
        self.N_Us = N_Us
        order_to_j = {deg: i for i, deg in enumerate(dif_degrees)}
        N_derivs = len(dif_degrees)


        C = []
        for p in bc_specs.values():
            for deriv in p:
                C.append(float(deriv.value))
        self.C = torch.tensor(C, device=device)

        # Convert from indexing per point to indexing per component (like Us_)
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
        self.weights_cat = torch.as_tensor([x for r in weights for x in r], device=device)
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


# class NeumanBCCalc(FinDerivCalcSPMV):
#     """ Compute FinDiff derivatives for (linear) Neumann BCs, and full jacobian for R = sum_n grad_n(u) - constant.
#         Precompute the selection derivatives and jacobian, that directly returns residuals / residual jacobian without going through autograd / sparse matmuls
#     """
#     def __init__(self, fd_graphs: dict[tuple, DerivGraph], bc_eq_mask: torch.Tensor, grad_mask: torch.Tensor, deriv_orders: dict[int, Deriv],
#                  N_comp, device="cpu"):
#         """
#             deriv_orders: Derivative order for each derivative BC
#         """
#         # Construct all required derivatives and jacobian.
#         super().__init__(fd_graphs, bc_eq_mask, grad_mask, N_comp, device=device)
#         N_bc_eqs = sum(bc_eq_mask)
#         N_Us = bc_eq_mask.shape[0]
#         N_Us_tot = bc_eq_mask.shape[0] * N_comp
#
#         # self.fd_spms[(1, 0)].shape = [N_bc_eqs, N_us_tot]
#         # Reshape to blocks, which allows for mixing up the derivatives.
#         fd_spms_old = {order: csr_block_repeat(spm, N_comp) for order, spm in self.fd_spms.items()}    # shape = [N_derivs][N_bc_eqs*N_comp, N_us_tot*N_comp]
#         orders = [(0, 0)] + list(self.fd_spms.keys())  # Include zeroth order (original value) in orders.
#         order_dict = {o: i for i, o in enumerate(orders)}  # Map order to index.
#         # For (0, 0) component, make diagonal matrix x
#         zeroth_order_idx = torch.where(self.eq_mask.repeat(N_comp))[0]
#
#         # Build up full derivative matrix, combining all derivatives. [sum_n(w_n * deriv_n)] u - N = 0
#         deriv_mat = []          # shape = [N_bc_points*N_comp, N_us_tot*N_comp]. Ordered in component major (points grouped).
#
#         dRdD = torch.zeros((N_comp * N_bc_eqs, (self.N_deriv+1) * N_comp), device=self.device)
#         for bc_idx, derivs in enumerate(deriv_orders.values()):
#             # For each boundary condition:
#             for bc_x_idx, deriv_bc in enumerate(derivs):
#                 deriv_row = []
#                 # For each component of boundary condition:
#                 for component, order, weight in zip(deriv_bc.comp, deriv_bc.orders, deriv_bc.weights, strict=True):
#                     row = N_comp * bc_idx + bc_x_idx
#                     col = N_comp * order_dict[order] + component
#                     dRdD[row, col] = weight
#
#                     us_idx = bc_idx + component * N_bc_eqs
#                     if order == (0, 0):
#                         indices = torch.tensor([[zeroth_order_idx[us_idx]]])
#                         row_val = torch.sparse_coo_tensor(indices=indices, values=torch.tensor([weight], dtype=torch.float32), device=self.device, size=(N_Us_tot,))
#                         deriv_row.append(row_val)
#                     else:
#                         # Get the derivative matrix for this component and order.
#                         spm = fd_spms_old[order][us_idx] * weight
#                         deriv_row.append(spm)
#
#
#                 deriv_row = torch.stack(deriv_row, dim=0)
#                 deriv_row_sum = torch.sparse.sum(deriv_row, dim=0)
#                 deriv_mat.append(deriv_row_sum)
#
#         deriv_mat = torch.stack(deriv_mat, dim=0).coalesce()
#         self.deriv_mat = CSRToInt32(deriv_mat.to_sparse_csr())
#         self.jac_mat = coo_col_select(deriv_mat, self.grad_mask.repeat(N_comp)).to_sparse_csr()   # shape = [N_bc_eqs_, N_grad_]
#         self.jac_mat = CSRToInt32(self.jac_mat)
#         self.dRdD = dRdD        # shape = [N_bc_eqs*N_comp, N_deriv*N_comp]
#         self.N_bc_eqs = N_bc_eqs
#
#         # del self.fd_spms
#         del self.jac_spms
#
#
#     def residuals(self, Us) -> torch.Tensor:
#         """ Us.shape = [N_us_tot, N_components]
#             return.shape = [N_neumann*N_components]
#         """
#         Us2 = Us.T.flatten()
#         self.deriv_mat = self.deriv_mat.to(Us2.device)
#         bc_grads = torch.mv(self.deriv_mat, Us2)
#         # return bc_grads
#         grads_dict = self.derivative(Us)
#         U_dUs = torch.stack(list(grads_dict.values()), dim=1)    # shape = [N_bc_eqs, N_derivs, N_component]
#         U_dUs = U_dUs.view(self.N_bc_eqs, self.N_comp *(self.N_deriv + 1))  # shape = [N_bc_eqs, N_comp*(N_derivs+1)]
#         U_dUs = torch.repeat_interleave(U_dUs, self.N_comp, dim=0)
#
#         lhs = self.dRdD * U_dUs
#         lhs = lhs.sum(dim=1)  # shape = [N_bc_eqs*N_comp]
#         return lhs
#         # print(residuals.shape )
#         # exit(8)
#         # pass
#
#     def jacobian(self) -> torch.Tensor:
#         return self.jac_mat
#
