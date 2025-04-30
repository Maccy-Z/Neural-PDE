import torch
# from torch_geometric.nn import MessagePassing

from pde.graph_grid.graph_store import DerivGraph, Deriv
from pde.BaseDerivCalc import BaseDerivCalc
from pde.utils_sparse import coo_row_select, coo_col_select, CSRToInt32, block_repeat_csr, plot_sparsity


# class FinDerivCalc(MessagePassing, BaseDerivCalc):
#     """ Compute Grad^n(u) using finite differences. """
#
#     def __init__(self, fd_graphs: dict[tuple, DerivGraph], pde_mask):
#         """ edge_index (dict[tuple, Tensor]): Graph connectivity in COO format with shape [2, E].
#             edge_coeffs (dict[tuple, Tensor]): Finite difference coefficients for each edge."""
#         super().__init__(aggr='add')  # Aggregation method can be 'add', 'mean', etc.
#
#         self.fd_graphs = fd_graphs
#         self.pde_mask = pde_mask
#
#     def derivative(self, Xs) -> dict[tuple, torch.Tensor]:
#         return self(Xs)
#
#     def forward(self, Xs) -> dict[tuple, torch.Tensor]:
#         """
#         Args:
#             x (Tensor): Node feature matrix of shape [N, F].
#         """
#         # Include original node value
#         derivatives = {(0, 0): Xs[self.pde_mask]}
#         for order, graph in self.fd_graphs.items():
#             edge_idx = graph.edge_idx
#             coeff = graph.weights
#
#             derivatives[order] = self.propagate(edge_idx, x=Xs.unsqueeze(-1), edge_coeff=coeff.unsqueeze(-1))[self.pde_mask].squeeze()
#         return derivatives
#
#     def message(self, x_j, edge_coeff):
#         """
#         Constructs messages from source nodes to target nodes.
#
#         Args:
#             x_j (Tensor): Source node features for each edge of shape [E, F].
#             edge_coeff (Tensor): Edge coefficients for each edge of shape [E, 1] or [E, F].
#
#         Returns:
#             Tensor: Messages to be aggregated.
#         """
#         return  edge_coeff * x_j


class FinDerivCalcSPMV(BaseDerivCalc):
    """ Using sparse matrix-vector multiplication to compute Grad^n(u) using finite differences. """
    def __init__(self, fd_graphs: dict[tuple, DerivGraph], eq_mask: torch.Tensor, grad_mask: torch.Tensor, N_component, device="cpu"):
        """ Initialise sparse matrices from finite difference graphs.
            Compute d = A * u [eq_mask]. Compile eq_mask into A.
            Jacobian is A[eq_mask][us_mask]

            eq_mask: Points where constraint functions needed (PDEs / BCs)
            grad_mask: u points that need to be updated. Used for Jacobian.
            N_us_tot: Total number of u points on graph.
            N_components: Number of components in u.
        """
        self.eq_mask = eq_mask
        self.grad_mask = grad_mask
        self.device = device

        N_us_tot = self._check(fd_graphs)

        self.fd_spms = {}       # shape = [N_deriv], [N_eqs, N_us_tot]
        self.jac_spms = []      # shape = [N_deriv], [N_eqs, N_grad]

        # Order (0, 0) is original node value
        only_us = torch.eye(N_us_tot, device=self.device).to_sparse_coo()
        only_us = coo_row_select(only_us, self.eq_mask)
        only_us = coo_col_select(only_us, self.grad_mask)
        self.jac_spms.append(only_us.to_sparse_csr())   # shape = [N_deriv], [N_eqs, N_us_tot]

        for order, graph in fd_graphs.items():
            sp_mat = graph.coo().T.coalesce()
            sp_mat = coo_row_select(sp_mat, self.eq_mask)        # shape = [N_eqs, N_us_tot]
            self.fd_spms[order] = CSRToInt32(sp_mat.to_sparse_csr())

            sp_mat_jac = coo_col_select(sp_mat, self.grad_mask)   # shape = [N_eqs, N_grad]
            sp_mat_jac = CSRToInt32(sp_mat_jac.to_sparse_csr())
            self.jac_spms.append(sp_mat_jac)

        # Repeat jacobian to shape [N_deriv][N_eqs*N_comp, N_grad*N_comp]
        for i, jac_deriv in enumerate(self.jac_spms):
            self.jac_spms[i] = block_repeat_csr(jac_deriv, N_component)

        self.N_deriv = len(self.fd_spms)

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
        derivatives = {(0, 0): Us[self.eq_mask]}
        if get_orders is None:
            for order, spm in self.fd_spms.items():
                Us = Us.contiguous()
                derivatives[order] = torch.mm(spm, Us)
        else:
            for order in get_orders:
                spm = self.fd_spms[order]
                Us = Us.contiguous()
                derivatives[order] = torch.mm(spm, Us)
        return derivatives

    def jacobian(self) -> list[torch.FloatTensor]:
        """ Linear transform, so jacobian is the same as the sparse matrix.
            return.shape: [N_deriv], [N_pde*N_comp, N_points*N_comp]
         """
        return self.jac_spms


class NeumanBCCalc(FinDerivCalcSPMV):
    """ Compute FinDiff derivatives for (linear) Neumann BCs, and full jacobian for R = sum_n grad_n(u) - constant.
        Precompute the selection derivatives and jacobian, that directly returns residuals / residual jacobian without going through autograd / sparse matmuls
    """
    def __init__(self, fd_graphs: dict[tuple, DerivGraph], eq_mask: torch.Tensor, grad_mask: torch.Tensor, deriv_orders: dict[int, Deriv],
                 N_comp, device="cpu"):
        """
            deriv_orders: Derivative order for each derivative BC
        """
        # Construct all required derivatives and jacobian.
        super().__init__(fd_graphs, eq_mask, grad_mask, N_comp, device=device)
        N_bc_eqs = sum(eq_mask)

        # self.fd_spms[(1, 0)].shape = [N_bc_eqs, N_us_tot]
        # Reshape to blocks, which allows for mixing up the derivatives.
        self.fd_spms = {order: block_repeat_csr(spm, N_comp) for order, spm in self.fd_spms.items()}    # shape = [N_derivs][N_bc_eqs*N_comp, N_us_tot*N_comp]

        # Build up full derivative matrix, combining all derivatives. [sum_n(deriv_n)] u - N = 0
        deriv_mat = []          # shape = [N_bc_points*N_comp, N_us_tot*N_comp]. Ordered in component major (points grouped).
        for eq_idx, derivs in enumerate(deriv_orders.values()):
            # For each boundary condition:
            for deriv_bc in derivs:
                deriv_row = []
                # For each component of boundary condition:
                for component, order in zip(deriv_bc.comp, deriv_bc.orders):
                    us_idx = eq_idx + component * N_bc_eqs
                    deriv_row.append(self.fd_spms[order][us_idx])

                deriv_row = torch.stack(deriv_row, dim=0)
                deriv_row_sum = torch.sparse.sum(deriv_row, dim=0)
                deriv_mat.append(deriv_row_sum)

        deriv_mat = torch.stack(deriv_mat, dim=0).coalesce()
        self.deriv_mat = CSRToInt32(deriv_mat.to_sparse_csr())
        self.jac_mat = coo_col_select(deriv_mat, self.grad_mask.repeat(N_comp)).to_sparse_csr()   # shape = [N_bc_eqs_, N_grad_]
        self.jac_mat = CSRToInt32(self.jac_mat)


        del self.fd_spms
        del self.jac_spms


    def derivative(self, Us) -> torch.Tensor:
        """ Us.shape = [N_us_tot, N_components]
            return.shape = [N_neumann*N_components]
        """
        Us = Us.T.flatten()
        bc_grads = torch.mv(self.deriv_mat, Us)
        return bc_grads

    def jacobian(self) -> torch.Tensor:
        return self.jac_mat