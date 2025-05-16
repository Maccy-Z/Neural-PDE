import torch

from pde.graph_grid.graph_store import DerivGraph, Deriv
from pde.BaseDerivCalc import BaseDerivCalc
from pde.utils_sparse import coo_row_select, coo_col_select, CSRToInt32, block_repeat_csr, plot_sparsity, stack_coo, csr_col_shift


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
        self.N_us_grad = self.grad_mask.sum()
        self.device = device

        N_us_tot = self._check(fd_graphs)

        self.fd_spms = {}       # shape = [N_deriv], [N_eqs, N_us_tot]
        jac_spms = []      # shape = [N_deriv], [N_eqs, N_grad]

        # Order (0, 0) is original node value
        only_us = torch.eye(N_us_tot, device=self.device).to_sparse_coo()
        only_us = coo_row_select(only_us, self.eq_mask)
        only_us = coo_col_select(only_us, self.grad_mask)
        jac_spms.append(only_us)   # shape = [N_deriv], [N_eqs, N_us_tot]

        for order, graph in fd_graphs.items():
            sp_mat = graph.coo().T.coalesce()
            sp_mat = coo_row_select(sp_mat, self.eq_mask)        # shape = [N_eqs, N_us_tot]
            self.fd_spms[order] = CSRToInt32(sp_mat.to_sparse_csr())

            sp_mat_jac = coo_col_select(sp_mat, self.grad_mask)   # shape = [N_eqs, N_grad]
            jac_spms.append(sp_mat_jac)

        # Repeat jacobian to shape [N_deriv][N_eqs*N_comp, N_grad*N_comp]
        for i, jac_deriv in enumerate(jac_spms):
            spm = stack_coo(jac_deriv, N_component)
            jac_spms[i] = spm.to_sparse_csr()

        self.jac_spms = []
        # Jacobians need to be modified for each component
        for U_comp in range(N_component):
            for d_comp, jac_single in enumerate(jac_spms):
                new_jac = csr_col_shift(jac_single, U_comp * self.N_us_grad)
                self.jac_spms.append(CSRToInt32(new_jac))

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
        Us = Us.contiguous()

        if get_orders is None:
            for order, spm in self.fd_spms.items():
                derivatives[order] = torch.mm(spm, Us)
        else:
            for order in get_orders:
                spm = self.fd_spms[order]
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
        N_us_tot = eq_mask.shape[0]

        # self.fd_spms[(1, 0)].shape = [N_bc_eqs, N_us_tot]
        # Reshape to blocks, which allows for mixing up the derivatives.
        self.fd_spms = {order: block_repeat_csr(spm, N_comp) for order, spm in self.fd_spms.items()}    # shape = [N_derivs][N_bc_eqs*N_comp, N_us_tot*N_comp]
        # For (0, 0) component, make diagonal matrix x
        zeroth_order_idx = torch.where(self.eq_mask.repeat(N_comp))[0]

        # Build up full derivative matrix, combining all derivatives. [sum_n(deriv_n)] u - N = 0
        deriv_mat = []          # shape = [N_bc_points*N_comp, N_us_tot*N_comp]. Ordered in component major (points grouped).
        for eq_idx, derivs in enumerate(deriv_orders.values()):
            # For each boundary condition:
            for deriv_bc in derivs:
                deriv_row = []
                # For each component of boundary condition:
                for component, order, weight in zip(deriv_bc.comp, deriv_bc.orders, deriv_bc.weights, strict=True):
                    us_idx = eq_idx + component * N_bc_eqs
                    if order == (0, 0):
                        indices = torch.tensor([[zeroth_order_idx[us_idx]]])
                        row_val = torch.sparse_coo_tensor(indices=indices, values=torch.tensor([weight], dtype=torch.float32), device=self.device, size=(N_us_tot * N_comp,))
                        deriv_row.append(row_val)
                    else:
                        # Get the derivative matrix for this component and order.
                        spm = self.fd_spms[order][us_idx] * weight
                        deriv_row.append(spm)

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
        self.deriv_mat = self.deriv_mat.to(Us.device)
        bc_grads = torch.mv(self.deriv_mat, Us)
        return bc_grads

    def jacobian(self) -> torch.Tensor:
        return self.jac_mat