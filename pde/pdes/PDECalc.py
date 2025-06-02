import torch
import torch.func as func
from abc import ABC, abstractmethod

from pde.utils_sparse import CSRSummer, CSRRowMultiplier, CSRTransposer, CSRConcatenator, CSRPermuter, plot_sparsity
from pde.graph_grid.U_graph import UGraph
from pde.pdes.PDEs import PDEFunc

class PDECalc(ABC):
    @abstractmethod
    def jacobian(self, pde_aux_input=None):
        """ Compute the Jacobian of the PDE residuals with respect to the u values. """
        pass

    @abstractmethod
    def jacob_transpose(self):
        """ Compute the transpose of the Jacobian of the PDE residuals with respect to the u values. """
        pass

    @abstractmethod
    def residuals(self, aux_input=None):
        """ Compute the residuals of the PDE. """
        pass



class GraphPDECalc(PDECalc):
    """ Computes PDE Jacobian and residuals for graph-based PDEs, including Neumann BCs."""

    def __init__(self, U_graph: UGraph, pde_func: PDEFunc):
        self.pde_func = pde_func
        self.device = U_graph.device

        self.U_graph = U_graph
        self.N_pdes = U_graph.N_pdes
        self.N_us_grad = U_graph.N_us_grad
        self.N_component = U_graph.N_comp
        self.N_deriv = U_graph.N_deriv

        self.deriv_calc = U_graph.deriv_calc
        self.resid_jac_val = torch.func.vmap(torch.func.jacrev(self.pde_func.residuals, has_aux=True, argnums=0))

        self.neumann_mode = U_graph.neumann_mode
        self.pde_perm = U_graph.pde_perm
        self.bc_perm = U_graph.bc_perm

        if U_graph.neumann_mode:
            self.deriv_calc_bc = U_graph.deriv_calc_bc

        # Precompute transforms with jacobian structure
        deriv_jac_list = self.U_graph.deriv_calc.jacobian()
        self.row_multipliers = [CSRRowMultiplier(spm, check_sparsity=True) for spm in deriv_jac_list]
        self.csr_summer = CSRSummer(deriv_jac_list, check_sparsity=True)

        dummy_jac = self.csr_summer.blank_csr()

        self.transposer = CSRTransposer(dummy_jac, check_sparsity=True)

    @torch.no_grad()  # Gradient explicity handled.
    def jacobian(self, pde_aux_input=None):
        """
            Compute jacobian dR/dU = dR/dD * dD/dU.

            us_grad.shape = [N_u_grad]. Gradients of trained u values.
            dR/dU.shape = [N_pde, N_u_grad]
            dR/dD.shape = [N_pde, N_derivs]
            dD/dU.shape = [N_pde, N_derivs, N_u_grad]

            dR_i/dU_j = sum_k dR_i/dD_jk * dD_jk/dU_j

            Vector derivatives are handled as batches of [N_comp, N_pde], then merged into a column-concatenated vector [N_comp*N_pde]. Components are grouped together.
        """
        # 1) Finite differences D.
        U_dUs, Xs = self.U_graph.get_Us_dUs()  # shape = [N_pde, N_derivs, N_components]

        # 2) dD/dU. shape = [N_derivs, N_u_grad]
        dDdU = self.deriv_calc.jacobian()  # shape = [N_derivs][N_Us_, N_Us_]

        # 3) dR/dD. shape = [N_pde*N_comp, N_derivs*N_comp] = [N_pde_, N_derivs_]
        dRdD_main, resid_main = self.resid_jac_val(U_dUs, Xs) if (pde_aux_input is None) else self.resid_jac_val(U_dUs, Xs, pde_aux_input)    # [N_pde, N_component, N_deriv, N_component]
                                                                                                                                        # residuals.shape = [N_pde, N_component]
        dRdD_main = dRdD_main.reshape(self.N_pdes * self.N_component, (self.N_deriv + 1) * self.N_component)  # [N_pde_, N_deriv_]
        resid_main = resid_main.reshape(self.N_pdes * self.N_component)  # [N_pde_]

        # 4) Get Neumann preds / dRdD:
        if self.neumann_mode:
            bc_deriv_pred = self.U_graph.get_neum_preds()  # shape = [N_bc_derivs_]
            bc_deriv_true = self.U_graph.deriv_val
            resid_bc = bc_deriv_pred - bc_deriv_true  # shape = [N_bc_derivs_]
            dRdD_bc = self.deriv_calc_bc.dRdD  # shape = [N_bc_derivs, N_deriv_]

        # 5) Reshape to cannonical ordering
        dRdD = torch.zeros(((self.U_graph.N_us_tot * self.N_component), (self.N_deriv + 1) * self.N_component), device=self.device)
        dRdD[self.pde_perm] = dRdD_main  # [N_Us_, N_deriv_]
        residuals = torch.zeros((self.U_graph.N_us_tot * self.N_component), device=self.device)
        residuals[self.pde_perm] = resid_main  # [N_Us_]
        if self.neumann_mode:
            dRdD[self.bc_perm] = dRdD_bc
            residuals[self.bc_perm] = resid_bc  # [N_Us_]

        # 4.1) Take product over j: dR_i/dD_jk * dD_jk/dU_j . shape = [N_deriv_][N_pde_, N_u_grad_]
        partials = []
        for d in range(self.N_component*(self.N_deriv+1)):
            prod = self.row_multipliers[d].mul(dDdU[d], dRdD[:, d])  # shape = [N_pde_, N_u_grad_]
            partials.append(prod)

        # 4.2) Sum over k: sum_k partials_ijk
        jacobian = self.csr_summer.sum(partials)

        # if self.U_graph.neumann_mode:
        #     # 5.1) Neumann boundary conditions: R = grad_n(u) - constant
        #     bc_deriv_pred = self.U_graph.get_neum_preds()  # shape = [N_bc_derivs, N_comp]
        #     bc_deriv_true = self.U_graph.deriv_val
        #     bc_residuals = bc_deriv_pred - bc_deriv_true
        #     # 5.2_ Neumann jacobian: dR/dD = 1, so select corresponding rows of jacobian.
        #     bc_deriv_jac = self.deriv_calc_bc.jac_mat  # shape = [N_bc_derivs_, N_u_grad_]
        #
        #     # 6) Concatenate on jacobian and residuals
        #     # residuals = [p0_0, p1_0, ..., p0_1, p1_1, ..., b0_0, b0_1, ..., b0_1, b_1_1, ...]
        #     residuals = torch.cat([residuals, bc_residuals])  # shape = [N_pde_+N_bc_]
        #     jacobian = self.concatenator.cat(jacobian, bc_deriv_jac)  # shape = [N_pde_+N_bc_, N_total_]
        #     # 6)  Neuman Jacobian is concatenated onto the end of the main Jacobian. Permute it back to correct order
        #     jacobian = self.permuter.matrix_permute(jacobian)
        #     residuals = self.permuter.vector_permute(residuals)

        return jacobian, residuals

    def jacob_transpose(self):
        jacobian, _ = self.jacobian()
        return self.transposer.transpose(jacobian)

    def residuals(self, aux_input=None):
        U_dUs, Xs = self.U_graph.get_Us_dUs()  # shape = [N_pde, N_derivs, N_components]

        # residuals.shape = [N_pde, N_component]
        if aux_input is None:
            resid_main, _ = func.vmap(self.pde_func.residuals)(U_dUs, Xs)
        else:
            resid_main, _ = func.vmap(self.pde_func.residuals)(U_dUs, Xs, aux_input)

        resid_main = resid_main.reshape(self.N_pdes * self.N_component) # shape = [N_pde * N_component]



        # 2) Neumann BCs
        if self.neumann_mode:
            bc_deriv_pred = self.U_graph.get_neum_preds()
            bc_deriv_true = self.U_graph.deriv_val
            bc_residuals = bc_deriv_pred - bc_deriv_true

            # residuals = torch.cat([residuals, bc_residuals])  # shape = [N_pde_+N_bc_]
            # residuals = self.permuter.vector_permute(residuals)

        residuals = torch.zeros((self.U_graph.N_us_tot * self.N_component), device=self.device) # shape = [N_Us_]
        residuals[self.pde_perm] = resid_main  # [N_Us_]

        if self.neumann_mode:
            residuals[self.bc_perm] = bc_residuals

        return residuals