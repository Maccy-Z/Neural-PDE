import torch
import torch.func as func
from abc import ABC, abstractmethod

from pde.utils_sparse import CSRSummer, CSRRowMultiplier, CSRTransposer, CSRSystemSimplifier, plot_sparsity
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

        # Simplify solver system for linear solver
        dirich_mask = self.U_graph.dirich_mask.flatten()
        trivial_rows = torch.where(dirich_mask)[0]
        self.simplifier = CSRSystemSimplifier(dummy_jac, trivial_rows, trivial_rows)

    #@torch.no_grad()  # Gradient explicity handled.
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

        # 4) Reshape to cannonical ordering
        dRdD = torch.zeros(((self.U_graph.N_us_tot * self.N_component), (self.N_deriv + 1) * self.N_component), device=self.device)
        dRdD[self.pde_perm] = dRdD_main  # [N_Us_, N_deriv_]
        residuals = torch.zeros((self.U_graph.N_us_tot * self.N_component), device=self.device)
        residuals[self.pde_perm] = resid_main  # [N_Us_]

        # 5) Add on Neumann preds / dRdD:
        if self.neumann_mode:
            bc_deriv_pred = self.U_graph.get_neum_preds()  # shape = [N_bc_derivs_]
            bc_deriv_true = self.U_graph.deriv_val
            resid_bc = bc_deriv_pred - bc_deriv_true  # shape = [N_bc_derivs_]
            dRdD_bc = self.deriv_calc_bc.dRdD  # shape = [N_bc_derivs, N_deriv_]
            dRdD[self.bc_perm] = dRdD_bc
            residuals[self.bc_perm] = resid_bc

        # 6.1) Take product over j: dR_i/dD_jk * dD_jk/dU_j . shape = [N_deriv_][N_pde_, N_u_grad_]
        partials = []
        for d in range(self.N_component*(self.N_deriv+1)):
            prod = self.row_multipliers[d].mul(dDdU[d], dRdD[:, d])  # shape = [N_pde_, N_u_grad_]
            partials.append(prod)

        # 6.2) Sum over k: sum_k partials_ijk
        jacobian = self.csr_summer.sum(partials)


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


        residuals = torch.zeros((self.U_graph.N_us_tot * self.N_component), device=self.device) # shape = [N_Us_]
        residuals[self.pde_perm] = resid_main  # [N_Us_]

        # 2) Neumann BCs
        if self.neumann_mode:
            bc_deriv_pred = self.U_graph.get_neum_preds()
            bc_deriv_true = self.U_graph.deriv_val
            bc_residuals = bc_deriv_pred - bc_deriv_true
            residuals[self.bc_perm] = bc_residuals

        return residuals


    def preproc_solve(self, jacobian: torch.Tensor, b: torch.Tensor):
        """ Simplify the jacobian by removing trivial rows and columns. """
        return self.simplifier.simplify_system(jacobian, b)

    def postproc_solve(self, deltas: torch.Tensor):
        """ Recover the full solution from the deltas. """
        return self.simplifier.get_full_solution(deltas)