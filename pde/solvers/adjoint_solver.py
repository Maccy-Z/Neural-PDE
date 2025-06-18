import torch
import torch.func as func
from codetiming import Timer
import logging

from pde.pdes.PDECalc import PDECalc
from pde.graph_grid.U_graph import UGraph
from pde.pdes.PDEs import PDEFunc
from pde.loss import Loss


class PDEAdjoint:
    def __init__(self, U_graph: UGraph, pde_calc: PDECalc, adj_lin_solver, loss_fn: Loss):
        self.U_graph = U_graph
        self.pde_calc = pde_calc
        self.adj_lin_solver = adj_lin_solver
        self.loss_fn = loss_fn

        self.DEVICE = U_graph.device

    def adjoint_solve(self, Us_now=None):
        """ Solve for adjoint.
            dgdU = J^T * adjoint
            Us_now: Optional. If different Us is needed to compute loss gradient than the jacobian J(Us, theta)
                    shape = [N_us_grad, N_comp]
         """

        with torch.no_grad():
            jac_T = self.pde_calc.jacob_transpose()    # Shape = [N_eq, N_us]

        # One adjoint value for each trained u value, including boundary points.
        if Us_now is None:
            Us, updt_mask, _ = self.U_graph.get_us_mask()
            Us_grad = Us[updt_mask].flatten()

        else:
            Us_grad = Us_now.flatten()

        loss = self.loss_fn(Us_grad)
        loss_u = self.loss_fn.gradient()

        with Timer(text="Adjoint solve: {:.4f}s", logger=logging.debug):
            # Free memory of dense jacobian before solving adjoint equation.
            # jac_T_proc, loss_u = self.adj_lin_solver.preproc_tensor(jac_T, loss_u)
            # del jac_T
            adjoint = self.adj_lin_solver.solve(jac_T, loss_u)

            residual = (jac_T @ adjoint - loss_u).norm()
        logging.debug(f'Adjoint residual: {residual:.3g}')

        return adjoint, loss

    def backpropagate(self, adjoint):
        """
            Computes grads and populates model.parameters.grads
            adjoint.shape = [N_us] = [N_PDEs]
        """
        # Computes adjoint * dfdp as vector jacobian product.
        residuals = self.pde_calc.residuals()
        residuals.backward(-adjoint)
        return residuals

