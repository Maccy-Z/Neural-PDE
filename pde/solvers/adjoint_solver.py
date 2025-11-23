import torch
from codetiming import Timer
import logging
from cprint import c_print

from pde.pdes.PDECalc import GraphPDECalc
from pde.graph_grid.U_graph import UGraph
from pde.loss import Loss
from pde.solvers.linear_solvers import LinearSolver

class PDEAdjoint:
    def __init__(self, pde_calc: GraphPDECalc, adj_lin_solver: LinearSolver, loss_fn: Loss):
        self.pde_calc = pde_calc
        self.adj_lin_solver = adj_lin_solver
        self.loss_fn = loss_fn

    def adjoint_solve(self, U_graph: UGraph, jac=None, Us_loss=None):
        """ Solve for adjoint.
            dgdU = J^T * adjoint
            Us_loss: Optional. If different Us is needed to compute loss gradient than the jacobian J(Us, theta)
                    shape = [N_us_grad, N_comp]
         """
        with torch.no_grad():
            jac_T = self.pde_calc.jacob_transpose(jac)    # Shape = [N_eq, N_Us]

        # One adjoint value for each trained u value, including boundary points.
        if Us_loss is None:
            Us, updt_mask, _ = U_graph.get_us_mask()
            Us_grad = Us[updt_mask]
        else:
            Us_grad = Us_loss

        loss = self.loss_fn(Us_grad)
        loss_u = self.loss_fn.gradient().flatten()

        with Timer(text="Adjoint solve: {:.4f}s", logger=None) as timer:
            adjoint = self.adj_lin_solver.solve(jac_T, loss_u)
        t_adjoint = timer.last

        residual = (jac_T @ adjoint - loss_u).norm()
        logging.info(f'Adjoint lin solve. Residual: {residual:.3g}, T: {t_adjoint:.3g}s')
        return adjoint, loss, residual

    def backpropagate(self, adjoint):
        """
            Computes grads and populates model.parameters.grads
            adjoint.shape = [N_us] = [N_PDEs]
        """
        # Computes adjoint * dfdp as vector jacobian product.
        residuals = self.pde_calc.residuals()
        residuals.backward(-adjoint)
        return residuals

