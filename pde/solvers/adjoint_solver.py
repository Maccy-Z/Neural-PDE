import torch
from codetiming import Timer
import logging
from cprint import c_print

from pde.pdes.PDECalc import GraphPDECalc
from pde.graph_grid.U_graph import UGraph, UValues
from pde.loss import Loss
from pde.solvers.linear_solvers import LinearSolver

class PDEAdjoint:
    def __init__(self, adj_lin_solver: LinearSolver, loss_fn: Loss):
        self.adj_lin_solver = adj_lin_solver
        self.loss_fn = loss_fn

    def adjoint_solve(self, pde_calc: GraphPDECalc, Us_new: UValues, Us_old: UValues, Us_true: UValues, jac=None):
        """ Solve for adjoint.
            dgdU = J^T * adjoint
            Args:
                pde_calc: GraphPDECalc object to compute Jacobian and residuals
                Us_new: UValues at which to compute loss gradient, after forward solve
                Us_old: UValues at which to compute Jacobian
                Us_true: UValues at true solution.
                jac: Cached Jacobian J(Us_old, theta), for efficiency. If None, it will be recomputed.
         """
        with torch.no_grad():
            jac_T = pde_calc.jacob_transpose(Us_old, jac)    # Shape = [N_eq, N_Us]

        # One adjoint value for each trained u value, including boundary points.
        loss = self.loss_fn(Us_new, Us_true)
        loss_u = self.loss_fn.gradient().flatten()

        adjoint = self.adj_lin_solver.solve(jac_T, loss_u)

        frac_err = (jac_T @ adjoint - loss_u).norm() / loss_u.norm()
        return adjoint, loss, frac_err

    def backpropagate(self, pde_calc: GraphPDECalc, U_values, adjoint):
        """
            Computes grads and populates model.parameters.grads
            adjoint.shape = [N_us] = [N_PDEs]
        """
        # Computes adjoint * dfdp as vector jacobian product.
        residuals = pde_calc.residuals(U_values)
        residuals.backward(-adjoint)
        return residuals

