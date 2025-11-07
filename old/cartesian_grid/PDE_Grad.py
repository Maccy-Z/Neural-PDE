import torch
from codetiming import Timer
import logging
from cprint import c_print
from pde.cartesian_grid.discrete_derivative import DerivativeCalc
from pde.cartesian_grid.U_grid import UGrid, USubGrid
from pde.cartesian_grid.X_grid import XGrid
from pde.pdes.PDEs import PDEFunc
from pde.loss import Loss

class PDEForward:
    def __init__(self, u_grid: UGrid, pde_func: PDEFunc):
        self.pde_func = pde_func
        self.u_grid = u_grid
        self.deriv_calc = u_grid.deriv_calc

    def residuals(self, us_grad, subgrid: USubGrid):
        """
            Returns residuals of equations that require gradients only.
        """
        us_bc = subgrid.add_nograd_to_us(us_grad)  # Shape = [N+2, N+2]. Need all Us to calculate derivatives.
        Xs = subgrid.Xs_pde  # Shape = [N+2, N+2, 2]. Only need Xs for residuals.


        deriv_dict = self.deriv_calc.derivative(us_bc)  # shape = [N, ...]. Derivative removes boundary points.
        us = us_bc[1:-1, 1:-1]
        u_dus = [us] + list(deriv_dict.values())

        residuals = self.pde_func(u_dus, Xs)

        resid_grad = residuals[subgrid.pde_mask]

        return resid_grad, resid_grad


class PDEAdjoint:
    def __init__(self, u_grid: UGrid, pde_func: PDEFunc, adj_jacob_calc, adj_lin_solver, loss_fn: Loss):
        self.pde_func = pde_func
        self.us_grid = u_grid
        self.deriv_calc = u_grid.deriv_calc
        self.adj_jacob_calc = adj_jacob_calc
        self.adj_lin_solver = adj_lin_solver
        self.loss_fn = loss_fn

        self.DEVICE = u_grid.device

    def adjoint_solve(self):
        """ Solve for adjoint.
            dgdU = J^T * adjoint
         """
        with torch.no_grad():
            jacobian, _ = self.adj_jacob_calc.jacobian()    # Shape = [N_eq, N_us]
            jac_T = jacobian.T
            del jacobian

        # One adjoint value for each trained u value, including boundary points.
        us, grad_mask, _ = self.us_grid.get_us_mask()
        us_grad = us[grad_mask]

        loss = self.loss_fn(us_grad)
        loss_u = self.loss_fn.gradient()
        with Timer(text="Adjoint solve: {:.4f}", logger=logging.debug):
            # Free memory of dense jacobian before solving adjoint equation.
            jac_T_proc = self.adj_lin_solver.preproc_tensor(jac_T)
            del jac_T
            adjoint = self.adj_lin_solver.solve(jac_T_proc, loss_u)

        return adjoint, loss

    def backpropagate(self, adjoint):
        """
            Computes grads and populates model.parameters.grads
            adjoint.shape = [N_us] = [N_PDEs]
        """

        # 1) Calculate u_x, u_xx
        us_bc, _ = self.us_grid.get_all_us_Xs()  # Shape = [N+2, N+2]
        us, Xs = self.us_grid.get_real_us_Xs()  # Shape = [N, N], [N, N, 2]
        dudx, d2udx2 = self.deriv_calc.derivative(us_bc)  # Shape = [N]
        # Mask for boundary points
        pde_mask = self.us_grid.pde_mask[1:-1, 1:-1]    # Shape = [N, N]

        # 2) Reshape Us for vmap
        us = us[pde_mask].unsqueeze(-1)
        dudx = dudx[pde_mask]
        d2udx2 = d2udx2[pde_mask]
        dUs_flat = [us, dudx, d2udx2]
        Xs_flat = Xs[pde_mask]

        # Computes adjoint * dfdp as vector jacobian product.
        residuals = self.pde_func(dUs_flat, Xs_flat)
        residuals.backward(- adjoint)
        return residuals


class Loss_fn:
    def __init__(self, X_grid: XGrid, true_us: torch.Tensor):
        """
        :param xs: Positions where loss is taken at
        :param true_us: Value of us at xs
        Note: Grid is assumed to be fixed

        Returns: dL(u)/du for adjoint equation
        """

        self.true_us = true_us  # Shape = [N-2]
        self.xs = X_grid

    def loss_fn(self, us):
        loss = torch.sum((us - self.true_us) ** 2)  # Shape = [N-2]
        return loss

    def general_loss(self, us):
        us.requires_grad_(True)
        loss = self.loss_fn(us)
        loss.backward()
        return us.grad

    def quad_loss(self, us: torch.Tensor):
        # Special case for quadratic loss
        # loss = torch.sum((us - self.true_us)** 2)
        dL_du = 2 * (us - self.true_us)

        return dL_du


def main():
    pass


if __name__ == "__main__":
    main()
