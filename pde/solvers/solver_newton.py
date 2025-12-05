from codetiming import Timer
import logging
import torch

from pde.config import FwdConfig

from pde.graph_grid.U_graph import UGraph, UValues
from pde.pdes.PDECalc import GraphPDECalc
from pde.solvers.linear_solvers import LinearSolver
from pde.utils_sparse import plot_sparsity
import math


class EfficientIntervalOptimizer:
    # Golden ratio constants for calculating test points
    # _INVPHI is 1/phi, where phi is the golden ratio ( (1+sqrt(5))/2 )
    _INVPHI = (math.sqrt(5) - 1) / 2
    # _INVPHI_COMPLEMENT is 1 - _INVPHI, which is also 1/phi^2
    _INVPHI_COMPLEMENT = 1 - _INVPHI

    def __init__(self, low=0.0, high=1., tol=1e-5, max_iter=10, device="cpu"):
        """
        Initializes the optimizer with a tolerance and maximum iterations.

        Args:
            tol: The tolerance for the width of the search interval.
            max_iter: The maximum number of iterations to perform.
        """
        # self.device = device
        self.low = low
        self.high = high

        self.tol = tol
        self.max_iter = max_iter

    def optimize(self, loss_func, f_zero, high=None):
        """
        Finds a value x in the interval [low, high] that attempts to minimize
        the given loss_func using the Golden Section Search algorithm.

        Returns:
            The value of x that approximately minimizes the loss_func.
        """
        low = self.low
        # Use previous high value if provided as starting guidline
        if high is None:
            high = self.high
        else:
            high = min(high*2+0.005, self.high)

        current_width = high - low
        if current_width <= self.tol:
            return (low + high) / 2

        # Calculate initial interior test points using golden ratio proportions
        x_upper = high - self._INVPHI_COMPLEMENT * current_width
        # Evaluate the loss function at the upper point
        f_upper = loss_func(x_upper)

        # Handle case if f_zero is lower than upper bound.
        if f_zero is not None:
            if f_zero < f_upper:
                # Immediately rule out upper region
                high = x_upper
                low = 0
                # Reinit test points
                current_width = high - low
                x_upper = high - self._INVPHI_COMPLEMENT * current_width
                f_upper = loss_func(x_upper)

        # Calculate the lower interior test point
        x_lower = low + self._INVPHI_COMPLEMENT * current_width
        f_lower = loss_func(x_lower)

        for i in range(self.max_iter):
            current_width = high - low  # Update current width
            if current_width <= self.tol:
                break

            if (f_lower < f_upper) or (f_upper > f_zero):  # Minimum is likely in the interval [low, x_upper]
                high = x_upper  # Narrow the interval from the right

                # The old x_lower becomes the new x_upper (closer to new 'high')
                x_upper = x_lower
                f_upper = f_lower  # Reuse its function evaluation (cached)

                # Calculate the new x_lower point
                x_lower = low + self._INVPHI_COMPLEMENT * (high - low)
                f_lower = loss_func(x_lower)  # Only one new function evaluation
            else:  # Minimum is likely in the interval [x_lower, high]
                low = x_lower  # Narrow the interval from the left

                # The old x_upper becomes the new x_lower (closer to new 'low')
                x_lower = x_upper
                f_lower = f_upper  # Reuse its function evaluation (cached)

                # Calculate the new x_upper point
                x_upper = high - self._INVPHI_COMPLEMENT * (high - low)
                f_upper = loss_func(x_upper)  # Only one new function evaluation

        pred_alpha = (low + high) / 2

        # Test alpha against low loss, and reduce further if needed.
        if f_zero is not None:
            f_alpha = loss_func(pred_alpha)
            if f_alpha > f_zero*1.01:
                logging.info(f"When doing interval optimisation, {pred_alpha = } is still too high. {f_alpha = }, {f_zero = }.")
                pred_alpha = pred_alpha / 2
                # print(loss_func(0.))
                # exit(5)
        return pred_alpha


class SolverNewton:
    """ Newton Raphson solver for finding roots of PDE residuals.
        pde_calc can be replaced for loading different PDEs.
    """
    def __init__(self, lin_solver: LinearSolver, cfg: FwdConfig):
        self.cfg = cfg

        self.lin_solver = lin_solver
        self.N_iter = cfg.N_iter
        self.solve_acc = cfg.solve_acc

        self.line_search_optim = EfficientIntervalOptimizer(max_iter=5)
        self.timer = Timer(name="timer", logger=None)

    def _test_residual(self, pde_calc, U_graph, U_values, alpha, deltas, aux_input):
        """ Run test with test Us and get residuals. U_values are not changed.
            Returns residual(Us + alpha * deltas).norm()
        """
        Us_test = U_graph.get_test_update(alpha * deltas, U_values)
        pde_resid = pde_calc.residuals(Us_test, aux_input)
        pde_resid_norm = pde_resid.norm()
        return pde_resid_norm

    @torch.no_grad()
    def newton_step(self, pde_calc: GraphPDECalc, jacobian, resid):
        """ Run a single Newton-Raphson step.
            Include pre and postprocessing for efficiency.
        """
        jacobian, resid = pde_calc.preproc_solve(jacobian, resid)
        deltas = self.lin_solver.solve(jacobian, resid)
        deltas = pde_calc.postproc_solve(deltas)
        return deltas

    @torch.no_grad()
    def find_pde_root(self, pde_calc: GraphPDECalc, U_graph: UGraph, U_values: UValues, aux_input=None):
        """
        Find the root of the PDE using Newton Raphson:
            grad(F(x_n)) * (x_{n+1} - x_n) = -F(x_n)

        aux_input: Additional conditioning for the PDE
        """
        converged, last_i = False, 0
        best_alpha = 1.0

        for i in range(self.N_iter):
            last_i = i

            # Compute Jacobian, residuals and solve linear system
            with self.timer:
                jacobian, old_resid = pde_calc.jacobian(U_values, aux_input)
            t_jacob = self.timer.last
            with self.timer:
                deltas = self.newton_step(pde_calc, jacobian, old_resid)
            t_solve = self.timer.last

            # Find best alpha using line search
            with self.timer:
                zero_alpha_norm = old_resid.norm()
                resid_fn = lambda alpha: self._test_residual(pde_calc, U_graph, U_values, alpha, deltas, aux_input)
                best_alpha = self.line_search_optim.optimize(resid_fn, high=best_alpha, f_zero=zero_alpha_norm)
                dUs = deltas * best_alpha
                U_graph.update_grid(dUs, U_values)

                # Evaluate residuals
                lin_error = jacobian @ deltas - old_resid
                lin_error_norm = lin_error.norm()

                # Error from PDE with updated Us
                new_resid = pde_calc.residuals(U_values, aux_input)
                new_resid_norm = new_resid.norm()
                max_abs_residual = torch.max(new_resid.abs())

            t_line = self.timer.last

            logging.info(f'NR Iteration {i}: Linear residual: {lin_error_norm:.3g}, Norm residual: {new_resid_norm:.3g}, Max residual: {max_abs_residual:.3g}')
            logging.debug(f'jacob time: {t_jacob:.4f}, solve time: {t_solve:.4g}, Line+postproc time: {t_line:.4f}s')


            if new_resid_norm < self.solve_acc:
                logging.debug(f"Newton solver converged early at iteration {i+1}.")
                converged = True
                # break
        else:
            logging.warning(f"Newton solver did not converge within the maximum iterations {i}. Linear residual: {lin_error_norm:.3g}, Norm residual: {new_resid_norm:.3g}, Max residual: {max_abs_residual:.3g}")

        return {"converged": converged, "iter": last_i, "residual_norm": new_resid_norm, "max_residual": max_abs_residual}
