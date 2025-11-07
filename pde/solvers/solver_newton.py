from codetiming import Timer
import logging
import torch

from pde.config import FwdConfig

from pde.graph_grid.U_graph import UGraph
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

    def optimize(self, loss_func, high=None, low_loss=None):
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
        # x_lower is the test point closer to 'low'
        # x_upper is the test point closer to 'high'
        x_lower = low + self._INVPHI_COMPLEMENT * current_width
        x_upper = high - self._INVPHI_COMPLEMENT * current_width

        # Evaluate the loss function at these two initial points
        f_lower = loss_func(x_lower)
        f_upper = loss_func(x_upper)

        for i in range(self.max_iter):
            current_width = high - low  # Update current width
            if current_width <= self.tol:
                break

            if f_lower < f_upper:  # Minimum is likely in the interval [low, x_upper]
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
        if low_loss is not None:
            if loss_func(pred_alpha) > low_loss:
                logging.debug(f"When doing interval optimisation, {pred_alpha = } is still too high. Reducing alpha.")
                pred_alpha = pred_alpha / 2

        return pred_alpha


class SolverNewton:
    """ Newton Raphson solver for finding roots of PDE residuals.
        pde_calc can be replaced for loading different PDEs.
    """
    def __init__(self, pde_calc: GraphPDECalc, lin_solver: LinearSolver, cfg: FwdConfig):
        # self.device = cfg.DEVICE
        self.cfg = cfg

        self.lin_solver = lin_solver

        self.N_iter = cfg.N_iter
        self.solve_acc = cfg.solve_acc

        self.pde_calc = pde_calc

        self.line_search_optim = EfficientIntervalOptimizer(max_iter=5)
        self.timer = Timer(name="timer", logger=None)

    def _test_residual(self, U_graph, alpha, deltas, Us_init, aux_input):
        """ Run test with test Us and get residuals. Then reset grid back to initial state. """
        Us = U_graph.get_test_update(alpha * deltas)
        U_graph.set_grid(Us)
        pde_resid = self.pde_calc.residuals(aux_input)
        pde_resid_norm = pde_resid.norm()

        # Reset grid to initial state
        U_graph.set_grid(Us_init)
        return pde_resid_norm

    @torch.no_grad()
    def newton_step(self, jacobian, old_resid, aux_input=None):
        """ Run a single Newton step. Done differentiably. """

        jac_proc, old_resid_proc = self.pde_calc.preproc_solve(jacobian, old_resid)
        deltas = self.lin_solver.solve(jac_proc, old_resid_proc)
        deltas = self.pde_calc.postproc_solve(deltas)
        return deltas

    @torch.no_grad()
    def find_pde_root(self, U_graph: UGraph, aux_input=None):
        """
        Find the root of the PDE using Newton Raphson:
            grad(F(x_n)) * (x_{n+1} - x_n) = -F(x_n)

        :param aux_input: Additional conditioning for the PDE
        """
        converged, last_i = False, 0
        best_alpha = 1.0

        for i in range(self.N_iter):
            Us_init = U_graph.get_all_us_Xs()[0]

            # Compute Jacobian, residuals and solve linear system
            with self.timer:
                jacobian, old_resid = self.pde_calc.jacobian(aux_input)
            t_jacob = self.timer.last
            with self.timer:
                deltas = self.newton_step(jacobian, old_resid, aux_input=aux_input)
            t_solve = self.timer.last

            # Find best alpha using line search
            with self.timer:
                zero_alpha_norm = old_resid.norm()
                resid_fn = lambda alpha: self._test_residual(U_graph, alpha, deltas, Us_init, aux_input)
                best_alpha = self.line_search_optim.optimize(resid_fn, high=best_alpha, low_loss=zero_alpha_norm)
                dUs = deltas * best_alpha #self.lr
                U_graph.update_grid(dUs)

                # Evaluate residuals
                lin_error = jacobian @ deltas - old_resid
                lin_error_norm = lin_error.norm()

                # Error from PDE with updated Us
                new_resid = self.pde_calc.residuals(aux_input)
                new_resid_norm = new_resid.norm()
                max_abs_residual = torch.max(new_resid.abs())
            t_line = self.timer.last

            logging.info(f'NR Iteration {i}: Linear residual: {lin_error_norm:.3g}, Norm residual: {new_resid_norm:.3g}, Max residual: {max_abs_residual:.3g}')
            logging.debug(f'jacob time: {t_jacob:.4f}, solve time: {t_solve:.4g}, Line+postproc time: {t_line:.4f}s')


            if new_resid_norm < self.solve_acc:
                logging.debug(f"Newton solver converged early at iteration {i+1}.")
                converged = True
                last_i = i
                return {"converged": converged, "iter": last_i}

        logging.warning(f"Newton solver did not converge within the maximum iterations {i}. Linear residual: {lin_error_norm:.3g}, Norm residual: {new_resid_norm:.3g}, Max residual: {max_abs_residual:.3g}")
        return {"converged": converged, "iter": last_i}
