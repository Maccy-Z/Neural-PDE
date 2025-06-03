from codetiming import Timer
import logging
import torch

from pde.config import FwdConfig
from pde.BaseU import UBase
from pde.pdes.PDECalc import PDECalc
from pde.solvers.linear_solvers import LinearSolver
from pde.utils_sparse import plot_sparsity

class SolverNewton:
    def __init__(self,  U_graph: UBase, lin_solver: LinearSolver, pde_calc: PDECalc, cfg: FwdConfig):
        #self.pde_func = pde_func
        self.U_graph = U_graph
        self.lin_solver = lin_solver

        self.N_iter = cfg.N_iter
        self.lr = cfg.lr
        self.solve_acc = cfg.acc

        self.pde_calc = pde_calc
        self.device = U_graph.device


    def find_pde_root(self, aux_input=None):
        """
        Find the root of the PDE using Newton Raphson:
            grad(F(x_n)) * (x_{n+1} - x_n) = -F(x_n)

        :param aux_input: Additional conditioning for the PDE
        """
        timer = Timer(name="timer", logger=None)

        for i in range(self.N_iter):
            # Compute Jacobian and residuals
            with timer:
                jacobian, residuals = self.pde_calc.jacobian(aux_input)
            t_jacob = timer.last

            # Solve the linear system
            with timer:
                # Convert jacobian to sparse here instead of in lin_solver, so we can delete the dense Jacobian asap.
                jac_preproc, resid_preproc = self.lin_solver.preproc_tensor(jacobian, residuals)
                # del jacobian # torch.cuda.empty_cache()
                deltas, lin_resid_norm = self.lin_solver.solve(jac_preproc, resid_preproc)
            t_solve = timer.last

            # Evaluate solution
            with timer:
                lin_error = jacobian @ deltas - residuals
                lin_error_norm = lin_error.norm()
                deltas *= self.lr

                self.U_graph.update_grid(deltas)

                # Error from PDE with updated Us
                pde_resid = self.pde_calc.residuals(aux_input)
                pde_resid_norm = pde_resid.norm()
                max_abs_residual = torch.max(pde_resid.abs())

            t_post = timer.last
            # logging.debug("")
            logging.debug(f'Newton solver Iteration {i}')
            logging.debug(f'    Jacobian time: {t_jacob:.4f}s, Solve time: {t_solve:.4f}s, postproc time: {t_post:.4f}s')
            logging.debug(f'    Linear residual: {lin_error_norm:.3g}, Norm residual: {pde_resid_norm:.3g}, Max residual: {max_abs_residual:.3g}')


            if torch.mean(torch.abs(residuals)) < self.solve_acc:
                logging.info(f"Newton solver converged early at iteration {i+1}")
                break
