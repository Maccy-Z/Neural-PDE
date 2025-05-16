from codetiming import Timer
import time
import logging
import torch

from pde.config import FwdConfig
from pde.BaseU import UBase
from pde.cartesian_grid.PDE_Grad import PDEForward
from pde.solvers.jacobian import JacobCalc
from pde.solvers.linear_solvers import LinearSolver


class SolverNewton:
    def __init__(self,  sol_grid: UBase, lin_solver: LinearSolver, jac_calc: JacobCalc, cfg: FwdConfig):
        #self.pde_func = pde_func
        self.sol_grid = sol_grid
        self.lin_solver = lin_solver

        self.N_iter = cfg.N_iter
        self.lr = cfg.lr
        self.solve_acc = cfg.acc

        self.jac_calc = jac_calc
        self.device = sol_grid.device

        self.logging = {"time": 0.0, "residual": 0.}

    def find_pde_root(self, aux_input=None):
        """
        Find the root of the PDE using Newton Raphson:
            grad(F(x_n)) * (x_{n+1} - x_n) = -F(x_n)

        :param aux_input: Additional conditioning for the PDE
        """
        for i in range(self.N_iter):
            logging.debug("\n")
            with Timer(text="Time to calculate jacobian: : {:.4f}", logger=logging.debug):
                jacobian, residuals = self.jac_calc.jacobian(aux_input)

            with Timer(text="Time to solve: : {:.4f}", logger=logging.debug):
                # Convert jacobian to sparse here instead of in lin_solver, so we can delete the dense Jacobian asap.
                jac_preproc, residuals = self.lin_solver.preproc_tensor(jacobian, residuals)
                del jacobian # torch.cuda.empty_cache()
                deltas, lin_resid_norm = self.lin_solver.solve(jac_preproc, residuals)

            deltas *= self.lr

            # print(deltas)

            self.sol_grid.update_grid(deltas)

            # Error from PDE
            pde_resid = self.jac_calc.residuals(aux_input)
            pde_resid_norm = pde_resid.norm()
            max_abs_residual = torch.max(pde_resid.abs())

            logging.debug(f'Linear solver Iteration {i}')
            logging.debug(f'    Linear residual: {lin_resid_norm:.3g}')
            logging.debug(f'    Mean residual: {pde_resid_norm:.3g}, Max residual: {max_abs_residual:.3g}')


            # self.residuals = residuals
            # global I
            # I += 1
            # logging.debug(f'{I = }')
            # if I == 100:
            #     print(I)
            #     error = jacobian @ deltas - residuals
            #     max_err = torch.max(torch.abs(error)).item()
            #
            #     _mask = torch.zeros(self.sol_grid.grad_mask.sum()).cuda().bool()
            #     _pde_mask = self.sol_grid.pde_mask[self.sol_grid.grad_mask]
            #     _mask[_pde_mask] = 1
            #
            #     print(self.sol_grid.get_us_grad().squeeze())
            #
            #     save_dict = {"jacobian": jacobian, "residuals": residuals, "aux_input": aux_input, "mask": _mask}
            #     torch.save(save_dict, "jacobian_residuals.pt")
            #
            #     print(f'{max_err = }')
            #     exit("Newton Solver")

            self.logging["residual"] = pde_resid_norm

            if torch.mean(torch.abs(residuals)) < self.solve_acc:
                logging.info(f"Newton solver converged early at iteration {i+1}")
                break
