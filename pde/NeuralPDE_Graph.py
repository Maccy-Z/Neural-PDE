import torch
from codetiming import Timer
import logging

from pde.graph_grid.U_graph import UGraph, UValues
from pde.pdes.PDECalc import GraphPDECalc
from pde.pdes.PDEs import PDEFunc
from pde.solvers.adjoint_solver import PDEAdjoint
from pde.solvers.linear_solvers import LinearSolver
from pde.solvers.solver_newton import SolverNewton
from pde.config import Config
from pde.loss import Loss
from pde.graph_grid.graph_utils import plot_interp, plot_points
from pde.utils import ARTEFACT_DIR

class NeuralPDEGraph:
    u_graph: UGraph
    loss_fn: Loss
    adjoint: torch.Tensor

    def __init__(self, pde_fn: PDEFunc, U_graph: UGraph, cfg: Config, loss_fn:Loss = None):
        self.cfg = cfg
        self.device = cfg.device
        adj_cfg = cfg.adj_cfg
        fwd_cfg = cfg.fwd_cfg

        self.loss_fn = loss_fn
        self.U_graph = U_graph
        self.pde_fn = pde_fn

        # self.pde_calc = GraphPDECalc(self.U_graph, pde_fn, device=cfg.device)

        # Forward solver
        fwd_lin_solver = LinearSolver(fwd_cfg.lin_mode, cfg.device, cfg=fwd_cfg)
        self.newton_solver = SolverNewton(fwd_lin_solver, cfg=fwd_cfg)

        # Adjoint solver
        adj_lin_solver = LinearSolver(adj_cfg.lin_mode, self.device, cfg=adj_cfg)
        self.pde_adjoint = PDEAdjoint(adj_lin_solver, loss_fn)

        self.timer = Timer(name="timer", logger=None)

    def forward_solve(self, U_graph: UGraph, Us: UValues, aux_input=None):
        """ Solve PDE forward problem. """
        pde_calc = GraphPDECalc(U_graph, self.pde_fn, device=self.device)

        converged = self.newton_solver.find_pde_root(pde_calc, U_graph, Us, aux_input)
        return converged

    def adjoint_solve(self, Us: UValues):
        """ Solve for adjoint. Call self.backward to get gradients, using adjoints. """
        adjoint, loss = self.pde_adjoint.adjoint_solve(self.pde_calc, Us, Us)
        self.adjoint = adjoint
        return loss

    def backward(self):
        """ Once adjoint is calculated, backpropagate through PDE to get gradients.
            dL/dP = - adjoint * df/dP
         """
        residuals = self.pde_adjoint.backpropagate(self.pde_calc, self.adjoint)  # Shape = [N, ..., Nparams]

        # Delete adjoint to stop reuse.
        self.adjoint = None

        return residuals

    def single_step(self, U_graph: UGraph, Us_current: UValues):
        """ Perform a single step of the Newton solver and compute the exact derivative:
                U_old - U_new = dU = J^-1(u_old, theta) f(U_old, theta)
                J^T lambda = dL/dtheta|(U_new)
                dL/dtheta = lambda.T @ (dj/dtheta @ dU - df/dtheta)
         """
        pde_calc = GraphPDECalc(U_graph, self.pde_fn, device=self.device)

        # Compute forward step form Us_old
        J, old_resid = pde_calc.jacobian(Us_current)
        deltas = self.newton_solver.newton_step(pde_calc, J, old_resid)
        fwd_err = (J @ deltas - old_resid).norm() / old_resid.norm()

        # Compute loss derivative at Us_new, Jacobian at Us_old
        Us_new: UValues = U_graph.get_test_update(deltas, Us_old=Us_current)
        adjoint, _, adj_err = self.pde_adjoint.adjoint_solve(pde_calc, Us_new, Us_current, jac=J)

        with self.timer:
            # dL/dtheta = lambda.T @ (dj/dtheta @ dU - df/dtheta)
            adj_f = adjoint @ (J @ deltas - old_resid)
            adj_f.backward()
        t_backward = self.timer.last

        with torch.no_grad():
            Us_old = U_graph.get_all_us_Xs(Us_current)[0]
            init_loss = self.loss_fn(Us_old, requires_grad=False)
            final_loss = self.loss_fn(Us_new.Us, requires_grad=False)

        logging.debug(f"Backprop time: {t_backward:.4f}s")

        if adj_err > 0.005 or fwd_err > 0.005:
            logging.warning(f'High residuals in single step: Forward resid: {fwd_err:.3g}, Adjoint resid: {adj_err:.3g}')
        return init_loss, final_loss, old_resid

