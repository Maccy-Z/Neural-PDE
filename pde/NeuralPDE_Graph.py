import torch

from pde.graph_grid.U_graph import UGraph
from pde.solvers.jacobian import get_jac_calc
from pde.pdes.PDEs import PDEFunc
from pde.graph_grid.PDE_Grad import PDEForward, PDEAdjoint
from pde.solvers.linear_solvers import LinearSolver
from pde.solvers.solver_newton import SolverNewton
from pde.config import Config
from pde.loss import Loss
from pde.graph_grid.graph_utils import plot_interp_graph

class NeuralPDEGraph:
    us_graph: UGraph
    loss_fn: Loss
    adjoint: torch.Tensor

    def __init__(self, pde_fn: PDEFunc, us_graph: UGraph, cfg: Config, loss_fn:Loss = None):
        adj_cfg = cfg.adj_cfg
        fwd_cfg = cfg.fwd_cfg
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.DEVICE = cfg.DEVICE

        pde_forward = PDEForward(us_graph, pde_fn)

        # Forward solver
        fwd_lin_solver = LinearSolver(fwd_cfg.lin_mode, cfg.DEVICE, cfg=fwd_cfg.lin_solve_cfg)
        fwd_jacob_calc = get_jac_calc(us_graph, pde_forward, fwd_cfg)
        newton_solver = SolverNewton(us_graph, fwd_lin_solver, jac_calc=fwd_jacob_calc, cfg=fwd_cfg)

        # Adjoint solver
        # adj_lin_solver = LinearSolver(adj_cfg.lin_mode, self.DEVICE, adj_cfg.lin_solve_cfg)
        # adj_jacob_calc = get_jac_calc(us_graph, pde_forward, adj_cfg)
        # pde_adjoint = PDEAdjoint(us_graph, pde_fn, adj_jacob_calc, adj_lin_solver, loss_fn)

        self.pde_fn = pde_fn
        self.us_graph = us_graph
        self.newton_solver = newton_solver
        # self.pde_adjoint = pde_adjoint

    def forward_solve(self, aux_input=None):
        """ Solve PDE forward problem. """
        self.newton_solver.find_pde_root(aux_input)

    def adjoint_solve(self):
        """ Solve for adjoint """

        adjoint, loss = self.pde_adjoint.adjoint_solve()
        self.adjoint = adjoint

        # c_print(f'loss: {loss.item():.3g}', "bright_magenta")
        return loss

    def backward(self):
        """ Once adjoint is calculated, backpropagate through PDE to get gradients.
            dL/dP = - adjoint * df/dP
         """
        residuals = self.pde_adjoint.backpropagate(self.adjoint)  # Shape = [N, ..., Nparams]

        # Delete adjoint to stop reuse.
        self.adjoint = None

        return residuals

    def plot_interp(self, Xs=None, us=None):
        """ Plot the interpolated solution. """
        if Xs is None:
            Xs = self.us_graph.Xs
        if us is None:
            us = self.us_graph.us

        plot_interp_graph(Xs, us[:, 0])
