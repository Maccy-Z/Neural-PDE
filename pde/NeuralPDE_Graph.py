import torch

from pde.graph_grid.U_graph import UGraph
from pde.pdes.PDECalc import GraphPDECalc
from pde.pdes.PDEs import PDEFunc
from pde.solvers.adjoint_solver import PDEAdjoint
from pde.solvers.linear_solvers import LinearSolver
from pde.solvers.solver_newton import SolverNewton
from pde.config import Config
from pde.loss import Loss
from pde.graph_grid.graph_utils import plot_interp, plot_points

class NeuralPDEGraph:
    u_graph: UGraph
    loss_fn: Loss
    adjoint: torch.Tensor

    def __init__(self, pde_fn: PDEFunc, U_graph: UGraph, cfg: Config, loss_fn:Loss = None, triangles=None):
        adj_cfg = cfg.adj_cfg
        fwd_cfg = cfg.fwd_cfg
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.DEVICE = cfg.DEVICE

        # pde_forward = PDEForward(U_graph, pde_fn)
        pde_calc = GraphPDECalc(U_graph, pde_fn)

        # Forward solver
        fwd_lin_solver = LinearSolver(fwd_cfg.lin_mode, cfg.DEVICE, cfg=fwd_cfg.lin_solve_cfg)
        newton_solver = SolverNewton(U_graph, fwd_lin_solver, pde_calc=pde_calc, cfg=fwd_cfg)

        # # Adjoint solver
        adj_lin_solver = LinearSolver(adj_cfg.lin_mode, self.DEVICE, adj_cfg.lin_solve_cfg)
        pde_adjoint = PDEAdjoint(U_graph, pde_fn, pde_calc, adj_lin_solver, loss_fn)

        self.pde_fn = pde_fn
        self.U_graph = U_graph
        self.newton_solver = newton_solver
        self.pde_adjoint = pde_adjoint

        self.triangles = triangles

    def forward_solve(self, aux_input=None):
        """ Solve PDE forward problem. """

        converged = self.newton_solver.find_pde_root(aux_input)
        return converged

    def adjoint_solve(self):
        """ Solve for adjoint """

        adjoint, loss = self.pde_adjoint.adjoint_solve()
        self.adjoint = adjoint
        return loss

    def backward(self):
        """ Once adjoint is calculated, backpropagate through PDE to get gradients.
            dL/dP = - adjoint * df/dP
         """
        residuals = self.pde_adjoint.backpropagate(self.adjoint)  # Shape = [N, ..., Nparams]

        # Delete adjoint to stop reuse.
        self.adjoint = None

        return residuals

    def plot_interp(self, Us=None, Xlims=None, title="Interpolated solution"):
        """ Plot the interpolated solution. """
        if Us is None:
            Us, Xs = self.U_graph.get_all_us_Xs()
        else:
            _, Xs = self.U_graph.get_all_us_Xs()

        plot_interp(Xs, Us.T, Xlims=Xlims, title=title, triangles=self.U_graph.tri)


    def plot_derivs(self, order):
        us_all, Xs = self.U_graph.get_all_us_Xs()

        deriv_dict = self.U_graph.deriv_calc_eval.derivative(us_all)
        derivs = deriv_dict[order]


        plot_interp(Xs, derivs.T, title=str(order), triangles=self.U_graph.tri)
        #
        divergence = deriv_dict[(1, 0)][:, 0] + deriv_dict[(0, 1)][:, 1]
        laplace_y = deriv_dict[(2, 0)][:, 1] + deriv_dict[(0, 2)][:, 1]
        laplace_x = deriv_dict[(0, 2)][:, 0] + deriv_dict[(2, 0)][:, 0]
        deriv_mats = self.u_graph.deriv_calc_eval.fd_spms
        #
        # x, y = Xs[:, 0], Xs[:, 1]
        # u_test = 0.14 - 0.25*(y - 0.75) ** 2
        # deriv_test = self.u_graph.deriv_calc_eval.derivative(u_test.unsqueeze(-1))
        # div_test = deriv_test[(1, 0)][:, 0]
        # exit(7)
        # plot_interp(Xs, divergence, title=str(order), triangles=self.u_graph.tri)
        pass


    def _plot_interp(self, value):
        us_all, Xs = self.U_graph.get_all_us_Xs()
        plot_interp(Xs, value, triangles=self.U_graph.tri)

    def plot_points(self, values, Xlims=None, show_index=False, title=""):
        Xlims = None # [(0,0.2), (0, 1.5)]
        _, Xs = self.U_graph.get_all_us_Xs()

        plot_points(Xs, values, Xlims=Xlims, show_index=show_index, title=title)

    def _plot_points(self, values, Xlims=None):
        _, Xs = self.U_graph.get_all_us_Xs()
        plot_points(Xs, values, Xlims=Xlims)


