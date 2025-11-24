import torch
from codetiming import Timer
import logging

from pde.graph_grid.U_graph import UGraph
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

        self.pde_calc = GraphPDECalc(self.U_graph, pde_fn)

        # Forward solver
        fwd_lin_solver = LinearSolver(fwd_cfg.lin_mode, cfg.device, cfg=fwd_cfg)
        self.newton_solver = SolverNewton(fwd_lin_solver, cfg=fwd_cfg)

        # Adjoint solver
        adj_lin_solver = LinearSolver(adj_cfg.lin_mode, self.device, cfg=adj_cfg)
        self.pde_adjoint = PDEAdjoint(adj_lin_solver, loss_fn)

        self.timer = Timer(name="timer", logger=None)

    def forward_solve(self, aux_input=None):
        """ Solve PDE forward problem. """
        converged = self.newton_solver.find_pde_root(self.pde_calc, self.U_graph, aux_input)
        return converged

    def adjoint_solve(self):
        """ Solve for adjoint. Call self.backward to get gradients, using adjoints. """
        adjoint, loss = self.pde_adjoint.adjoint_solve(self.pde_calc, self.U_graph)
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

    def single_step(self):
        """ Perform a single step of the Newton solver and compute the exact derivative:
                U_old - U_new = dU = J^-1(u_old, theta) f(U_old, theta)
                J^T lambda = dL/dtheta|(U_new)
                dL/dtheta = lambda.T @ (dj/dtheta @ dU - df/dtheta)
         """
        # Compute forward step form Us_old
        J, old_resid = self.pde_calc.jacobian()
        with self.timer:
            deltas = self.newton_solver.newton_step(self.pde_calc, J, old_resid)
        fwd_resid = (J @ deltas - old_resid).norm()
        logging.info(f'One Newton step residual norm: {fwd_resid:.3g}, T = {self.timer.last:.3g}s')

        # Compute loss derivative at Us_new, Jacobian at Us_old
        Us_new = self.U_graph.get_test_update(deltas)
        adjoint, _, adj_resid = self.pde_adjoint.adjoint_solve(self.pde_calc, self.U_graph, jac=J, Us_loss=Us_new)

        with self.timer:
            # dL/dtheta = lambda.T @ (dj/dtheta @ dU - df/dtheta)
            adj_f = adjoint @ (J @ deltas - old_resid)
            adj_f.backward()
        t_backward = self.timer.last

        with torch.no_grad():
            Us_old = self.U_graph.get_all_us_Xs()[0]
            init_loss = self.loss_fn(Us_old, requires_grad=False)
            final_loss = self.loss_fn(Us_new, requires_grad=False)

        # print(f'{adj_f = }, {init_loss = }, {final_loss = }')
        logging.debug(f"Backprop time: {t_backward:.4f}s")

        if adj_resid > 0.01 or fwd_resid > 0.01:
            logging.warning(f'High residuals in single step: Forward resid: {fwd_resid:.3g}, Adjoint resid: {adj_resid:.3g}')
        return init_loss, final_loss, old_resid

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


