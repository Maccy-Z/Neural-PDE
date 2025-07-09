import torch
from codetiming import Timer

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

        self.pde_calc = GraphPDECalc(U_graph, pde_fn)

        # Forward solver
        fwd_lin_solver = LinearSolver(fwd_cfg.lin_mode, cfg.DEVICE, cfg=fwd_cfg.lin_solve_cfg)
        self.newton_solver = SolverNewton(self.pde_calc, fwd_lin_solver, cfg=fwd_cfg)

        # Adjoint solver
        adj_lin_solver = LinearSolver(adj_cfg.lin_mode, self.DEVICE, adj_cfg.lin_solve_cfg)
        self.pde_adjoint = PDEAdjoint(self.pde_calc, adj_lin_solver, loss_fn)

        self.pde_fn = pde_fn
        self.U_graph = U_graph

        self.triangles = triangles

    def forward_solve(self, aux_input=None):
        """ Solve PDE forward problem. """

        converged = self.newton_solver.find_pde_root(self.U_graph, aux_input)
        return converged

    def adjoint_solve(self):
        """ Solve for adjoint """

        adjoint, loss = self.pde_adjoint.adjoint_solve(self.U_graph)
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


    def single_step(self):
        """ Perform a single step of the Newton solver and compute the exact derivative:
                U_old - U_new = dU = J^-1(u_old, theta) f(U_old, theta)
                J^T lambda = dL/dtheta|(U_new)
                dL/dtheta = lambda.T @ (dj/dtheta @ dU - df/dtheta)
         """
        Us_old = self.U_graph.get_all_us_Xs()[0]
        # init_loss = self.loss_fn(Us_old)


        # with Timer( text="Newton step and adjoint: {:.4f}s"):
        # Compute at u_old
        deltas, J, old_resid = self.newton_solver.newton_step()
        deltas = deltas.detach()
        # Compute loss derivative at u_new, Jacobian a u_old
        Us_new = self.U_graph.get_test_update(deltas)
        adjoint, _ = self.pde_adjoint.adjoint_solve(self.U_graph, Us_loss=Us_new)


        # dL/dtheta = lambda.T @ (dj/dtheta @ dU - df/dtheta)
        # with Timer(text="Backward: {:.4f}s"):
        adj_f = adjoint @ (J @ deltas - old_resid)
        adj_f.backward()
        # J_delta = J @ deltas - old_resid
        # J_delta.backward(adjoint)

        with torch.no_grad():
            init_loss = self.loss_fn(Us_old, requires_grad=False)
            final_loss = self.loss_fn(Us_new, requires_grad=False)

        # print(f'{adj_f = }, {init_loss = }, {final_loss = }')
        return init_loss, final_loss, Us_new


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


