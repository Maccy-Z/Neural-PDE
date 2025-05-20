import torch

from pde.cartesian_grid.U_grid import UGrid2D
from pde.solvers.jacobian import get_jac_calc
from pdes.PDEs import PDEFunc
from pde.cartesian_grid.PDE_Grad import PDEForward, PDEAdjoint
from solvers.linear_solvers import LinearSolver
from solvers.solver_newton import SolverNewton
from config import Config
from pde.loss import Loss


class NeuralPDE:
    us_grid: UGrid2D
    loss_fn: Loss
    adjoint: torch.Tensor

    def __init__(self, pde_fn: PDEFunc, grid_setup, loss_fn: Loss, cfg: Config):
        adj_cfg = cfg.adj_cfg
        fwd_cfg = cfg.fwd_cfg
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.DEVICE = cfg.DEVICE

        # Grid and BC
        Xs_grid = grid_setup.Xs_grid
        dirichlet_bc = grid_setup.dirichlet_bc
        neuman_bc = grid_setup.neuman_bc

        # PDE classes
        us_grid = UGrid2D(Xs_grid, dirichlet_bc=dirichlet_bc, neuman_bc=neuman_bc)
        pde_forward = PDEForward(us_grid, pde_fn)

        # Forward solver
        fwd_lin_solver = LinearSolver(fwd_cfg.lin_mode, cfg.DEVICE, cfg=fwd_cfg.lin_solve_cfg)
        fwd_jacob_calc = get_jac_calc(us_grid, pde_forward, fwd_cfg)
        newton_solver = SolverNewton(pde_forward, us_grid, fwd_lin_solver, jac_calc=fwd_jacob_calc, cfg=fwd_cfg)

        # Adjoint solver
        adj_lin_solver = LinearSolver(adj_cfg.lin_mode, self.DEVICE, adj_cfg.lin_solve_cfg)
        adj_jacob_calc = get_jac_calc(us_grid, pde_forward, adj_cfg)
        pde_adjoint = PDEAdjoint(us_grid, pde_fn, adj_jacob_calc, adj_lin_solver, loss_fn)

        self.pde_fn = pde_fn
        self.us_grid = us_grid
        self.newton_solver = newton_solver
        self.pde_adjoint = pde_adjoint

    def forward_solve(self):
        """ Solve PDE forward problem. """

        self.newton_solver.find_pde_root()
        us, _ = self.us_grid.get_real_us_Xs()
        # show_grid(us, "Fitted values")

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

    def get_us_Xs(self):
        us, Xs = self.us_grid.get_real_us_Xs()
        return us, Xs





