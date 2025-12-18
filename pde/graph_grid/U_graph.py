import torch
from torch import Tensor
from cprint import c_print
import logging
from codetiming import Timer

from pde.graph_grid.graph_store import DerivGraph, Point, Deriv
from pde.graph_grid.graph_store import P_Types as T
from pde.findiff.findiff_coeff import gen_multi_idx_tuple, calc_coeff, nearest_neighbors
from pde.findiff.fin_deriv_calc import FinDerivCalcSPMV, BCCalc
from pde.graph_grid.graph_utils import plot_interp, plot_points
from pde.utils_sparse import CSRSummer, CSRRowMultiplier, CSRTransposer, CSRSystemSimplifier
from pde.solvers.nvmath_cudss import CUDSSSolver
from pde.solvers.linear_solvers import LinMode
from pde.config import Config


def tri_to_n_hop(tris, hops=6):
    """
    Build an (undirected) n-hop adjacency matrix from a [n_tri,3] triangle index tensor.

    Returns vertices reachable within n hops.
    """
    n_verts = int(tris.max().item()) + 1

    # same edge stacking
    e0 = tris[:, [0, 1]]
    e1 = tris[:, [1, 2]]
    e2 = tris[:, [2, 0]]
    edges = torch.cat([e0, e1, e2], dim=0)

    # mirror for undirected
    rev_edges = edges[:, [1, 0]]
    all_edges = torch.cat([edges, rev_edges], dim=0).t()  # shape [2, 6*n_tri]

    # values are all 1
    vals = torch.ones(all_edges.shape[1], dtype=torch.float32, device=tris.device)

    adj_mat = torch.sparse_coo_tensor(all_edges, vals, (n_verts, n_verts)).to_sparse_csr()

    M = adj_mat
    n_hop_idx = {}
    for n in range(2, hops + 1):
        M = M @ adj_mat
        n_hop_idx[n] = (M.crow_indices(), M.col_indices())

    # Get n-hop neighbours
    n_hop_reach_cols = {}
    for hop in range(3, hops):
        crow_idxs, col_idxs = n_hop_idx[hop]
        reached_cols = {}
        for i in range(n_verts):
            start = crow_idxs[i]
            stop = crow_idxs[i + 1]
            reached_cols[i] = col_idxs[start:stop].to(torch.int32).numpy()

        n_hop_reach_cols[hop] = reached_cols

    return n_hop_reach_cols

class UValues:
    """ Holds numerical values at each node. """
    Xs: Tensor   # [N_us_tot, 2]                # Coordinates of nodes
    Us: Tensor   # [N_us_tot, N_component]                   # Value at
    def __init__(self, Xs: Tensor, Us: Tensor):
        self.Xs = Xs
        self.Us = Us

    @torch.no_grad()
    def cuda(self):
        self.Us = self.Us.cuda(non_blocking=True)
        self.Xs = self.Xs.cuda(non_blocking=True)

    def clone(self):
        return UValues(self.Xs.clone(), self.Us.clone())


class UGraph:
    """ Holder for graph structure. """
    device: torch.device | str

    U_values: UValues                           # Current U values on the grid
    _Xs: Tensor  # [N_us_tot, 2]                # Coordinates of nodes
    tri: Tensor | None                          # Triangle mesh for plotting

    pde_mask: Tensor  # [N_us_tot]                   # Mask for where to enforce PDE on. Bool
    dirich_mask: Tensor  # [N_us_tot * N_component]   # Mask for Dirichlet BCs
    pde_idx: Tensor  # [N_pde * N_component]        # Indices for PDE nodes in flattened Us
    bc_idx: Tensor   # [N_bc * N_component]         # Indices for

    N_Us_tot: int           # Total number of points
    N_us_grad: int          # Number of points that need fitting
    N_pdes: int             # Number of points to enforce PDEs (excl BC)
    N_comp: int           # Number of vector components
    N_deriv: int         # Number of derivatives used

    deriv_calc: FinDerivCalcSPMV
    bc_calc: BCCalc
    row_multipliers: list[CSRRowMultiplier]
    csr_summer: CSRSummer
    transposer: CSRTransposer
    simplifier: CSRSystemSimplifier
    solver_opts: dict[str, CUDSSSolver | None]

    def _check(self, setup_dict):
        """ Check problem is well specified """
        points = list(setup_dict.values())
        types = [p.point_type for p in points]
        assert types.count(T.Ghost) == types.count(T.NeumCentralBC), "Number of ghost points must equal central Neumann BC points."

        for p in points:
            if p.derivatives is not None:
                assert p.n_deriv == self.N_comp, "Number of BC components must match number of components."

    def __init__(self, setup_dict: dict[int, Point], N_comp, grad_neigh, tri, max_degree:int = 2,  device="cpu"):
        """ Initialize the graph with a set of points.
        Args:
            setup_dict: dict[node_id, Point]. Dictionary of each type of point
            N_comp: Number of components in U
            grad_neigh: Number of neighbors to use for gradient calculations
            max_degree: Maximum derivative degree to compute
            tri: Triangle mesh for the domain, for generating stencils
            device: Device to use
         """
        self.tri = tri
        self.device = device
        self.N_Us_tot = len(setup_dict)
        self.N_us_grad = sum(T.UPDATE in P.point_type for P in setup_dict.values())
        self.N_pdes = sum(T.PDE in P.point_type for P in setup_dict.values())
        self.N_comp = N_comp
        self._check(setup_dict)

        # 1) Node properties and masks
        # PDE is enforced on normal points.
        self.pde_mask = torch.tensor([T.PDE in P.point_type for P in setup_dict.values()])
        self.bc_mask = torch.tensor([T.DERIV in P.point_type for P in setup_dict.values()])

        # 1.2) Derivative BC properties
        deriv_orders, deriv_val, neum_mask = {}, [], []
        for point_num, point in setup_dict.items():
            if T.DERIV in point.point_type:
                derivs = point.derivatives
                deriv_orders[point_num] = derivs
                deriv_val.append([d.value for d in derivs])
                neum_mask.append(True)
            else:
                neum_mask.append(False)

        # 1.3) Set up node positions
        Xs = torch.stack([point.X for point in setup_dict.values()]).to(torch.float32)
        self._Xs = Xs

        # Compute finite difference stencils. Make sure to respect connectivity of graph.
        # 2.1) Get the neighborhood graph
        stencils = nearest_neighbors(self.tri, self._Xs, grad_neigh)
        # 2.2) Compute finite difference stencils / graphs.
        diff_degrees = gen_multi_idx_tuple(max_degree)
        graphs = {}
        for degree in diff_degrees[1:]:  # 0th order is just itself.
            with Timer(text=f"Degree {degree}: Time to solve: : {{:.4f}}", logger=logging.debug):
                edge_idx, fd_weights = calc_coeff(self._Xs, stencils, grad_neigh, degree)
                graphs[degree] = DerivGraph(edge_idx, fd_weights, shape=(self.N_Us_tot, self.N_Us_tot), device=self.device)
                            # edge_index: torch.Tensor   # [2, num_edges]      # Edges between nodes
                            # edge_coeff: torch.Tensor  # [num_edges]       # Finite diff coefficients for each edge
                            # neighbors: list[Tensor]     # [N_us_tot, N_neigh]           # Neighborhood for each node

        self.deriv_calc = FinDerivCalcSPMV(graphs, N_comp=self.N_comp, device=self.device)
        self.N_deriv = len(diff_degrees) - 1        # Exclude zeroth order

        # 3) Derivative boundary conditions and Compute jacobian permutation
        # Single loop through all points in setup_dict
        jacob_main_pos, jacob_neum_pos, dirich_bc = [], [], []
        for i, point in enumerate(setup_dict.values()):
            if T.UPDATE in point.point_type:
                # Categorize the point
                if T.NeumOffsetBC in point.point_type:
                    jacob_neum_pos.append(i)
                    dirich_bc.append(point.is_dirichlet)
                else:
                    jacob_main_pos.append(i)
        pde_idx, bc_idx = torch.tensor(jacob_main_pos), torch.tensor(jacob_neum_pos)
        dirich_bc = torch.tensor(dirich_bc, dtype=torch.bool)
        self.dirich_mask = torch.zeros((self.N_Us_tot, self.N_comp), dtype=torch.bool)
        self.dirich_mask[bc_idx] = dirich_bc
        # 3.1) Repeat for each component. Ordering as Us.flatten()
        self.pde_idx = torch.stack([self.N_comp * pde_idx + i for i in range(self.N_comp)], dim=-1).flatten()       # shape = [N_pde * N_comp]
        self.bc_idx = torch.stack([self.N_comp * bc_idx + i for i in range(self.N_comp)], dim=-1).flatten()         # shape = [N_bc * N_comp]
        self.bc_calc = BCCalc(deriv_orders, dirich_bc, self.N_comp, self.N_Us_tot, diff_degrees, device=self.device)

        # 4) Helper for efficient sparse operations
        deriv_jac_pde = self.deriv_calc.jacobian()
        self.row_multipliers = [CSRRowMultiplier(spm, check_sparsity=True) for spm in deriv_jac_pde]
        self.csr_summer = CSRSummer(deriv_jac_pde, check_sparsity=True)
        dummy_jac = self.csr_summer.blank_csr()
        self.transposer = CSRTransposer(dummy_jac, check_sparsity=True)
        # Simplify trivial rows for linear solver
        trivial_rows = torch.where(self.dirich_mask.flatten())[0]
        self.simplifier = CSRSystemSimplifier(dummy_jac, trivial_rows, trivial_rows)

    def init_lin_solver(self, cfg: Config):
        """ Initialise CUDSS solver individual to graph """
        self.solver_opts = {"fwd": None, "adj": None}
        if cfg.fwd_cfg.lin_mode == LinMode.CUDSS:
            self.solver_opts['fwd'] = CUDSSSolver(cfg.fwd_cfg.solver_cfg)
        if cfg.adj_cfg.lin_mode == LinMode.CUDSS:
            self.solver_opts['adj'] = CUDSSSolver(cfg.adj_cfg.solver_cfg)
        #
        # print(f'{self.solver_opts = }')
        # exit(5)

    def get_Us_dUs(self, U_values: UValues):
        """ Get U values and their derivatives at each point. """
        Us, Xs = U_values.Us, U_values.Xs

        # Finite differences D. shape = [N_pde, N_derivs, N_components]
        grads_dict = self.deriv_calc.derivative(Us)  # shape = [N_pde, N_comp]. Derivative removes boundary points.
        U_dUs = torch.stack(list(grads_dict.values()), dim=1)    # shape = [N_pde, N_derivs, N_component]
        return U_dUs, Xs

    def set_grid(self, new_Us: torch.Tensor, U_values: UValues):
        """
        Set grid to new values. Used for Jacobian computation.
        Enforce dirichlet boundary condition to BC values
        """
        U_values.Us = new_Us
        U_values.Us[self.dirich_mask] = self.bc_calc.dirich_bc_vals() # Enforce Dirichlet BCs

    def update_grid(self,deltas, U_values: UValues):
        """
        Update grid with changes, and fix boundary conditions with new grid.
        deltas.shape = [N*N_comp]
        us -> us - deltas
        """
        deltas = deltas.view(-1, self.N_comp)
        self.set_grid(U_values.Us - deltas, U_values)

    def new_Us(self, Us: torch.Tensor) -> UValues:
        """ Create new UValues object from Us tensor, respecting boundary conditions. """
        Us_values = UValues(self._Xs, Us.clone())
        self.set_grid(Us, Us_values)
        return Us_values

    def get_test_update(self, deltas, Us_old: UValues) -> UValues:
        """
        Get test update for grid with changes, without applying them.
        deltas.shape = [N*N_comp]
        us -> us - deltas
        """
        deltas = deltas.view(-1, self.N_comp)
        Us_test = torch.clone(Us_old.Us) - deltas
        Us_test[self.dirich_mask] = self.bc_calc.dirich_bc_vals()
        return UValues(Us_old.Xs, Us_test)

    def get_zero_U_values(self, Us_old) -> UValues:
        """ Get UValues object with all zeros (except BCs). """
        zeros = torch.zeros_like(Us_old.Us)
        Us_zeros = UValues(Us_old.Xs, zeros)
        self.set_grid(zeros, Us_zeros)
        return Us_zeros

    def get_all_us_Xs(self, U_values: UValues):
        """ Return all grid points, including fake boundaries. """
        return U_values.Us, self._Xs

    # --------------- Plotting functions ---------------
    def plot_interp(self, Us_values: UValues, Xlims=None, title="Interpolated solution"):
        """ Plot the interpolated solution. """
        Us, Xs = Us_values.Us, Us_values.Xs
        plot_interp(Xs, Us.T, Xlims=Xlims, title=title, triangles=self.tri)

    def plot_derivs(self, Us_values: UValues, order):
        us_all, Xs = Us_values.Us, Us_values.Xs

        deriv_dict = self.deriv_calc.derivative(us_all)
        derivs = deriv_dict[order]
        plot_interp(Xs, derivs.T, title=str(order), triangles=self.tri)

    def plot_points(self, Us_values: UValues, Xlims=None, show_index=False, title=""):
        Us, Xs = Us_values.Us, Us_values.Xs
        plot_points(Xs, Us.T, Xlims=Xlims, show_index=show_index, title=title)

def setup_graph(setup_dict: dict[int, Point], tri, N_comp, grad_neigh, max_degree:int = 2, device="cpu") -> tuple[UGraph, UValues]:
    """ Create UGraph and UValues. """
    U_graph = UGraph(setup_dict, N_comp, grad_neigh, max_degree=max_degree, tri=tri, device=device)

    Xs = torch.stack([point.X for point in setup_dict.values()]).to(torch.float32).to(device)
    Us = torch.tensor([point.value for point in setup_dict.values()], dtype=torch.float32, device=device)

    Us_values = UValues(Xs, Us)
    return U_graph, Us_values
