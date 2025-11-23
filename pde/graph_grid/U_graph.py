import torch
from torch import Tensor
from cprint import c_print
from codetiming import Timer

from pde.BaseU import UBase
from pde.graph_grid.graph_store import DerivGraph, Point, Deriv
from pde.graph_grid.graph_store import P_Types as T
from pde.findiff.findiff_coeff import gen_multi_idx_tuple, calc_coeff, nearest_neighbors
from pde.findiff.fin_deriv_calc import FinDerivCalcSPMV, BCCalc
from pde.graph_grid.graph_utils import plot_interp, plot_points
from pde.utils_sparse import CSRSummer, CSRRowMultiplier, CSRTransposer, CSRSystemSimplifier, plot_sparsity

print_fn = lambda s: c_print(f"{s}", color="bright_black")


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


class UGraph(UBase):
    """ Holder for graph structure. """
    _Xs: Tensor   # [N_us_tot, 2]                # Coordinates of nodes
    _Us: Tensor   # [N_us_tot, N_component]                   # Value at node
    deriv_val: Tensor # [N_deriv_BC*N_component]            # Derivative values at nodes for BC

    pde_mask: Tensor  # [N_us_tot]                   # Mask for where to enforce PDE on. Bool
    updt_mask: Tensor  # [N_us_tot]                   # Mask for nodes that need to be updated. Bool
    pde_idx: Tensor  # [N_pde * N_component]        # Indices for PDE nodes in flattened Us
    bc_idx: Tensor   # [N_bc * N_component]         # Indices for

    N_us_tot: int           # Total number of points
    N_us_grad: int          # Number of points that need fitting
    N_pdes: int             # Number of points to enforce PDEs (excl BC)
    N_comp: int           # Number of vector components
    N_deriv: int         # Number of derivatives used
    #N_dirich: int         # Number of Dirichlet BCs

    deriv_calc: FinDerivCalcSPMV
    bc_calc: BCCalc
    row_multipliers: list[CSRRowMultiplier]
    csr_summer: CSRSummer
    transposer: CSRTransposer
    simplifier: CSRSystemSimplifier

    def _check(self, setup_dict):
        """ Check problem is well specified """
        points = list(setup_dict.values())
        types = [p.point_type for p in points]
        assert types.count(T.Ghost) == types.count(T.NeumCentralBC), "Number of ghost points must equal central Neumann BC points."

        for p in points:
            if p.derivatives is not None:
                assert p.n_deriv == self.N_comp, "Number of BC components must match number of components."


    def __init__(self, setup_dict: dict[int, Point], N_component, grad_neigh, max_degree:int = 2, tri=None, device="cpu"):
        """ Initialize the graph with a set of points.
            setup_dict: dict[node_id, Point]. Dictionary of each type of point
         """
        self.tri = tri
        self.device = device
        self.N_us_tot = len(setup_dict)
        self.N_us_grad = sum(T.UPDATE in P.point_type for P in setup_dict.values())
        self.N_pdes = sum(T.PDE in P.point_type for P in setup_dict.values())
        self.N_comp = N_component
        self._check(setup_dict)

        self._Xs = torch.stack([point.X for point in setup_dict.values()]).to(torch.float32)

        # 1) Node properties and masks
        # PDE is enforced on normal points.
        self.pde_mask = torch.tensor([T.PDE in P.point_type for P in setup_dict.values()])
        self.bc_mask = torch.tensor([T.DERIV in P.point_type for P in setup_dict.values()])
        # U requires gradient for normal or ghost points.
        self.updt_mask = torch.tensor([T.UPDATE in P.point_type for P in setup_dict.values()])

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
        #self.neumann_mask = torch.tensor(neum_mask)
        # 1.3) Set up initial node values
        self._Xs = torch.stack([point.X for point in setup_dict.values()]).to(torch.float32)
        self._Us = torch.tensor([point.value for point in setup_dict.values()], dtype=torch.float32)

        # 2.1) Get the neighborhood graph
        stencils = nearest_neighbors(self.tri, self._Xs, grad_neigh)
        # 2.2) Compute finite difference stencils / graphs.
        diff_degrees = gen_multi_idx_tuple(max_degree)
        graphs = {}
        for degree in diff_degrees[1:]:  # 0th order is just itself.
            with Timer(text=f"Degree {degree}: Time to solve: : {{:.4f}}", logger=print_fn):
                edge_idx, fd_weights = calc_coeff(self._Xs, stencils, grad_neigh, degree)
                graphs[degree] = DerivGraph(edge_idx, fd_weights, shape=(self.N_us_tot, self.N_us_tot))
                            # edge_index: torch.Tensor   # [2, num_edges]      # Edges between nodes
                            # edge_coeff: torch.Tensor  # [num_edges]       # Finite diff coefficients for each edge
                            # neighbors: list[Tensor]     # [N_us_tot, N_neigh]           # Neighborhood for each node

        if device == "cuda":
            self._cuda()
            [graph.cuda() for graph in graphs.values()]

        self.deriv_calc = FinDerivCalcSPMV(graphs, N_comp=self.N_comp, device=self.device)
        self.N_deriv = len(diff_degrees) - 1        # Exclude zeroth order

        # 3) Derivative boundary conditions and Compute jacobian permutation
        # Single loop through all points in setup_dict
        jacob_main_pos, jacob_neum_pos, neum_dirich_bc = [], [], []
        for i, point in enumerate(setup_dict.values()):
            if T.UPDATE in point.point_type:
                # Categorize the point
                if T.NeumOffsetBC in point.point_type:
                    jacob_neum_pos.append(i)
                    neum_dirich_bc.append(point.is_dirichlet)
                else:
                    jacob_main_pos.append(i)
        neum_dirich_bc = torch.tensor(neum_dirich_bc, dtype=torch.bool)
        pde_idx, bc_idx = torch.tensor(jacob_main_pos), torch.tensor(jacob_neum_pos)
        dirich_mask = torch.zeros((self.N_us_tot, self.N_comp), dtype=torch.bool)
        dirich_mask[bc_idx] = neum_dirich_bc
        # 3.2) Repeat for each component. Ordering as Us.flatten()
        self.pde_idx = torch.stack([self.N_comp * pde_idx + i for i in range(self.N_comp)], dim=-1).flatten()       # shape = [N_pde * N_comp]
        self.bc_idx = torch.stack([self.N_comp * bc_idx + i for i in range(self.N_comp)], dim=-1).flatten()         # shape = [N_bc * N_comp]
        self.bc_calc = BCCalc(deriv_orders, self.N_comp, self.N_us_tot, diff_degrees, device=self.device)

        # 4) Helper for efficient sparse operations
        deriv_jac_pde = self.deriv_calc.jacobian()
        self.row_multipliers = [CSRRowMultiplier(spm, check_sparsity=True) for spm in deriv_jac_pde]
        self.csr_summer = CSRSummer(deriv_jac_pde, check_sparsity=True)
        dummy_jac = self.csr_summer.blank_csr()
        self.transposer = CSRTransposer(dummy_jac, check_sparsity=True)

        # Simplify trivial rows for linear solver

        trivial_rows = torch.where(dirich_mask.flatten())[0]
        self.simplifier = CSRSystemSimplifier(dummy_jac, trivial_rows, trivial_rows)

    def get_Us_dUs(self):
        Xs = self._Xs  # Shape = [N_total, 2].
        # Finite differences D. shape = [N_pde, N_derivs, N_components]
        grads_dict = self.deriv_calc.derivative(self._Us)  # shape = [N_pde, N_comp]. Derivative removes boundary points.
        U_dUs = torch.stack(list(grads_dict.values()), dim=1)    # shape = [N_pde, N_derivs, N_component]
        return U_dUs, Xs

    def reset(self):
        self._Us = torch.zeros_like(self._Us)

    def _cuda(self):
        """ Move graph data to CUDA. """
        self._Us = self._Us.cuda(non_blocking=True)
        self._Xs = self._Xs.cuda(non_blocking=True)

        self.pde_mask = self.pde_mask.cuda(non_blocking=True)
        self.updt_mask = self.updt_mask.cuda(non_blocking=True)

    def set_grid(self, new_Us):
        """
        Set grid to new values. Used for Jacobian computation.
        """
        self._Us = new_Us
