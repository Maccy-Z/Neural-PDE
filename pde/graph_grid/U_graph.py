import torch
from torch import Tensor
from cprint import c_print
from codetiming import Timer

from pde.BaseU import UBase
from pde.graph_grid.graph_store import DerivGraph, Point, Deriv
from pde.graph_grid.graph_store import P_Types as T
from pde.findiff.findiff_coeff import gen_multi_idx_tuple, calc_coeff, nearest_neighbors
from pde.findiff.fin_deriv_calc import FinDerivCalcSPMV, NeumanBCCalc

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

        n_hop_idx[n] = ((M.crow_indices(), M.col_indices()))

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
    dirich_mask: Tensor  # [N_us_tot]                   # Mask for derivative BC nodes. Bool
    neumann_mask: Tensor  # [N_us_tot]                   # Mask for derivative BC nodes. Bool
    neumann_mode: bool    # True if there are derivative BCs.

    N_us_tot: int           # Total number of points
    N_us_grad: int          # Number of points that need fitting
    N_pdes: int             # Number of points to enforce PDEs (excl BC)
    N_component: int           # Number of vector components
    N_deriv: int         # Number of derivatives used
    N_dirich: int         # Number of Dirichlet BCs

    graphs: dict[tuple, DerivGraph] # [N_graphs]                  # Gradient graphs for each gradient type.
        # edge_index: torch.Tensor   # [2, num_edges]      # Edges between nodes
        # edge_coeff: torch.Tensor  # [num_edges]       # Finite diff coefficients for each edge
        # neighbors: list[Tensor]     # [N_us_tot, N_neigh]           # Neighborhood for each node

    deriv_calc: FinDerivCalcSPMV
    deriv_orders_bc: dict[int, Deriv]  # [N_deriv_BC, 2]     # Derivative order for each derivative BC

    # If Neumann:
    deriv_val: Tensor # [N_deriv_BC]            # Derivative values at nodes for BC
    deriv_calc_bc: NeumanBCCalc

    def _check(self, setup_dict):
        """ Check problem is well specified """
        points = list(setup_dict.values())
        types = [p.point_type for p in points]
        assert types.count(T.Ghost) == types.count(T.NeumCentralBC), "Number of ghost points must equal central Neumann BC points."


    def __init__(self, setup_dict: dict[int, Point], N_component, grad_neigh, max_degree:int = 2, tri=None, device="cpu"):
        """ Initialize the graph with a set of points.
            setup_dict: dict[node_id, Point]. Dictionary of each type of point
         """
        self.tri = tri
        self.device = device
        self.N_us_tot = len(setup_dict)
        self.N_us_grad = sum(T.GRAD in P.point_type for P in setup_dict.values())
        self.N_pdes = sum(T.PDE in P.point_type for P in setup_dict.values())
        self.N_component = N_component
        self._check(setup_dict)

        # 1) Node properties and masks
        dirich_mask = [T.DirichBC in P.point_type for P in setup_dict.values()]
        self.dirich_mask = torch.tensor(dirich_mask, dtype=torch.bool)
        self.N_dirich = self.dirich_mask.sum().item()
        # PDE is enforced on normal points.
        self.pde_mask = torch.tensor([T.PDE in P.point_type for P in setup_dict.values()])
        # U requires gradient for normal or ghost points.
        self.updt_mask = torch.tensor([T.GRAD in P.point_type  for P in setup_dict.values()])
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
        self.neumann_mode = len(deriv_val) > 0
        # 1.3) Set up initial node values
        self._Xs = torch.stack([point.X for point in setup_dict.values()]).to(torch.float32)
        self._Us = torch.tensor([point.value for point in setup_dict.values()], dtype=torch.float32)

        # 2.1) Get the neighborhood graph
        #n_hop_adj = tri_to_n_hop(self.tri)
        # nearest_neighbors(self.tri, self._Xs)
        stencils = nearest_neighbors(self.tri, self._Xs, grad_neigh)

        # 2.2) Compute finite difference stencils / graphs.
        # Each gradient type has its own stencil and graph.
        diff_degrees = gen_multi_idx_tuple(max_degree)[1:] # 0th order is just itself.
        self.graphs = {}
        for degree in diff_degrees:
            print_fn(f"Generating graph for degree {degree}")
            with Timer(text="Time to solve: : {:.4f}", logger=print_fn):
                edge_idx, fd_weights = calc_coeff(self._Xs, stencils, grad_neigh, degree)
                self.graphs[degree] = DerivGraph(edge_idx, fd_weights, shape=(self.N_us_tot, self.N_us_tot))

        # # 2.1) Add additional stencils
        # edge_mask = torch.ones(len(neum_mask))
        # laplacian = DerivGraph.add(DerivGraph.compose(self.graphs[(1, 0)], self.graphs[(1, 0)], mask=edge_mask),
        #                            DerivGraph.compose(self.graphs[(0, 1)], self.graphs[(0, 1)], mask=edge_mask)
        #                            )
        # self.graphs["laplacian"] = laplacian

        if device == "cuda":
            self._cuda()

        self.deriv_calc = FinDerivCalcSPMV(self.graphs, self.pde_mask, self.updt_mask, self.N_component, device=self.device)
        self.N_deriv = self.deriv_calc.N_deriv

        # 3) Derivative boundary conditions. Linear equations N X derivs - value = 0
        if self.neumann_mode:
            # 3.1) Compute jacobian permutation
            jacob_dict = {i: point for i, point in enumerate(v for v in setup_dict.values() if T.GRAD in v.point_type)}
            jacob_main_pos = {i: point for i, point in jacob_dict.items() if (T.NeumOffsetBC not in point.point_type and T.GRAD in point.point_type)}
            jacob_neum_pos = {i: point for i, point in jacob_dict.items() if T.NeumOffsetBC in point.point_type}
            # 3.2) Repeat for each component. Ordering:  [p0_0, p1_0, ..., p0_1, p1_1, ..., ..., b0_0, b0_1, ..., b1_0, b_1_1, ...]
            pde_perm, bc_perm = torch.tensor(list(jacob_main_pos.keys())), torch.tensor( list(jacob_neum_pos.keys()))
            pde_perm = torch.cat([pde_perm + i * self.N_us_grad for i in range(self.N_component)])
            bc_perm = torch.stack([bc_perm + i * self.N_us_grad for i in range(self.N_component)], dim=-1).flatten(0)
            self.row_perm = torch.cat([pde_perm, bc_perm])

            self.deriv_val = torch.tensor(deriv_val).flatten()
            self.deriv_orders_bc = deriv_orders
            self.neumann_mask = torch.tensor(neum_mask)

            self._cuda_bc()
            self.deriv_calc_bc = NeumanBCCalc(self.graphs, self.neumann_mask, self.updt_mask, self.deriv_orders_bc, N_component, device=self.device)

        # TODO: Testing
        mask = torch.ones_like(self.updt_mask)
        self.deriv_calc_eval = FinDerivCalcSPMV(self.graphs, mask, mask, self.N_component, device=self.device)

    def get_Us_dUs(self):
        _, Xs = self.get_us_Xs_pde()  # Shape = [N_total, 2].

        # 1) Finite differences D. shape = [N_pde, N_derivs, N_components]
        grads_dict = self.deriv_calc.derivative(self._Us)  # shape = [N_pde, N_comp]. Derivative removes boundary points.
        U_dUs = torch.stack(list(grads_dict.values()), dim=1)    # shape = [N_pde, N_derivs, N_component]
        return U_dUs, Xs

    def get_neum_preds(self):
        """ Get the derivative values for the boundary conditions. """
        return self.deriv_calc_bc.derivative(self._Us)      # shape = [N_bc_derivs, N_comp]


    def reset(self):
        self._Us = torch.zeros_like(self._Us)

    def _cuda(self):
        """ Move graph data to CUDA. """
        self._Us = self._Us.cuda(non_blocking=True)
        self._Xs = self._Xs.cuda(non_blocking=True)

        self.pde_mask = self.pde_mask.cuda(non_blocking=True)
        self.dirich_mask = self.dirich_mask.cuda(non_blocking=True)
        self.updt_mask = self.updt_mask.cuda(non_blocking=True)
        [graph.cuda() for graph in self.graphs.values()]

    def _cuda_bc(self):
        self.deriv_val = self.deriv_val.cuda(non_blocking=True)
        self.neumann_mask = self.neumann_mask.cuda(non_blocking=True)

