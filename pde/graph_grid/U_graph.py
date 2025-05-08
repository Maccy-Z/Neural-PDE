import torch
from torch import Tensor
from cprint import c_print
from codetiming import Timer

from pde.BaseU import UBase
from pde.graph_grid.graph_store import DerivGraph, Point, Deriv
from pde.graph_grid.graph_store import P_Types as T
from pde.findiff.findiff_coeff import gen_multi_idx_tuple, calc_coeff
from pde.findiff.fin_deriv_calc import FinDerivCalcSPMV, NeumanBCCalc


class UGraph(UBase):
    """ Holder for graph structure. """
    _Xs: Tensor   # [N_us_tot, 2]                # Coordinates of nodes
    _us: Tensor   # [N_us_tot, N_component]                   # Value at node
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
    deriv_calc_bc: FinDerivCalcSPMV

    def _check(self, setup_dict):
        """ Check problem is well specified """
        points = list(setup_dict.values())
        types = [p.point_type for p in points]
        assert types.count(T.Ghost) == types.count(T.NeumCentralBC), "Number of ghost points must equal central Neumann BC points."


    def __init__(self, setup_dict: dict[int, Point], N_component, grad_acc:int = 2, max_degree:int = 2, tri=None, device="cpu"):
        """ Initialize the graph with a set of points.
            setup_dict: dict[node_id, Point]. Dictionary of each type of point
         """
        self.tri = tri
        self.device = device
        self.N_us_tot = len(setup_dict)
        self.N_us_grad = len([P for P in setup_dict.values() if T.GRAD in P.point_type])
        self.N_pdes = len([P for P in setup_dict.values() if T.PDE in P.point_type])
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
        self._us = torch.tensor([point.value for point in setup_dict.values()], dtype=torch.float32)

        # 2) Compute finite difference stencils / graphs.
        # Each gradient type has its own stencil and graph.
        diff_degrees = gen_multi_idx_tuple(max_degree)[1:] # 0th order is just itself.
        self.graphs = {}
        for degree in diff_degrees:
            c_print(f"Generating graph for degree {degree}", color="black")
            with Timer(text="Time to solve: : {:.4f}"):
                edge_idx, fd_weights = calc_coeff(setup_dict, grad_acc, degree)
                self.graphs[degree] = DerivGraph(edge_idx, fd_weights, shape=(self.N_us_tot, self.N_us_tot))

        # 2.1) Add additional stencils
        edge_mask = torch.ones(len(neum_mask))
        laplacian = DerivGraph.add(DerivGraph.compose(self.graphs[(1, 0)], self.graphs[(1, 0)], mask=edge_mask),
                                   DerivGraph.compose(self.graphs[(0, 1)], self.graphs[(0, 1)], mask=edge_mask)
                                   )
        self.graphs["laplacian"] = laplacian

        if device == "cuda":
            self._cuda()

        self.deriv_calc = FinDerivCalcSPMV(self.graphs, self.pde_mask, self.updt_mask, self.N_component, device=self.device)
        self.N_deriv = self.deriv_calc.N_deriv

        # 3) Derivative boundary conditions. Linear equations N X derivs - value = 0
        if self.neumann_mode:
            self.deriv_val = torch.tensor(deriv_val).flatten()
            self.deriv_orders_bc = deriv_orders
            self.neumann_mask = torch.tensor(neum_mask)

            self._cuda_bc()
            self.deriv_calc_bc = NeumanBCCalc(self.graphs, self.neumann_mask, self.updt_mask, self.deriv_orders_bc, N_component, device=self.device)


    def reset(self):
        self._us = torch.zeros_like(self._us)

    # def set_bc(self, dirich_bc=None, neuman_bc=None):
    #     """ Set boundary conditions. """
    #     if dirich_bc is not None:
    #         dirich_bc = dirich_bc.to(self.device, non_blocking=True)
    #         assert dirich_bc.sum() == self.N_dirich, "Dirichlet BC must match number of Dirichlet points."
    #         assert dirich_bc.sum() == self.N_dirich, "Dirichlet BC must match number of Dirichlet points."
    #         self._us[self.dirich_mask] = dirich_bc
    #
    #     if neuman_bc is not None:
    #         neuman_bc = neuman_bc.to(self.device, non_blocking=True)
    #         self.deriv_val = neuman_bc
    #         self.neumann_mask = torch.tensor([True for _ in range(len(neuman_bc))], device=self.device)

    # def set_eval_deriv_calc(self, mask):
    #     assert self.deriv_calc_eval is None, "Already set eval deriv calc"
    #     self.deriv_calc_eval = FinDerivCalcSPMV(self.graphs, mask, mask, self.N_component, device=self.device)


    # def get_grads(self):
    #     grad_dict = self.deriv_calc.derivative(self._us)
    #     return grad_dict
    #
    # def get_neuman_grads(self):
    #     grad_dict = self.deriv_calc_bc.derivative(self._us)
    #     return grad_dict

    # def get_grads_all(self):
    #     grad_dict = self.deriv_calc.derivative(self._us)
    #     bc_grad_dict = self.deriv_calc_bc.derivative(self._us)
    #
    #     full_grad_dict = {}
    #     for name, deriv in grad_dict.items():
    #         a = torch
    #         full_grad_dict[name] = deriv
    #
    #     return full_grad_dict

    def _cuda(self):
        """ Move graph data to CUDA. """
        self._us = self._us.cuda(non_blocking=True)
        self._Xs = self._Xs.cuda(non_blocking=True)

        self.pde_mask = self.pde_mask.cuda(non_blocking=True)
        self.dirich_mask = self.dirich_mask.cuda(non_blocking=True)
        self.updt_mask = self.updt_mask.cuda(non_blocking=True)
        [graph.cuda() for graph in self.graphs.values()]

    def _cuda_bc(self):
        self.deriv_val = self.deriv_val.cuda(non_blocking=True)
        self.neumann_mask = self.neumann_mask.cuda(non_blocking=True)

