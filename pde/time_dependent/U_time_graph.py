import torch
from torch import Tensor
from cprint import c_print
from codetiming import Timer

from pde.BaseU import UBase
from pde.graph_grid.graph_store import DerivGraph, Point, Deriv, T_Point
from pde.graph_grid.graph_store import P_Types as T, P_TimeTypes as TT
from pde.findiff.findiff_coeff import gen_multi_idx_tuple, calc_coeff
from pde.findiff.fin_deriv_calc import FinDerivCalcSPMV, NeumanBCCalc
from pde.graph_grid.graph_utils import plot_points, plot_interp_graph
from pde.utils_sparse import coo_col_select


class UTemp(UBase):
    _Xs: Tensor   # [N_us_tot, 2]                # Coordinates of nodes
    _us: Tensor   # [N_us_tot, N_component]                   # Value at node
    deriv_calc: FinDerivCalcSPMV

    def __init__(self, Xs, us, deriv_calc, pde_mask, grad_mask):
        self._Xs = Xs
        self._us = us
        self.deriv_calc = deriv_calc
        self.pde_mask = pde_mask
        self.grad_mask = grad_mask

    def reset(self):
        self._us = torch.zeros_like(self._us)

    def get_grads(self):
        grad_dict = self.deriv_calc.derivative(self._us)

        return grad_dict
    def _cuda(self):
        pass

    def set_grid(self, new_us):
        """
        Set grid to new values. Used for Jacobian computation.
        """
        self._us[self.grad_mask] = new_us.flatten()

class BC_enforce:
    def __init__(self, deriv_val, deriv_mat, grad_mask, neumann_mask):
        self.deriv_val = deriv_val.T.flatten()
        self.neumann_mask_T = neumann_mask.T

        # 3.1: Setup neuman BC updates
        A = deriv_mat.to_sparse_coo()
        # A is stacked over components, [x1, x2, x3, ..., y1, y2, y3, ...]. Repeat mask
        self.bc_mask = grad_mask.repeat(2)
        # Solve A_bc u_bc + A_main u_main = deriv_val
        A_bc = coo_col_select(A, self.bc_mask)
        self.A_main = coo_col_select(A, ~self.bc_mask)
        self.A_bc_inv = torch.inverse(A_bc.to_dense())


    def set_bc_(self, us_):
        """ Solve A_bc u_bc + A_main u_main = deriv_val"""
        u_main = us_.T.flatten()[~self.bc_mask]
        b = self.deriv_val - self.A_main @ u_main
        u_bc = self.A_bc_inv @ b

        # Don't let BC blow up
        top_vals, top_idxs = torch.topk(u_bc, 2)
        u_bc = torch.clamp(u_bc, max=u_bc[top_idxs[1]] * 1.05)
        u_bc = torch.clamp(u_bc, max=us_.max(), min=us_.min())
        # print(u_bc.max())

        us_.T[self.neumann_mask_T] = u_bc


class UGraphTime(UBase):
    """ Holder for graph structure. """
    setup_dict: dict[int, T_Point]  # [N_us_tot]                  # Dictionary of points

    _Xs: Tensor   # [N_us_tot, 2]                # Coordinates of nodes
    _us: Tensor   # [N_us_tot, N_component]                   # Value at node
    deriv_val: Tensor # [N_deriv_BC*N_component]            # Derivative values at nodes for BC

    pde_mask: Tensor  # [N_us_tot]                   # Where dU/dt is enforced. Bool
    updt_mask: Tensor  # [N_us_tot, N_component]                   # Nodes that need to be updated. Bool
    neumann_mask: Tensor  # [N_us_tot, N_component]                   # Mask for derivative BC nodes. Bool
    neumann_mode: bool    # True if there are derivative BCs.
    dirich_mask: Tensor  # [N_us_tot, N_component]                   # Mask for Dirichlet BC nodes. Bool

    N_us_tot: int           # Total number of points
    N_us_grad: int          # Number of points that need fitting
    N_pdes: int             # Number of points to enforce PDEs (excl BC)
    N_component: int           # Number of vector components
    N_deriv: int         # Number of derivatives used
    N_update: int         # Number of us that need updating

    graphs: dict[tuple, DerivGraph] # [N_graphs]                  # Gradient graphs for each gradient type.
        # edge_index: torch.Tensor   # [2, num_edges]      # Edges between nodes
        # edge_coeff: torch.Tensor  # [num_edges]       # Finite diff coefficients for each edge
        # neighbors: list[Tensor]     # [N_us_tot, N_neigh]           # Neighborhood for each node

    deriv_calc: FinDerivCalcSPMV
    deriv_orders_bc: dict[int, list[Deriv]]  # [N_deriv_BC, 2]     # Derivative order for each derivative BC
    deriv_calc_bc: NeumanBCCalc
    row_perm: Tensor # [N_us_grad] # Permutation for Jacobian matrix

    def _check(self, setup_dict):
        """ Check problem is well specified """
        points = list(setup_dict.values())
        types = [p.point_type for p in points]
        assert types.count(T.Ghost) == types.count(T.NeumCentralBC), "Number of ghost points must equal central Neumann BC points."

        # for idx, t_p in setup_dict.items():
        #     P_type = t_p.point_type
        #     assert P_type[0] == P_type[1]

    def __init__(self, setup_dict: dict[int, T_Point], N_component, grad_acc:int = 2, max_degree:int = 2, device="cpu"):
        """ Initialize the graph with a set of points.
            setup_dict: dict[node_id, Point]. Dictionary of each type of point
         """
        self.device = device
        self.N_us_tot = len(setup_dict)
        self.N_us_grad = len([P for P in setup_dict.values() if TT.NORMAL in P.point_type])
        self.N_pdes = self.N_us_grad #len([P for P in setup_dict.values() if TT.NORMAL in P.point_type])
        self.N_component = N_component

        self._check(setup_dict)
        # 1) Reorder points. Redefines node values.
        sorted_points = sorted(setup_dict.values(), key=lambda x: x.X[1])
        self.setup_dict = {i: point for i, point in enumerate(sorted_points)}
        self._Xs = torch.stack([point.X for point in self.setup_dict.values()]).to(torch.float32)
        self._us = torch.tensor([point.init_val for point in self.setup_dict.values()], dtype=torch.float32)
        # 1.1) Masks for points that need updating.
        grad_mask = [[TT.NORMAL in P_type for P_type in P_comps.point_type]
                          for P_comps in self.setup_dict.values()]
        self.updt_mask = torch.tensor(grad_mask, dtype=torch.bool)
        self.N_update = self.updt_mask.sum().item()
        neumann_mask = [[TT.EXIT in P_type for P_type in P_comps.point_type]
                            for P_comps in self.setup_dict.values()]
        self.neumann_mask = torch.tensor(neumann_mask, dtype=torch.bool)
        self.neumann_mode = self.neumann_mask.sum().item() > 0
        dirich_mask = [[TT.FIXED in P_type for P_type in P_comps.point_type]
                             for P_comps in self.setup_dict.values()]
        self.dirich_mask = torch.tensor(dirich_mask, dtype=torch.bool)

        # 1.2) Masks for points updated using du/dt ODE
        grad_setup_dict = {}
        for idx, t_p in self.setup_dict.items():
            P_type = t_p.point_type
            if P_type[0] == TT.NORMAL:
                grad_setup_dict[idx] = Point(T.GRAD, t_p.X, value=None)
            else:
                grad_setup_dict[idx] = Point(T.DirichBC, t_p.X, value=None)
        pde_mask = [(T.GRAD in P.point_type) for P in grad_setup_dict.values()]
        self.pde_mask = torch.tensor(pde_mask, dtype=torch.bool)

        # 2) Compute finite difference stencils / graphs for points that require fitting
        self.graphs = self._gen_graph(grad_setup_dict, grad_acc, max_degree)

        if device == "cuda":
            self._cuda()

        self.deriv_calc = FinDerivCalcSPMV(self.graphs, self.pde_mask, self.pde_mask, self.N_component, device=self.device)
        self.N_deriv = self.deriv_calc.N_deriv

        # 3) Derivative boundary conditions. Linear equations N X derivs - value = 0
        if self.neumann_mode:
            deriv_orders, deriv_val = {}, []
            for point_num, point in setup_dict.items():
                if TT.EXIT in point.point_type:
                    derivs = point.derivatives
                    deriv_orders[point_num] = derivs
                    deriv_val.append([d.value for d in derivs])
            self.deriv_orders_bc = deriv_orders
            self.deriv_val = torch.tensor(deriv_val, dtype=torch.float32)
            self.neumann_mask = torch.tensor(neumann_mask, dtype=torch.bool)

            self._cuda_bc()
            self.deriv_calc_bc = NeumanBCCalc(self.graphs, self.neumann_mask[:, 0], self.neumann_mask[:, 0],
                                              self.deriv_orders_bc, N_component, device=self.device)
            # 3.1: Setup neuman BC updates
            self.bc_enforce = BC_enforce(self.deriv_val, self.deriv_calc_bc.deriv_mat, self.deriv_calc_bc.grad_mask, self.neumann_mask)


    def _cuda_bc(self):
        self.deriv_val = self.deriv_val.cuda(non_blocking=True)
        self.neumann_mask = self.neumann_mask.cuda(non_blocking=True)

    def _gen_graph(self, setup_dict, grad_acc, max_degree):
        """ Generate graph for each gradient type. Each gradient type has its own stencil and graph."""
        diff_degrees = gen_multi_idx_tuple(max_degree)[1:]  # 0th order is just itself.
        graphs = {}
        for degree in diff_degrees:
            c_print(f"Generating graph for degree {degree}", color="black")
            with Timer(text="Time to solve: : {:.4f}"):
                edge_idx, fd_weights = calc_coeff(setup_dict, grad_acc, degree)
                graphs[degree] = DerivGraph(edge_idx, fd_weights, shape=(self.N_us_tot, self.N_us_tot))
        return graphs

    def get_subgraph(self, components=None, all_grads=False):
        """ Make copy of subgraph, except deriv_calc returns all gradients instead of just pde. """
        # if components is None:
        #     components = [i for i in range(self.N_component)]
        #
        # if all_grads:
        #     mask = torch.ones_like(self.pde_mask).bool()
        # else:
        #     mask = self.updt_mask

        # deriv_calc = FinDerivCalcSPMV(self.graphs, mask, mask, self.N_component, device=self.device)
        # us_clone = self._us.clone()
        # subraph_copy = UTemp(self._Xs.clone(), us_clone, deriv_calc, self.pde_mask, self.updt_mask)
        # return subraph_copy
        mask = torch.ones_like(self.pde_mask).bool()
        new_instance = self.__class__.__new__(self.__class__)

        new_instance.device = self.device
        new_instance._Xs = self._Xs.clone()
        new_instance._us = self._us.clone()
        new_instance.deriv_val = self.deriv_val.clone()
        new_instance.pde_mask = self.pde_mask.clone()

        new_instance.updt_mask = self.updt_mask.clone()
        new_instance.neumann_mask = self.neumann_mask.clone()
        new_instance.neumann_mode = self.neumann_mode
        new_instance.N_us_tot = self.N_us_tot
        new_instance.N_us_grad = self.N_us_grad
        new_instance.N_pdes = self.N_pdes
        new_instance.N_component = self.N_component
        new_instance.N_deriv = self.N_deriv
        new_instance.deriv_calc = FinDerivCalcSPMV(self.graphs, mask, mask, self.N_component, device=self.device)
        new_instance.deriv_calc_bc = self.deriv_calc_bc
        # Keep bc_enforce, since it's not dependent on components
        new_instance.bc_enforce = self.bc_enforce
        return new_instance

    def reset(self):
        self._us = torch.zeros_like(self._us)

    def set_bc(self, dirich_bc=None):
        """ Set boundary conditions. """
        if dirich_bc is not None:
            dirich_bc = dirich_bc.to(self.device, non_blocking=True)
            assert self.dirich_mask.sum() == dirich_bc.numel(), "Dirichlet BC must match number of Dirichlet points."

            self._us[self.dirich_mask] = dirich_bc

        if self.neumann_mode:
            # bc_deriv = self.deriv_calc_bc.derivative(self._us)
            # print(bc_deriv)

            self.bc_enforce.set_bc_(self._us)




    def get_grads(self, get_orders=None):
        grad_dict = self.deriv_calc.derivative(self._us, get_orders=get_orders)
        return grad_dict

    def get_us_grad(self) -> list[torch.Tensor]:
        """ Returns us that need fitting, as list for each component since shapes are different.
            return.shape: [N_component][N_us_grad[i], 1]
        """
        # print(f'{self.grad_mask.shape = }')


        us_fit = [(self._us[:, i][self.updt_mask[:, i]]).unsqueeze(-1) for i in range(self.N_component)]
        return us_fit
        # [print(us_fit[i].shape) for i in range(self.N_component)]
        # assert False

    def update_grid(self, deltas):
        raise NotImplementedError

    def set_grid(self, new_us):
        """
        Set grid to new values.
        new_us.shape = [N_us_grad_total]
        """
        self._us[self.updt_mask] = new_us.flatten()

    def set_grid_irreg(self, new_us):
        """
        Set grid to new values, if grid is irregular accross components.
        new_us.shape = [N_component][N_us_grad_comp]
        """
        for i, us in enumerate(new_us):
            self._us[self.updt_mask[:, i], i] = us
        #self._us = new_us

    def _cuda(self):
        """ Move graph data to CUDA. """
        self._us = self._us.cuda(non_blocking=True)
        self._Xs = self._Xs.cuda(non_blocking=True)

        self.updt_mask = self.updt_mask.cuda(non_blocking=True)
        self.pde_mask = self.pde_mask.cuda(non_blocking=True)
        #self.dirich_mask = self.dirich_mask.cuda(non_blocking=True)
        #self.grad_mask = self.grad_mask.cuda(non_blocking=True)
        [graph.cuda() for graph in self.graphs.values()]

    def _cuda_bc(self):
        self.deriv_val = self.deriv_val.cuda(non_blocking=True)
        self.neumann_mask = self.neumann_mask.cuda(non_blocking=True)

