import torch
from cprint import c_print
import pickle
import time
from abc import ABC, abstractmethod
from codetiming import Timer

from pde.config import Config
from pde.mesh_generation.generate_mesh import gen_mesh_fvm
from pde.graph_grid.graph_utils import plot_points, plot_interp_graph, plot_edges
from pde.time_dependent.time_cfg import ConfigTime
from pde.graph_grid.fvm_store import EdgeBCTypes as E
from pde.graph_grid.fvm_store import Edge
from pde.time_fvm.fvm_mesh import FVMMesh
from pde.time_fvm.edge_process import FVMEdgeInfo

class FVMCells:
    values: torch.Tensor  # shape = (n_cells, N_component)
    areas: torch.Tensor  # shape = (n_cells)
    tri_to_edge: torch.Tensor  # shape = (n_cells, 3)
    tri_edge_sign: torch.Tensor  # shape = (n_cells, 3)

    def __init__(self, mesh: FVMMesh, n_component, init_val=None, device="cpu"):
        self.device = device

        self.tri_to_edge = mesh.tri_to_edge
        self.tri_edge_sign = mesh.tri_edge_signs.unsqueeze(-1).to(device)
        self.areas = mesh.areas.to(device)

        n_cells = mesh.n_cells
        if init_val is None:
            self.values = torch.zeros(n_cells, n_component, device=device)
        else:
            assert init_val.shape == (n_cells, n_component), f'Incorrect us init shape {init_val.shape = }'
            self.values = init_val.clone().to(device)

    def update_cells(self, fluxes, dt):
        """ Update cell values using fluxes.
            fluxes.shape = (n_edges, N_component)

            du/dt = -div(flux) = -sum_i (sign_i * flux_i)
        """
        # Vectorised version
        tri_fluxes = fluxes[self.tri_to_edge]  # shape: [n_tri, 3, n_component]

        divergence = dt * torch.sum(self.tri_edge_sign * tri_fluxes, dim=1).squeeze() / self.areas.unsqueeze(-1)     # shape = [n_cells, N_component]
        self.values -=  divergence

        return divergence

    def get_primatives(self):
        momentum_x, momentum_y, density = self.values[:, 0], self.values[:, 1], self.values[:, 2]
        u_x, u_y = momentum_x / density, momentum_y / density
        primatives = torch.stack([u_x, u_y, density], dim=1)
        return primatives

class FVMEdgeFunc(ABC):
    device: str

    @abstractmethod
    def edge_fluxes(self, Us, *args):
        """ Compute flux for each edge
        """
        pass

    # @abstractmethod
    # def _main_fluxes(self, Us):
    #     pass
    #
    # @abstractmethod
    # def _bc_fluxes_(self, Us):
    #     pass


class Advect(FVMEdgeFunc, ABC):
    def __init__(self, E_props: FVMEdgeInfo, device="cpu"):
        self.edge_idx = torch.arange(E_props.n_edges_m, device=device)

    def _beta(self, r):
        # van Albada scheme
        #beta = (r**2 + r) / (1 + r**2)
        # van Leer scheme
        beta = (r + abs(r)) / (1 + abs(r))
        # minmod scheme
        #beta = torch.clamp(r, max=1)
        # limited linear
        #beta = torch.clamp(2 * r, max=1)
        return beta


class AdvectScalar(Advect):
    """ div(q V) for scalar u, fixed vector V.
        The convection dimension is dim.
    """
    E_props: FVMEdgeInfo
    rho_dim: int
    V_dims: list[int]

    def __init__(self, E_props: FVMEdgeInfo, V_dims, rho_dim, device="cpu"):
        super().__init__(E_props, device)
        self.device = device
        self.E_props = E_props

        self.rho_dim = rho_dim
        self.V_dims = V_dims

        # Mask for selecting q main edges
        row_mask = ~E_props.bc_edge_mask  # True for rows we want
        col_mask = torch.zeros(self.E_props.n_component, dtype=torch.bool, device=self.device)
        col_mask[list([rho_dim])] = True  # mark the desired columns
        self.q_main_mask = row_mask.unsqueeze(1) & col_mask.unsqueeze(0)
        # Mask for selecting q bc edges
        row_mask = E_props.bc_edge_mask  # True for rows we want
        self.q_bc_mask = row_mask.unsqueeze(1) & col_mask.unsqueeze(0)


    def edge_fluxes(self, Us):
        """ Compute flux for each edge.
            us.shape = (n_cells, n_component)
        """
        us = Us[:, self.rho_dim]

        fluxes = torch.zeros(self.E_props.n_edges, self.E_props.n_component, device=self.device)
        fluxes[self.q_main_mask] = self._main_fluxes(us)
        fluxes[self.q_bc_mask] = self._bc_fluxes_()
        return fluxes

    def _upwind_coef(self, V_face):
        E_props = self.E_props

        # 1. Compute the upwind indicator.
        dot_vn = (E_props.normals_main * V_face).sum(dim=1)  # [n_edges]
        upwind = torch.sign(dot_vn).long() # [n_edges], values 0 or 1
        upwind_idx = E_props.edge_to_tri_main[self.edge_idx, upwind]  # [n_edges]
        # 2. Gather the upwind cell gradients.
        upwind_grad = E_props.cell_grads[upwind_idx, :, self.rho_dim]  # [n_edges, 2]
        # 3. Compute the directional derivative at the face (du/dn_face)
        dudn_face = E_props.grad_faces_n[self.q_main_mask]
        # 4. Compute the limiter. r = max( 2 * (grad_cell . d) / (|d| * du/dn_face) - 1 , 0)
        dot_disp_grad = (E_props.edge_disps * upwind_grad).sum(dim=1)  # [n_edges]
        denom = E_props.cell_dist * dudn_face + 1e-7  # [n_edges]
        r = torch.clamp(2 * dot_disp_grad / denom - 1, min=0)  # enforce r >= 0, [n_edges]
        beta = self._beta(r)

        return beta, upwind

    def _main_fluxes(self, us):
        """ Compute u_face for each edge.
            div(u V) = sum_i( V_f . dS_f * u_f)
                Linear flux interpolation
                Upwinding

            Us.shape = (n_cells)
            V_dir.shape = (n_cells, 2)

            Return.shape = (n_edges_main)
        """
        E_props = self.E_props

        Us_face = E_props.U_face[~E_props.bc_edge_mask]  # shape = [n_edges, 2]
        V_face_lin = Us_face[:, self.V_dims]  # shape = [n_edges, 2]
        rho_face_lin = Us_face[:, self.rho_dim]    # shape = [n_edges]

        # phi = dS_face * V_face
        phi = (E_props.normals_main * V_face_lin).sum(-1)

        # Upwinding
        beta, upwind = self._upwind_coef(V_face_lin)

        # 5. Corrected face value: u_face = (1 - beta) * u_upwind + beta * u_face_lin
        rho_centroid = us[E_props.edge_to_tri_main]  # [n_edges_, 2]
        rho_face_cor = (1 - beta) * rho_centroid[self.edge_idx, upwind] + beta * rho_face_lin  # [n_edges]
        # 6. Compute div(uV) = phi * u_face_cor
        flux_uV = phi * rho_face_cor       # [n_edges]

        return flux_uV

    def _bc_fluxes_(self):
        """ Flux = rho_face * phi
            phi = n_face dot V_f.
        """
        E_props = self.E_props

        # Flux = U_face * phi
        # phi = normal dot Vf on face. Vf is the nearest cell value.
        U_face = E_props.U_face[E_props.bc_edge_mask]      # shape = [n_bc_edges, 3]
        rho_face = U_face[:, self.rho_dim]      # shape = [n_bc_edges]
        V_face = U_face[:, self.V_dims]                          # shape = [n_bc_edges, 2]

        # V_face = V_dir[E_props.edge_to_tri_bc]
        phi = (E_props.normals_bc * V_face).sum(-1)
        bc_fluxes = rho_face * phi

        return bc_fluxes


class AdvectVector(Advect):
    """ div(p U prod V) for advected vector U, fixed vector V.
        dims: Which dimensions of Us are advected.
    """
    E_props: FVMEdgeInfo
    V_dims: list[int]
    rho_dim: int

    def __init__(self, E_props: FVMEdgeInfo, V_dims: list[int], rho_dim:int, device="cpu"):
        super().__init__(E_props, device)
        self.device = device
        self.E_props = E_props

        self.V_dims = V_dims
        self.rho_dim = rho_dim

        # Mask for selecting V main edges
        row_mask = ~E_props.bc_edge_mask  # True for rows we want
        col_mask = torch.zeros(self.E_props.n_component, dtype=torch.bool, device=self.device)
        col_mask[list(self.V_dims)] = True  # mark the desired columns
        self.V_main_mask = row_mask.unsqueeze(1) & col_mask.unsqueeze(0)
        # Mask for selecting V bc edges
        row_mask = E_props.bc_edge_mask  # True for rows we want
        self.V_bc_mask = row_mask.unsqueeze(1) & col_mask.unsqueeze(0)

    def edge_fluxes(self, mom):
        """mom = rho * Vs  # shape = (n_cells, 2)"""
        E_props = self.E_props

        fluxes = torch.zeros(E_props.n_edges, E_props.n_component, device=self.device)
        fluxes[self.V_main_mask] = self._main_fluxes(mom).flatten()
        fluxes[self.V_bc_mask] = self._bc_fluxes_().flatten()

        return fluxes

    def _upwind_coef(self, V_face):
        """ V_face.shape = [n_edges_m, 2]"""
        E_props = self.E_props

        # 1. Find upwind cell
        dot_vn = (E_props.normals_main * V_face).sum(dim=1)  # [n_edges]
        upwind = torch.sign(dot_vn).long()  # [n_edges], values 0 or 1
        upwind_idx = E_props.edge_to_tri_main[self.edge_idx, upwind]  # [n_edges]
        # 2. Gather the upwind cell gradients.
        upwind_grad = E_props.cell_grads[upwind_idx][:, :, self.V_dims]  # [n_edges, 2, 2]
        # 3. Gather the directional derivative at the face (du/dn_face)
        dudn_face = E_props.grad_faces_n[self.V_main_mask].view(E_props.n_edges_m, 2) # [n_edges, 2]
        #dudn_face = E_props.grad_faces_n[~E_props.bc_edge_mask][:, self.V_dims]      # shape = [n_edges, 2]
        # 4. Compute the limiter. r = max( 2 * du/dn_face . (grad_cell . d) / (|d| * |du/dn_face|**2) - 1 , 0)
        d_dot_gradU = (E_props.edge_disps.unsqueeze(-1) * upwind_grad).sum(dim=1)  # [n_edges, 2]
        numerator = torch.sum(dudn_face * d_dot_gradU, dim=1)  # [n_edges]
        denom = E_props.cell_dist * torch.sum(dudn_face**2, dim=1) + 1e-7   # [n_edges]
        r = torch.clamp(2 * numerator / denom - 1, min=0)  # enforce r >= 0, [n_edges]
        beta = self._beta(r).unsqueeze(-1)  # [n_edges, 1]

        return beta, upwind

    def _main_fluxes(self, rho_Us):
        """ Compute u_face for each edge.
            div(u V) = sum_i( (V_f . dS_f) * u_f)
                Linear flux interpolation
                Upwinding

            Us.shape = (n_cells, n_component)
            V_dir.shape = (n_cells, 2)

            Return.shape = (n_edges_main)
        """
        E_props = self.E_props
        Us_face = E_props.U_face[~E_props.bc_edge_mask]      # shape = [n_bc_edges, 3]
        V_face_lin = Us_face[:, self.V_dims]  # shape = [n_bc_edges, n_component]
        rho_face = Us_face[:, self.rho_dim].unsqueeze(-1)  # shape = [n_bc_edges, 1]

        # Linear interpolation of convection vector
        # Face scalar: phi = dS_face * V_face
        phi = (E_props.normals_main * V_face_lin).sum(-1).unsqueeze(-1)

        # Upwinding: u_face = (1 - beta) * u_upwind + beta * u_face_lin
        # Compute upwind coefficient and upwind cell mask
        beta, upwind = self._upwind_coef(V_face_lin)

        # Corrected face value: u_face = (1 - beta) * u_upwind + beta * u_face_lin
        U_centroid = rho_Us[E_props.edge_to_tri_main]  # [n_edges_, 2, n_component]
        U_face_cor = (1 - beta) * U_centroid[self.edge_idx, upwind] + beta * V_face_lin * rho_face  # [n_edges, n_component]

        # 6. Compute div(uV) = phi * u_face_cor
        flux_UV = phi * U_face_cor       # [n_edges]
        return flux_UV

    def _bc_fluxes_(self):
        """ Flux = rho_face * V_face * phi
            phi = n_face dot V_face

            Use cached values.
        """
        E_props = self.E_props

        # Flux = rho U_face * phi
        Us_face = E_props.U_face[E_props.bc_edge_mask]      # shape = [n_bc_edges, 3]
        V_face = Us_face[:, self.V_dims]  # shape = [n_bc_edges, n_component]
        rho_face = Us_face[:, self.rho_dim].unsqueeze(-1)  # shape = [n_bc_edges, 1]

        phi = (E_props.normals_bc * V_face).sum(-1).unsqueeze(-1)       # shape = [n_bc_edges, 1]
        bc_fluxes = rho_face * V_face * phi         # shape = [n_bc_edges, n_component]
        return bc_fluxes


class Viscosity(FVMEdgeFunc):
    """ Viscous forces: div(mu grad(u)) = sum_f """
    E_props: FVMEdgeInfo
    dims: int
    def __init__(self, E_props: FVMEdgeInfo, dims, mu=0.01, device="cpu"):
        self.device = device
        self.E_props = E_props

        self.dims = dims

        self.mu = mu

    def edge_fluxes(self, mu=None):
        fluxes = torch.zeros(self.E_props.n_edges, self.E_props.n_component, device=self.device)
        fluxes[:, self.dims] = self._fluxes(mu)

        return fluxes


    def _fluxes(self, mu=None):
        E_props = self.E_props
        dUdn_face = E_props.grad_faces_n[:, self.dims]  # shape = [n_edges, n_component]
        edge_len = E_props.edge_len                       # shape = [n_edges]
        if mu is None:
            fluxes = edge_len.unsqueeze(-1) * dUdn_face * self.mu   # shape = [n_edges, n_component]
        else:
            fluxes = edge_len.unsqueeze(-1) * mu.unsqueeze(-1) * dUdn_face

        return fluxes


class Density(FVMEdgeFunc):
    """ Special case. N
        grad(p) = div(p I) """
    E_props: FVMEdgeInfo
    p_dim: int
    V_dims: list[int]

    def __init__(self, E_props: FVMEdgeInfo, p_dim: int, V_dims: list[int], device="cpu"):
        self.device = device
        self.E_props = E_props

        self.p_dim = p_dim
        self.V_dims = V_dims

    def edge_fluxes(self):
        fluxes = torch.zeros(self.E_props.n_edges, self.E_props.n_component, device=self.device)

        E_props = self.E_props
        u_face = E_props.U_face[:, self.p_dim]  # shape = [n_edges]
        normals = E_props.normals                       # shape = [n_edges, 2]

        fluxes[:, self.V_dims] = normals * u_face.unsqueeze(-1)     # shape = [n_edges, n_component]
        return fluxes


class FVMSolver:
    mesh: FVMMesh
    E_props: FVMEdgeInfo
    edges: FVMEdgeFunc
    cells: FVMCells

    def __init__(self, mesh: FVMMesh, n_comp, bc_tag, us_init=None, device="cuda"):
        self.device = device
        self.mesh = mesh
        self.E_props = FVMEdgeInfo(mesh, n_comp, bc_tag, device=device)
        self.cells = FVMCells(mesh, n_comp, us_init, device=device)

        self.P_advect = AdvectScalar(self.E_props, V_dims=[0, 1], rho_dim=2, device=device)
        self.P_force = Density(self.E_props, p_dim=2, V_dims=[0, 1], device=device)
        self.U_advect = AdvectVector(self.E_props, V_dims=[0, 1], rho_dim=2, device=device)
        self.U_visc = Viscosity(self.E_props, mu=0.005, dims=[0, 1], device=device)


        dt = 0.01
        for i in range(300):
            st = time.time()

            # cells: [momentum_x, momentum_y, density]
            primatives = self.cells.get_primatives()
            fluxes = torch.zeros(self.mesh.n_edges, n_comp, device=device)

            self.E_props.precompute_shared(primatives)
            # with Timer(name="Advect P", text="U_advect: {:.3g}"):
            #     self.E_props.precompute_shared(primatives)
            #     torch.cuda.synchronize()
            # with Timer(name="Advect P", text="U_visc: {:.3g}"):
            #     fluxes += self.U_visc.edge_fluxes()
            #     torch.cuda.synchronize()
            # with Timer(name="Advect P", text="P_force: {:.3g}"):
            #     fluxes += self.P_force.edge_fluxes()
            #     torch.cuda.synchronize()
            #
            # with Timer(name="Advect P", text="Final: {:.3g}"):
            #     self.cells.update_cells(fluxes, dt)
            #     torch.cuda.synchronize()

            fluxes += self.P_advect.edge_fluxes(primatives)
            fluxes += self.U_advect.edge_fluxes(self.cells.values[:, :2])
            fluxes += self.U_visc.edge_fluxes()
            fluxes += self.P_force.edge_fluxes()
            self.cells.update_cells(fluxes, dt)

            torch.cuda.synchronize()
            print(f'{i = }, {time.time() - st = :.3g}')
            #
            if i > 250 and i % 5 == 0:

                # edge_ln = self.E_props.edge_len
                # self.plot_flux(fluxes / edge_ln.unsqueeze(-1), title=f"Fluxes at t={i * dt :.2g}")
                # print(f'{fluxes.shape = }')
                # print(f'{self.E_props.U_face = }')
                #
                # self.plot_flux(self.E_props.U_face, title=f"Value at t={i * dt :.2g}")
                #
                self.plot_cells(self.cells.values, title=f"Values at t={i * dt :.2g}")
                # self.plot_cells(-dUdt, title=f"dUdt at t={i * dt :.2g}", convert=False)

                # exit(4)

    def plot_flux(self, fluxes, title="Fluxes"):
        plot_edges(self.mesh.vertices.cpu(), self.mesh.edges.cpu(), title=title, color=fluxes.abs())


    def plot_cells(self, values, title="Cell Values", convert=True):
        if convert:
            momentum_x, momentum_y, density = values[:, 0], values[:, 1], values[:, 2]
            u_x, u_y = momentum_x / density, momentum_y / density
            density = density
            values = torch.stack([u_x, u_y, density], dim=1)
        plot_points(self.mesh.centroids.cpu(), values.T, title=title)


def mesh_graph(cfg):
    N_comp = 3

    new_graph=False
    if new_graph:
        xmin, xmax = 0, 3
        ymin, ymax = 0.0, 1.5
        mesh_stuff = gen_mesh_fvm(xmin, xmax, ymin, ymax, areas=[10e-3, 15e-3])
        Xs, tri_idx, (int_edgs, bound_edgs), edge_tag = mesh_stuff
        pickle.dump(mesh_stuff, open("mesh_stuff.pkl", "wb"))
    else:
        mesh_stuff = pickle.load(open("mesh_stuff.pkl", "rb"))
        Xs, tri_idx, (int_edgs, bound_edgs), edge_tag = mesh_stuff

    Xs = torch.from_numpy(Xs).float()
    tri_idx = torch.from_numpy(tri_idx).int()
    int_edgs, bound_edgs = torch.from_numpy(int_edgs), torch.from_numpy(bound_edgs)
    all_edgs = torch.cat([int_edgs, bound_edgs], dim=0)
    bc_edge_mask = torch.cat([torch.zeros_like(int_edgs[:, 0], dtype=torch.bool), torch.ones_like(bound_edgs[:, 0], dtype=torch.bool)], dim=0)

    bc_tags = {}
    for bc_idx, (e_tag, e_vert) in enumerate(zip(edge_tag, bound_edgs, strict=True)):
        if e_tag == "Wall":
            bc_tags[bc_idx] = Edge([E.Dirich, E.Dirich, E.Neuman], [0, 0, None], [None, None, 0])   #(E.WALL, 0)
        elif e_tag == "Left":
            bc_tags[bc_idx] = Edge([E.Dirich, E.Dirich, E.Neuman], [0.1, 0, None], [None, None, 0]) #(E.INLET, 0)
        elif e_tag == "Right":
            bc_tags[bc_idx] = Edge([E.Neuman, E.Neuman, E.Dirich], [None, None, 1], [0, 0, None])  #(E.EXIT, 0)
        else:
            raise ValueError(f'Unknown edge tag {e_tag}')


    mesh = FVMMesh(Xs, tri_idx, all_edgs, bc_edge_mask)

    cent_x = mesh.centroids[:, 0].unsqueeze(-1).clone()
    #us_init = torch.exp(-((cent_x - 1.5) ** 2) / 1)
    us_init = (cent_x-3) ** 2
    us_init = us_init.repeat(1, 3)
    us_init[:, 0] = us_init[:, 0] * 0.0 + 0.1
    us_init[:, 1] *= 0.0
    us_init[:, 2] = us_init[:, 2] * 0.0 + 1

    solver = FVMSolver(mesh, N_comp, bc_tags, us_init=us_init)

    exit(4)
    c_print(f'Number of mesh points: {len(Xs)}', "green")



def main():
    # from pde.utils import setup_logging

    # setup_logging(debug=False)

    cfg = Config()
    time_cfg= ConfigTime()
    c_print(f'{time_cfg.dt = }', color="bright_magenta")

    #u_g_T = load_graph(cfg)
    u_g_T = mesh_graph(cfg)

    # time_pde = TimePDEBase(u_g_T, time_cfg, cfg)
    # time_pde.solve()

    # saved_graphs = time_pde.u_saves
    # for t, graph in saved_graphs.items():
    #     us, Xs = graph.us, graph.Xs
    #     plot_interp_graph(Xs, us[:, 0], title=f"t={t :.4g}")


if __name__ == "__main__":
    main()