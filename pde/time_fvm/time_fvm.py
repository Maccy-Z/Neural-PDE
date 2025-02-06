import torch
from cprint import c_print
import pickle
import time
from abc import ABC, abstractmethod

from pde.config import Config
from pde.mesh_generation.generate_mesh import gen_mesh_fvm
from pde.graph_grid.graph_utils import plot_points, plot_interp_graph, plot_edges
from pde.time_dependent.time_cfg import ConfigTime
from pde.graph_grid.fvm_store import EdgeBCTypes as E
from pde.time_fvm.fvm_mesh import FVMMesh


class FVMCells:
    values: torch.Tensor  # shape = (n_cells, N_component)
    areas: torch.Tensor  # shape = (n_cells)
    tri_to_edge: torch.Tensor  # shape = (n_cells, 3)
    tri_edge_sign: torch.Tensor  # shape = (n_cells, 3)

    def __init__(self, mesh: FVMMesh, n_component, init_val=None):
        self.tri_to_edge = mesh.tri_to_edge
        self.tri_edge_sign = mesh.tri_edge_signs.unsqueeze(-1)
        self.areas = mesh.areas

        n_cells = mesh.n_cells
        if init_val is None:
            self.values = torch.zeros(n_cells, n_component)
        else:
            assert init_val.shape == (n_cells, n_component), f'Incorrect us init shape {init_val.shape = }'
            self.values = init_val.clone()

    def update_cells(self, fluxes, dt):
        """ Update cell values using fluxes.
            fluxes.shape = (n_edges, N_component)

            du/dt = -div(flux) = -sum_i (sign_i * flux_i)
        """

        # print(f'{self.tri_edge_sign.shape = }, {fluxes.shape = }, {self.tri_to_edge.shape = }')
        # val_test = self.values.clone()
        # for i, tri in enumerate(self.tri_to_edge):
        #     divergence = torch.sum(self.tri_edge_sign[i] * fluxes[tri], dim=0) / self.areas[i]
        #     self.values[i] -= dt * divergence

        # Vectorised version
        tri_fluxes = fluxes[self.tri_to_edge]  # shape: [n_tri, 3, 1]
        divergence = torch.sum(self.tri_edge_sign * tri_fluxes, dim=1).squeeze() / self.areas

        self.values -= dt * divergence.unsqueeze(-1)

class FVMEdgeInfo:
    n_edges: int
    n_cells: int
    n_component: int

    # Main mesh
    n_edges_m: int
    #normals: torch.Tensor  # shape = (n_edges_m, 2)
    tri_to_edge: torch.Tensor  # shape = (n_cells, 3)
    edge_to_tri_main: torch.Tensor  # shape = [n_edges_m, 2], ordered so triangle parallel to edge normal comes last, antiparallel first.
    edge_to_tri_w: torch.Tensor  # shape = [n_edges_m, 2]

    # Boundary condition
    n_edges_bc: int             # Number of boundary edges
    bc_edge_mask: torch.Tensor  # shape = (n_edges)
    normals_bc: torch.Tensor  # shape = (n_edges_bc, 2)
    edge_to_tri_bc: torch.Tensor  # shape = (n_edges_bc, n_component)
    dirich_mask: torch.Tensor # shape = (n_edges_bc)
    neumann_mask: torch.Tensor # shape = (n_edges_bc)
    dirich_val: torch.Tensor # shape = (n_edges_bc, n_component)
    neumann_val: torch.Tensor # shape = (n_edges_bc, n_component)

    # Gradients
    neigh_idx: torch.Tensor  # shape = (n_cells, 3)
    G_mat: torch.Tensor  # shape = (n_cells, 2, 3)      # Gradient matrix
    mask: torch.Tensor  # shape = (n_cells, 3)          Mask for valid neighbour cells on BC edges
    edge_disps: torch.Tensor  # shape = (n_edges_m, 2)     Displacement vector between cell centroids, for every edge_main
    edge_dists_bc: torch.Tensor  # shape = (n_bc_edges)     Distance between cell centroids, for every edge_bc

    # Temporary Variables
    cell_grads: torch.Tensor  # shape = (n_cells, 2, N_component)  Gradient of cell values
    grad_faces_n: torch.Tensor  # shape = (n_edges, N_component)  n . grad(u) on faces
    U_face: torch.Tensor  # shape = (n_edges, N_component)  Face values

    def __init__(self, mesh: FVMMesh, n_comp, bc_tags):
        self.n_edges = mesh.n_edges
        self.n_cells = mesh.n_cells
        self.n_component = n_comp

        self.normals_main = mesh.normals_main
        self.edge_to_tri_main = mesh.edge_to_tri_main
        self.edge_to_tri_w = mesh.edge_to_tri_w_main

        self.edge_to_tri_bc = mesh.edge_to_tri_bc
        self.bc_edge_mask = mesh.bc_edge_mask
        self.normals_bc = mesh.normals_bc

        (self.neigh_idx, self.G_mat, self.mask, self.edge_disps, self.edge_dists_bc) = mesh.cell_grad_stuff
        self.cell_dist = torch.norm(self.edge_disps, dim=1)

        self.bc_tags = bc_tags # {edge_num: bc_tag}
        self._init_bc(bc_tags)

    def _init_bc(self, bc_tags: dict[int, tuple[E, float]]):
        self.n_edges_m = self.n_edges - self.bc_edge_mask.sum().item()
        self.n_edges_bc = self.bc_edge_mask.sum().item()

        dirich_mask, neumann_mask = [], []
        dirich_val, neumann_val = [], []
        for bc_idx, (e_type, val) in bc_tags.items():
            if e_type == E.WALL:
                dirich_mask.append(True), neumann_mask.append(False)
                dirich_val.append([val])
            elif e_type == E.INLET:
                dirich_mask.append(True), neumann_mask.append(False)
                dirich_val.append([val])
            elif e_type == E.EXIT:
                dirich_mask.append(False), neumann_mask.append(True)
                neumann_val.append([val])
            else:
                raise ValueError(f'Unknown bc tag {e_type}')
        self.dirich_mask, self.neumann_mask = torch.tensor(dirich_mask), torch.tensor(neumann_mask)
        self.dirich_val, self.neumann_val = torch.tensor(dirich_val, dtype=torch.float32), torch.tensor(neumann_val, dtype=torch.float32)
        assert self.dirich_mask.shape[0] == self.bc_edge_mask.sum(), f'Wrong mask shape'
        assert self.neumann_mask.shape[0] == self.bc_edge_mask.sum(), f'Wrong mask shape'

        bc_indices = torch.nonzero(self.bc_edge_mask, as_tuple=False).squeeze()
        self.dirich_idx = bc_indices[self.dirich_mask]
        self.neumann_idx = bc_indices[self.neumann_mask]

    def precompute_shared(self, Us):
        """ Precompute shared values that are used multiple times later """
        self.cell_grads = self._cell_grads(Us)
        self.grad_faces_n = self._face_grads(Us)
        self.U_face = self._face_vals_bc(Us)


    def _cell_grads(self, Us):
        """ Vectorised gradient computation
            Gradient = G @ (u_neigh - u_cell)
            Us.shape = (n_cells, N_component)
            Returns: Gradient matrix of shape (n_cells, 2, N_component)
        """
        u_neigh = Us[self.neigh_idx]        # shape = [n_cells, max_neigh, N_component]
        u_diff = u_neigh - Us.unsqueeze(1)   # shape = [n_cells, max_neigh, N_component]
        u_diff = u_diff * self.mask.unsqueeze(-1)  # shape = [n_cells, max_neigh, N_component]
        cell_grads = torch.bmm(self.G_mat, u_diff)      # shape = [n_cells, 2, N_component]

        return cell_grads

    def _face_grads(self, Us):
        """ n . grad(U) on faces.
            Us.shape = (n_cells, N_component)
            Returns: shape = [n_edges, N_component]
        """
        dUdn_face = torch.empty((self.n_edges, self.n_component))

        # On faces
        U_centroid = Us[self.edge_to_tri_main]      # shape = [n_edges, 2, N_component]
        dU = U_centroid[:, 1] - U_centroid[:, 0]
        dUdn_face[~self.bc_edge_mask] = dU / self.cell_dist.unsqueeze(-1)       # shape = [n_edges, N_component]

        # On boundary. Either u or du/dn is given
        #dUdn_face_bc = torch.empty((self.bc_edge_mask.sum(), self.n_component))
        u_centroid_bc = Us[self.edge_to_tri_bc[:, 0]]  # shape = [n_bc_edges, N_component]
        # Dirichlet: n.grad(u) = 1/d * (u_bc - u)
        U_cent_bc_dir = u_centroid_bc[self.dirich_mask]     # shape = [n_dirich_edges, N_component]
        dudn_face_bc_dir = (self.dirich_val - U_cent_bc_dir) / self.edge_dists_bc[self.dirich_mask].unsqueeze(-1)
        dUdn_face[self.dirich_idx] = dudn_face_bc_dir
        # Neumann: n.grad(u) = du/dn
        dUdn_face[self.neumann_idx] = self.neumann_val

        return dUdn_face

    def _face_vals_bc(self, Us):
        """ U_face, with linear interpolation """
        # Us = Us.repeat(1, 2)
        # Us[:, 1] = 2 * Us[:, 1]

        U_face = torch.empty((self.n_edges, self.n_component))
        # Main edges
        # Weighted linear interpolation of two cell values
        U_centroid = Us[self.edge_to_tri_main]  # [n_edges_m, 2, n_component]
        w = self.edge_to_tri_w.unsqueeze(-1)  # shape: [n_edges, 2, 1]
        U_lin_main = (w * U_centroid).sum(dim=1)  # shape: [n_edges, n_component]
        U_face[~self.bc_edge_mask] = U_lin_main

        # Boundary edges
        # Dirichlet
        U_face[self.dirich_idx] = self.dirich_val
        # Neumann
        tris = self.edge_to_tri_bc[self.neumann_mask].squeeze()     # shape = [n_neum_edges]
        U_centroid = Us[tris]        # shape = [n_neum_edges, n_component]
        U_face_neum = U_centroid + self.neumann_val / self.edge_dists_bc[self.neumann_mask].unsqueeze(-1)
        U_face[self.neumann_idx] = U_face_neum

        return U_face


class FVMEdgeFunc(ABC):

    @abstractmethod
    def edge_fluxes(self, Us):
        """ Compute flux for each edge
        """
        pass

    @abstractmethod
    def _main_fluxes(self, Us):
        pass

    @abstractmethod
    def _bc_fluxes_(self, Us):
        pass


class ConvectScalar(FVMEdgeFunc):
    """ div(u V) for scalar u, fixed vector V"""
    E_props: FVMEdgeInfo

    def __init__(self, E_props: FVMEdgeInfo):
        self.E_props = E_props

    def _beta(self, r):
        # van Albada scheme
        beta = (r**2 + r) / (1 + r**2)
        # van Leer scheme
        #beta = (r + abs(r)) / (1 + abs(r))
        # minmod scheme
        #beta = torch.clamp(r, max=1)
        # limited linear
        #beta = torch.clamp(2 * r, max=1)
        return beta

    def edge_fluxes(self, Us):
        V_dir = torch.tensor([1., 0])
        V_dir = V_dir.unsqueeze(0).repeat(self.E_props.n_cells, 1)

        fluxes = torch.empty(self.E_props.n_edges, Us.shape[1])
        fluxes[~self.E_props.bc_edge_mask] = self._main_fluxes(Us, V_dir)
        fluxes[self.E_props.bc_edge_mask] = self._bc_fluxes_(Us)
        return fluxes

    def _main_fluxes(self, us, V_dir):
        """ Compute u_face for each edge.
            Us.shape = (n_cells, 1)
            V_dir.shape = (n_cells, 2)
            div(u V) = sum_i( V_f . dS_f * u_f)
                Linear flux interpolation
                Upwinding
        """
        E_props = self.E_props

        V_dir_cells = V_dir[E_props.edge_to_tri_main]     # [n_edges, 2, 2]

        u_centroid = us[E_props.edge_to_tri_main, 0]  # [n_edges_, 2]
        u_face_lin = E_props.U_face[~E_props.bc_edge_mask, 0]    # shape = [n_edges, N_component]
        # Linear interpolation of convection vector
        w = E_props.edge_to_tri_w.unsqueeze(1)  # shape: [n_edges, 1, 2]
        V_face = torch.bmm(w, V_dir_cells).squeeze(1)  # shape: [n_edges, 2]
        # phi = dS_face * V_face
        phi = (E_props.normals_main * V_face).sum(-1)

        # Upwinding
        # 1. Compute the upwind indicator.
        dot_vn = (E_props.normals_main * V_face).sum(dim=1)  # [n_edges]
        upwind = (dot_vn > 0).long()  # [n_edges], values 0 or 1
        edge_idx = torch.arange(E_props.n_edges_m)
        upwind_idx = E_props.edge_to_tri_main[edge_idx, upwind]  # [n_edges]
        # 2. Gather the upwind cell gradients.
        cell_grads = E_props.cell_grads[:, :, 0]  # [n_cells, N_component]
        upwind_grad = cell_grads[upwind_idx]  # [n_edges, 2]
        # 3. Compute the directional derivative at the face (du/dn_face).
        dudn_face = E_props.grad_faces_n[~E_props.bc_edge_mask]
        # 4. Compute the limiter. r = max( 2 * (grad_cell . d) / (|d| * du/dn_face) - 1 , 0)
        dot_disp_grad = (E_props.edge_disps * upwind_grad).sum(dim=1)  # [n_edges]
        denom = E_props.cell_dist * dudn_face.squeeze(-1) + 1e-7  # [n_edges]
        r = torch.clamp(2 * dot_disp_grad / denom - 1, min=0)  # enforce r >= 0, [n_edges]
        beta = self._beta(r)
        # 5. Corrected face value: u_face = (1 - beta) * u_upwind + beta * u_face_lin
        u_face_cor = (1 - beta) * u_centroid[edge_idx, upwind] + beta * u_face_lin  # [n_edges]
        # 6. Compute div(uV) = phi * u_face_cor
        div_uV = phi * u_face_cor       # [n_edges]

        return div_uV.unsqueeze(-1)

    def _bc_fluxes_(self, Us):
        E_props = self.E_props

        # Vectorised bc fluxes
        bc_fluxes = torch.empty(E_props.bc_edge_mask.sum())
        # Dirichlet
        bc_fluxes[E_props.dirich_mask] = E_props.dirich_val.squeeze()
        # Neumann
        tris = E_props.edge_to_tri_bc[E_props.neumann_mask].squeeze()
        face_scal = Us[tris]            # shape = [n_neum_edges]
        face_vect = face_scal * torch.tensor([1, 0]) # shape = [n_neum_edges, 2]
        flux_val = torch.sum(face_vect * E_props.normals_bc[E_props.neumann_mask], dim=-1)   # shape = [n_neum_edges]
        bc_fluxes[E_props.neumann_mask] = flux_val

        return bc_fluxes.unsqueeze(-1)


class FVMSolver:
    mesh: FVMMesh
    E_props: FVMEdgeInfo
    edges: FVMEdgeFunc
    cells: FVMCells

    def __init__(self, mesh: FVMMesh, n_comp, bc_tag, us_init=None):
        self.mesh = mesh
        self.E_props = FVMEdgeInfo(mesh, n_comp, bc_tag)
        self.edges = ConvectScalar(self.E_props)
        self.cells = FVMCells(mesh, n_comp, us_init)

        #self.plot_flux(torch.zeros(mesh.n_edges, n_comp))
        #self.plot_cells(self.cells.values[:, 0], title="Inital Cell Values")

        dt = 0.01
        for i in range(201):

            st = time.time()
            self.E_props.precompute_shared(self.cells.values)
            fluxes = self.edges.edge_fluxes(self.cells.values)
            self.cells.update_cells(fluxes, dt)
            print(f'{i = }, {time.time() - st = :.3g}')

            if i % 100 == 0:
                #self.plot_flux(fluxes, title=f"Fluxes at t={i * dt :.2g}")
                self.plot_cells(self.cells.values[:, 0], title=f"Values at t={i * dt :.2g}")


    def plot_flux(self, fluxes, title="Fluxes"):
        plot_edges(self.mesh.vertices.cpu(), self.mesh.edges.cpu(), title=title, color=fluxes[:, 0].abs())

    def plot_cells(self, value, title="Cell Values"):
        plot_points(self.mesh.centroids.cpu(), value, title=title)


def mesh_graph(cfg):
    N_comp = 2
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
            bc_tags[bc_idx] = (E.WALL, 0)
        elif e_tag == "Left":
            bc_tags[bc_idx] = (E.INLET, 0)
        elif e_tag == "Right":
            bc_tags[bc_idx] = (E.EXIT, 0)
        else:
            raise ValueError(f'Unknown edge tag {e_tag}')


    mesh = FVMMesh(Xs, tri_idx, all_edgs, bc_edge_mask)

    cent_x = mesh.centroids[:, 0].unsqueeze(-1).clone()
    #us_init = torch.exp(-((cent_x - 1.5) ** 2) / 1)
    us_init = (cent_x-3) ** 2
    solver = FVMSolver(mesh, 1, bc_tags, us_init=us_init)

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