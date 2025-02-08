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
from pde.graph_grid.fvm_store import Edge
from pde.time_fvm.fvm_mesh import FVMMesh


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

        divergence = torch.sum(self.tri_edge_sign * tri_fluxes, dim=1).squeeze() / self.areas.unsqueeze(-1)     # shape = [n_cells, N_component]
        self.values -= dt * divergence


class FVMEdgeInfo:
    device: str
    n_edges: int
    n_cells: int
    n_component: int

    # Shared
    edge_len: torch.Tensor  # shape = (n_edges)
    normals: torch.Tensor  # shape = (n_edges, 2)
    # Main mesh
    n_edges_m: int
    tri_to_edge: torch.Tensor  # shape = (n_cells, 3)
    edge_to_tri_main: torch.Tensor  # shape = [n_edges_m, 2], ordered so triangle parallel to edge normal comes last, antiparallel first.
    edge_to_tri_w: torch.Tensor  # shape = [n_edges_m, 2]

    # Boundary condition
    n_edges_bc: int             # Number of boundary edges
    bc_edge_mask: torch.Tensor  # shape = (n_edges)
    normals_bc: torch.Tensor  # shape = (n_edges_bc, 2)
    edge_to_tri_bc: torch.Tensor  # shape = (n_edges_bc)
    dirich_mask: torch.Tensor # shape = (n_edges_bc, n_component)
    neumann_mask: torch.Tensor # shape = (n_edges_bc, n_component)
    dirich_val: torch.Tensor # shape: Us[dirich_mask] = dirich_val
    neumann_val: torch.Tensor # shape: Us[neumann_mask] = neumann_val

    # Gradients
    G_mats: list[torch.Tensor]  # shape = [2](n_cells, n_cells)  Gradient matrix for every cell
    edge_disps: torch.Tensor  # shape = (n_edges_m, 2)     Displacement vector between cell centroids, for every edge_main
    edge_dists_bc: torch.Tensor  # shape = (n_bc_edges)     Distance between cell centroids, for every edge_bc

    # Temporary Variables
    cell_grads: torch.Tensor  # shape = (n_cells, 2, N_component)  Gradient of cell values
    grad_faces_n: torch.Tensor  # shape = (n_edges, N_component)  n . grad(u) on faces
    U_face: torch.Tensor  # shape = (n_edges, N_component)  Face values

    def __init__(self, mesh: FVMMesh, n_comp, bc_tags, device="cpu"):
        self.device = device
        self.n_edges = mesh.n_edges
        self.n_cells = mesh.n_cells
        self.n_component = n_comp

        self.normals_main = mesh.normals_main.to(device)
        self.edge_to_tri_main = mesh.edge_to_tri_main.to(device)
        self.edge_to_tri_w = mesh.edge_to_tri_w_main.to(device)

        self.edge_to_tri_bc = mesh.edge_to_tri_bc.to(device)
        self.bc_edge_mask = mesh.bc_edge_mask.to(device)
        self.normals_bc = mesh.normals_bc.to(device)

        (edge_disps, edge_dists_bc, G_mats) = mesh.cell_grad_stuff
        self.edge_disps = edge_disps.to(device)
        self.edge_dists_bc = edge_dists_bc.to(device).unsqueeze(-1).expand(-1, self.n_component)
        self.G_mats = []
        for G in G_mats:
            self.G_mats.append(G.to(device))

        self.cell_dist = torch.norm(self.edge_disps, dim=1).to(device)
        self.normals = mesh.normals.to(device)
        self.edge_len = torch.norm(self.normals, dim=1).to(device)

        self.bc_tags = bc_tags # {edge_num: bc_tag}
        self._init_bc(bc_tags)

    def _init_bc(self, bc_tags: dict[int, Edge]):
        self.n_edges_m = self.n_edges - self.bc_edge_mask.sum().item()
        self.n_edges_bc = self.bc_edge_mask.sum().item()

        dirich_mask, neumann_mask = [], []
        dirich_val, neumann_val = [], []
        for bc_idx, e_type in bc_tags.items():
            # print(f'{e_type = }')
            # print(e_type.dirichlet())

            dirich_mask.append(e_type.dirichlet())
            neumann_mask.append(e_type.neumann())
            dirich_val.append(e_type.U)
            neumann_val.append(e_type.dUdn)

        self.dirich_mask, self.neumann_mask = torch.tensor(dirich_mask, device=self.device), torch.tensor(neumann_mask, device=self.device)
        dirich_val, neumann_val = torch.tensor(dirich_val, dtype=torch.float32, device=self.device), torch.tensor(neumann_val, dtype=torch.float32, device=self.device)
        self.dirich_val = dirich_val[dirich_mask]
        self.neumann_val = neumann_val[neumann_mask]

        assert self.dirich_mask.shape[0] == self.bc_edge_mask.sum(), f'Wrong mask shape'
        assert self.neumann_mask.shape[0] == self.bc_edge_mask.sum(), f'Wrong mask shape'

        # bc_indices = torch.nonzero(self.bc_edge_mask, as_tuple=False).squeeze()
        self.dirich_all = torch.zeros(self.n_edges, self.n_component, dtype=torch.bool, device=self.device)
        self.dirich_all[self.bc_edge_mask] = self.dirich_mask
        self.neum_all = torch.zeros(self.n_edges, self.n_component, dtype=torch.bool, device=self.device)
        self.neum_all[self.bc_edge_mask] = self.neumann_mask

        # self.neumann_idx = bc_indices[self.neumann_mask]

    def precompute_shared(self, Us):
        """ Precompute shared values that are used multiple times later """
        self.cell_grads = self._cell_grads(Us)
        self.grad_faces_n = self._face_grads(Us)
        self.U_face = self._face_vals(Us)


    def _cell_grads(self, Us):
        """ Vectorised gradient computation
            Gradient = G @ (u_neigh - u_cell)
            Us.shape = (n_cells, N_component)
            Returns: Gradient matrix of shape (n_cells, 2, N_component)
        """
        grad_x = torch.sparse.mm(self.G_mats[0], Us)  # Shape: [n_cells, N_component]
        grad_y = torch.sparse.mm(self.G_mats[1], Us)  # Shape: [n_cells, N_component]
        cell_grads = torch.stack([grad_x, grad_y], dim=1)  # Shape: [n_cells, 2, N_component]
        return cell_grads

    def _face_grads(self, Us):
        """ n . grad(U) on faces.
            Us.shape = (n_cells, N_component)
            Returns: shape = [n_edges, N_component]
        """
        dUdn_face = torch.empty((self.n_edges, self.n_component), device=self.device)

        # On faces
        U_centroid = Us[self.edge_to_tri_main]      # shape = [n_edges, 2, N_component]
        dU = U_centroid[:, 1] - U_centroid[:, 0]
        dUdn_face[~self.bc_edge_mask] = dU / self.cell_dist.unsqueeze(-1)       # shape = [n_edges, N_component]

        # On boundary. Either u or du/dn is given
        u_centroid_bc = Us[self.edge_to_tri_bc]  # shape = [n_bc_edges, N_component]
        # Dirichlet: n.grad(u) = 1/d * (u_bc - u)
        U_cent_bc_dir = u_centroid_bc[self.dirich_mask]     # shape = [n_dirich_edges]
        edge_dists = self.edge_dists_bc[self.dirich_mask]    # shape = [n_dirich_edges]
        dudn_face_bc_dir = (self.dirich_val - U_cent_bc_dir) / edge_dists
        dUdn_face[self.dirich_all] = dudn_face_bc_dir
        # Neumann: n.grad(u) = du/dn
        dUdn_face[self.neum_all] = self.neumann_val

        return dUdn_face

    def _face_vals(self, Us):
        """ U_face, with linear interpolation """
        # Us = Us.repeat(1, 2)
        # Us[:, 1] = 2 * Us[:, 1]

        U_face = torch.empty((self.n_edges, self.n_component), device=self.device)
        # Main edges
        # Weighted linear interpolation of two cell values
        U_centroid = Us[self.edge_to_tri_main]  # [n_edges_m, 2, n_component]
        w = self.edge_to_tri_w.unsqueeze(-1)  # shape: [n_edges, 2, 1]
        U_lin_main = (w * U_centroid).sum(dim=1)  # shape: [n_edges, n_component]
        U_face[~self.bc_edge_mask] = U_lin_main

        # Boundary edges
        u_centroid_bc = Us[self.edge_to_tri_bc] # shape = [n_bc_edges, N_component]
        # Dirichlet
        U_face[self.dirich_all] = self.dirich_val
        # Neumann
        U_cent_bc_neum = u_centroid_bc[self.neumann_mask]        # shape = [n_neum_edges]
        U_face_neum = U_cent_bc_neum + self.neumann_val / self.edge_dists_bc[self.neumann_mask]
        U_face[self.neum_all] = U_face_neum

        return U_face


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


class ConvectScalar(FVMEdgeFunc):
    """ div(u V) for scalar u, fixed vector V.
        The convection dimension is dim.
    """
    E_props: FVMEdgeInfo
    dim: int
    def __init__(self, E_props: FVMEdgeInfo, dim, device="cpu"):
        self.device = device
        self.E_props = E_props

        self.dim = dim

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

    def edge_fluxes(self, Us, V_dir):
        """ Compute flux for each edge.
            us.shape = (n_cells, n_component)
        """
        us = Us[:, self.dim]

        fluxes = torch.zeros(self.E_props.n_edges, self.E_props.n_component, device=self.device)
        fluxes[~self.E_props.bc_edge_mask, self.dim] = self._main_fluxes(us, V_dir)
        fluxes[self.E_props.bc_edge_mask, self.dim] = self._bc_fluxes_(us, V_dir)

        return fluxes

    def _main_fluxes(self, us, V_dir):
        """ Compute u_face for each edge.
            div(u V) = sum_i( V_f . dS_f * u_f)
                Linear flux interpolation
                Upwinding

            Us.shape = (n_cells)
            V_dir.shape = (n_cells, 2)

            Return.shape = (n_edges_main)
        """

        E_props = self.E_props

        V_dir_cells = V_dir[E_props.edge_to_tri_main]     # [n_edges, 2, 2]

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
        upwind_grad = E_props.cell_grads[upwind_idx, :, self.dim]  # [n_edges, 2]
        # 3. Compute the directional derivative at the face (du/dn_face)
        dudn_face = E_props.grad_faces_n[~E_props.bc_edge_mask, self.dim]
        # 4. Compute the limiter. r = max( 2 * (grad_cell . d) / (|d| * du/dn_face) - 1 , 0)
        dot_disp_grad = (E_props.edge_disps * upwind_grad).sum(dim=1)  # [n_edges]
        denom = E_props.cell_dist * dudn_face.squeeze(-1) + 1e-7  # [n_edges]
        r = torch.clamp(2 * dot_disp_grad / denom - 1, min=0)  # enforce r >= 0, [n_edges]
        beta = self._beta(r)
        # 5. Corrected face value: u_face = (1 - beta) * u_upwind + beta * u_face_lin
        u_centroid = us[E_props.edge_to_tri_main]  # [n_edges_, 2]
        u_face_lin = E_props.U_face[~E_props.bc_edge_mask, self.dim]    # shape = [n_edges]
        u_face_cor = (1 - beta) * u_centroid[edge_idx, upwind] + beta * u_face_lin  # [n_edges]
        # 6. Compute div(uV) = phi * u_face_cor
        flux_uV = phi * u_face_cor       # [n_edges]

        #self.beta = dot_disp_grad / denom
        return flux_uV

    def _bc_fluxes_(self, us, V_dir):
        """ Flux = U_face * phi
            phi = n_face dot V_f. V_f extrapolated from nearest cell value.
        """
        E_props = self.E_props

        # Flux = U_face * phi
        # phi = normal dot Vf on face. Vf is the nearest cell value.
        U_face = E_props.U_face[E_props.bc_edge_mask, self.dim]      # shape = [n_bc_edges]
        V_face = V_dir[E_props.edge_to_tri_bc]                          # shape = [n_bc_edges, 2]
        phi = (E_props.normals_bc * V_face).sum(-1)
        bc_fluxes = U_face * phi

        return bc_fluxes


class AdvectVector(FVMEdgeFunc):
    """ div(U prod V) for advected vector U, fixed vector V.
        dims: Which dimensions of Us are advected.
    """
    E_props: FVMEdgeInfo
    dims: list[int]
    def __init__(self, E_props: FVMEdgeInfo, dims: list[int], device="cpu"):
        self.device = device
        self.E_props = E_props

        self.dims = dims

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

    def edge_fluxes(self, Us, V_dir):
        Us = Us[:, self.dims]

        fluxes = torch.zeros(self.E_props.n_edges, self.E_props.n_component, device=self.device)

        f = torch.empty(self.E_props.n_edges, len(self.dims), device=self.device)
        f[~self.E_props.bc_edge_mask] = self._main_fluxes(Us, V_dir)
        f[self.E_props.bc_edge_mask] = self._bc_fluxes_(Us, V_dir)

        fluxes[:, self.dims] = f

        return fluxes

    def _main_fluxes(self, Us, V_dir):
        """ Compute u_face for each edge.
            div(u V) = sum_i( V_f . dS_f * u_f)
                Linear flux interpolation
                Upwinding

            Us.shape = (n_cells, n_component)
            V_dir.shape = (n_cells, 2)

            Return.shape = (n_edges_main)
        """
        E_props = self.E_props

        # Linear interpolation of convection vector
        V_dir_cells = V_dir[E_props.edge_to_tri_main]     # [n_edges, 2, 2]
        w = E_props.edge_to_tri_w.unsqueeze(1)  # shape: [n_edges, 1, 2]
        V_face = torch.bmm(w, V_dir_cells).squeeze(1)  # shape: [n_edges, 2]
        # Face scalar: phi = dS_face * V_face
        phi = (E_props.normals_main * V_face).sum(-1).unsqueeze(-1)

        # Upwinding
        # 1. Compute the upwind indicator.
        dot_vn = (E_props.normals_main * V_face).sum(dim=1)  # [n_edges]
        upwind = (dot_vn > 0).long()  # [n_edges], values 0 or 1
        edge_idx = torch.arange(E_props.n_edges_m)
        upwind_idx = E_props.edge_to_tri_main[edge_idx, upwind]  # [n_edges]
        # 2. Gather the upwind cell gradients.
        upwind_grad = E_props.cell_grads[upwind_idx][:, :, self.dims]  # [n_edges, 2, n_component]
        # 3. Gather the directional derivative at the face (du/dn_face)
        dudn_face = E_props.grad_faces_n[~E_props.bc_edge_mask][:, self.dims]      # shape = [n_edges, n_component]
        # 4. Compute the limiter. r = max( 2 * (grad_cell . d) / (|d| * du/dn_face) - 1 , 0)
        d_dot_gradU = (E_props.edge_disps.unsqueeze(-1) * upwind_grad).sum(dim=1)  # [n_edges, n_component]
        numerator = torch.sum(dudn_face * d_dot_gradU, dim=1)  # [n_edges]
        denom = E_props.cell_dist * torch.sum(dudn_face**2, dim=1) + 1e-7   # [n_edges]
        r = torch.clamp(2 * numerator / denom - 1, min=0)  # enforce r >= 0, [n_edges]
        beta = self._beta(r).unsqueeze(-1)  # # [n_edges, 1]

        # 5. Corrected face value: u_face = (1 - beta) * u_upwind + beta * u_face_lin
        U_centroid = Us[E_props.edge_to_tri_main]  # [n_edges_, 2, n_component]
        U_face_lin = E_props.U_face[~E_props.bc_edge_mask][:, self.dims]    # shape = [n_edges, n_component]
        U_face_cor = (1 - beta) * U_centroid[edge_idx, upwind] + beta * U_face_lin  # [n_edges, n_component]

        # 6. Compute div(uV) = phi * u_face_cor
        flux_UV = phi * U_face_cor       # [n_edges]
        return flux_UV

    def _bc_fluxes_(self, Us, V_dir):
        """ Flux = U_face * phi
            phi = n_face dot V_f. V_f extrapolated from nearest cell value.
        """
        E_props = self.E_props

        # Flux = U_face * phi
        # phi = normal dot Vf on face. Vf is the nearest cell value.
        U_face = E_props.U_face[E_props.bc_edge_mask][:, self.dims]      # shape = [n_bc_edges, n_component]
        V_face = V_dir[E_props.edge_to_tri_bc]                          # shape = [n_bc_edges, 2]
        phi = (E_props.normals_bc * V_face).sum(-1).unsqueeze(-1)
        bc_fluxes = U_face * phi

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

class Pressure(FVMEdgeFunc):
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

        self.P_advect = ConvectScalar(self.E_props, dim=2, device=device)
        self.P_force = Pressure(self.E_props, p_dim=2, V_dims=[0, 1], device=device)
        self.U_advect = AdvectVector(self.E_props, dims=[0, 1], device=device)
        self.U_visc = Viscosity(self.E_props, mu=0.0, dims=[0, 1], device=device)

        #self.plot_flux(torch.zeros(mesh.n_edges, n_comp))
        #self.plot_cells(self.cells.values[:, 0], title="Inital Cell Values")

        dt = 0.01
        for i in range(6 ):

            st = time.time()

            # cells: [momentum_x, momentum_y, density]
            self.cells.values[:, 2].clamp_(min=0.1)
            momentum_x, momentum_y, density = self.cells.values[:, 0], self.cells.values[:, 1], self.cells.values[:, 2]
            u_x, u_y = momentum_x / density, momentum_y / density
            primatives = torch.stack([u_x, u_y, density], dim=1)
            self.E_props.precompute_shared(primatives)
            fluxes = torch.zeros(self.mesh.n_edges, n_comp, device=device)

            fluxes += self.P_advect.edge_fluxes(primatives, primatives[:, :2])
            fluxes += self.U_advect.edge_fluxes(primatives,  primatives[:, :2])
            #fluxes += self.U_advect.edge_fluxes(density.unsqueeze(-1) * primatives,  primatives[:, :2])

            #fluxes += self.U_visc.edge_fluxes()
            #fluxes += self.P_force.edge_fluxes()

            self.cells.update_cells(fluxes, dt)
            self.cells.values[:, 2].clamp_(min=0.1)

            torch.cuda.synchronize()
            print(f'{i = }, {time.time() - st = :.3g}')

            if i % 1 == 0:
                #self.plot_flux(fluxes[:, 0], title=f"Fluxes at t={i * dt :.2g}")
                self.plot_cells(self.cells.values, title=f"Values at t={i * dt :.2g}")
                # self.plot_cells2(self.P_advect.beta, title=f"Values at t={i * dt :.2g}")

            exit(7)

    def plot_flux(self, fluxes, title="Fluxes"):
        plot_edges(self.mesh.vertices.cpu(), self.mesh.edges.cpu(), title=title, color=fluxes.abs())


    def plot_cells(self, values, title="Cell Values"):
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
            bc_tags[bc_idx] = Edge([E.Dirich, E.Dirich, E.Dirich], [0.10, 0, 2], [None, None, None]) #(E.INLET, 0)
        elif e_tag == "Right":
            bc_tags[bc_idx] = Edge([E.Neuman, E.Neuman, E.Dirich], [None, None, 2], [0, 0, None])  #(E.EXIT, 0)
        else:
            raise ValueError(f'Unknown edge tag {e_tag}')


    mesh = FVMMesh(Xs, tri_idx, all_edgs, bc_edge_mask)

    cent_x = mesh.centroids[:, 0].unsqueeze(-1).clone()
    #us_init = torch.exp(-((cent_x - 1.5) ** 2) / 1)
    us_init = (cent_x-3) ** 2
    us_init = us_init.repeat(1, 3)
    us_init[:, 0] = us_init[:, 0] * 0 + 0.2
    us_init[:, 1] *= 0.0
    us_init[:, 2] = us_init[:, 2] * 0.0 + 2

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