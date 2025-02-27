import torch
from abc import ABC, abstractmethod
from codetiming import Timer
from cprint import c_print

from pde.graph_grid.graph_utils import plot_points, plot_interp_graph, plot_edges
from pde.time_fvm.fvm_mesh import FVMMesh
from pde.time_fvm.edge_process import FVMEdgeInfo, create_selection_matrix, invert_selection_matrix
from pde.time_fvm.t_solvers import FVMCells, Euler, ExplMidpoint


def create_insertion_matrix(num_blocks, full_block_size, selected_indices, device=None, dtype=torch.float32):
    """
    Instead of fluxes[:, idxs] = A, use fluxes = S @ A.flatten()

    Create a sparse matrix S that maps a flattened tensor with shape
      (num_blocks * len(selected_indices))
    to a flattened tensor with shape
      (num_blocks * full_block_size)
    by scattering the values into positions determined by selected_indices for each block.

    For each block i and for each local index j (with v = selected_indices[j]),
    set:
        S[i * full_block_size + v,  i * len(selected_indices) + j] = 1.

    Args:
        num_blocks (int): Number of blocks (e.g. n_edges).
        full_block_size (int): Size of the full block (e.g. n_component).
        selected_indices (list or iterable): Indices within each block where values should be inserted.
        device (torch.device, optional): Device for the resulting tensor.
        dtype (torch.dtype, optional): Data type for the values.

    Returns:
        torch.Tensor: A sparse matrix of shape (num_blocks * full_block_size, num_blocks * len(selected_indices)).
    """
    num_selected = len(selected_indices)
    total_rows = num_blocks * full_block_size
    total_cols = num_blocks * num_selected

    row_indices = []
    col_indices = []
    values = []

    for block in range(num_blocks):
        for j, v in enumerate(selected_indices):
            row = block * full_block_size + v
            col = block * num_selected + j
            row_indices.append(row)
            col_indices.append(col)
            values.append(1.0)

    indices = torch.tensor([row_indices, col_indices], dtype=torch.long, device=device)
    values = torch.tensor(values, dtype=dtype, device=device)
    S = torch.sparse_coo_tensor(indices, values, (total_rows, total_cols))
    return S



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
        # print(f'{r.max() = }')
        beta = (r + abs(r)) / (1 + abs(r))
        # minmod scheme
        #beta = torch.clamp(r, max=1)
        # limited linear
        #beta = torch.clamp(2 * r, max=1)
        # print(f'{beta.abs().max() = }')
        return beta


class AdvectVector(Advect):
    """ out_i = div(rho V V_i) for velocity V, i = {x, y}
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

        self.proj_mat = E_props.V_insertion_matrix

    #@torch.compile()
    def edge_fluxes(self):
        #fluxes_all = torch.zeros(self.E_props.n_edges, self.E_props.n_component, device=self.device)

        E_props = self.E_props
        rho_faces = E_props.rho_faces  # shape = [n_edges, edges=2, n_comp=1]
        V_faces = E_props.Vs_faces  # shape = [n_edges, edges=2, n_comp=2]
        phi = E_props.phi           # Linear interpolation of convection vector = (v_faces dot normal). shape = [n_edges, 1]

        rho_u = rho_faces * V_faces         # shape = [n_edges, edges=2, n_comp=2]
        rho_u = rho_u.mean(dim=1)  # shape = [n_edges, n_comp=2]

        advc_flux = phi * rho_u  # shape = [n_edges, n_comp=2]

        # fluxes_all[:, self.V_dims] = advc_flux
        # fluxes_flat = fluxes_all.flatten()
        fluxes_flat = self.proj_mat @ advc_flux.flatten()
        return fluxes_flat



class Viscosity(FVMEdgeFunc):
    """ Viscous forces: div(mu grad(u)) = sum_f """
    E_props: FVMEdgeInfo
    V_dims: list[int]
    def __init__(self, E_props: FVMEdgeInfo, V_dims, mu=0.01, device="cpu"):
        self.device = device
        self.E_props = E_props

        self.V_dims = V_dims
        self.mu = mu
        self.edge_len_mu = E_props.edge_len * self.mu
        proj_mat = E_props.V_insertion_matrix # create_insertion_matrix(E_props.n_edges, E_props.n_component, [0, 1], device=device).to_sparse_csr()


        """ SPM TESTING """
        n_edges, n_comp = E_props.n_edges, E_props.n_component
        edge_len_mu = self.edge_len_mu.repeat(1, 2)
        visc_mat = create_selection_matrix(n_blocks=n_edges, block_size=n_comp, selected_dims=V_dims, weights=-edge_len_mu).to_sparse_csr().cuda()

        proj_visc_mat = proj_mat @ visc_mat

        self.A_visc = proj_visc_mat @ self.E_props.A_face_grad
        self.b_visc = proj_visc_mat @ self.E_props.b_face_grad
        # print(proj_visc_mat)
        # exit(8)

    def edge_fluxes(self, Us):

        # E_props = self.E_props
        # dUdn_face = E_props.grad_faces_n    # shape = [n_edges, n_component]
        """ Dense - spm"""
        # visc = self.edge_len_mu * dUdn_face             # shape = [n_edges, n_component]
        # visc = self.A @ E_props.grad_faces_n.flatten()
        # fluxes_flat = self.proj_mat @ visc#.flatten()
        # return fluxes_flat
        """ Full spm """
        # fluxes_flat = self.proj_visc_mat @ dUdn_face#.flatten()
        # return fluxes_flat

        """ Even fuller spm """
        Us_flat = Us.flatten()
        dUdn_face_flat = self.A_visc @ Us_flat + self.b_visc
        return dUdn_face_flat



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

        # self.proj_mat = create_insertion_matrix(E_props.n_edges, E_props.n_component, [rho_dim], device=device).to_sparse_csr()

    #@torch.compile()
    def edge_fluxes(self):
        """ Compute flux for each edge.
        """
        fluxes_all = torch.zeros(self.E_props.n_edges, self.E_props.n_component, device=self.device)

        #V_faces = self.E_props.Vs_faces        # shape = [n_edges, edges=2, n_comp=2]
        # normals = self.E_props.normals.unsqueeze(1)     # shape = [n_edges, 1, 2]
        # fluxes = (V_faces * normals).sum(dim=-1)        # shape = [n_edges, edges=2]
        # fluxes = fluxes.mean(dim=1)

        fluxes = self.E_props.phi.squeeze()

        fluxes_all[:, self.rho_dim] = fluxes
        fluxes_flat = fluxes_all.flatten()
        # fluxes_flat = self.proj_mat @ fluxes.flatten()

        return fluxes_flat

    # def _upwind_coef(self, V_face):
    #     E_props = self.E_props
    #
    #     # 1. Compute the upwind indicator.
    #     dot_vn = (E_props.normals_main * V_face).sum(dim=1)  # [n_edges]
    #     upwind = torch.sign(dot_vn).long() # [n_edges], values 0 or 1
    #     upwind_idx = E_props.edge_to_tri_main[self.edge_idx, upwind]  # [n_edges]
    #     # 2. Gather the upwind cell gradients.
    #     upwind_grad = E_props.cell_grads[upwind_idx, :, self.rho_dim]  # [n_edges, 2]
    #     # 3. Compute the directional derivative at the face (du/dn_face)
    #     dudn_face = E_props.grad_faces_n[self.q_main_mask]
    #     # 4. Compute the limiter. r = max( 2 * (grad_cell . d) / (|d| * du/dn_face) - 1 , 0)
    #     dot_disp_grad = (E_props.cell_disps * upwind_grad).sum(dim=1)  # [n_edges]
    #     denom = E_props.cell_dist * dudn_face + 1e-7  # [n_edges]
    #     r = torch.clamp(2 * dot_disp_grad / denom - 1, min=0)  # enforce r >= 0, [n_edges]
    #     beta = self._beta(r)
    #     # print(f'{r.max().cpu() = }')
    #     # print(torch.nonzero(r == r.max()))
    #     return beta, upwind
    #
    # def _main_fluxes(self, rho):
    #     """ Compute u_face for each edge.
    #         div(u V) = sum_i( V_f . dS_f * u_f)
    #             Linear flux interpolation
    #             Upwinding
    #
    #         Us.shape = (n_cells)
    #         V_dir.shape = (n_cells, 2)
    #
    #         Return.shape = (n_edges_main)
    #     """
    #     E_props = self.E_props
    #
    #     Us_face = E_props.U_face[~E_props.bc_edge_mask]  # shape = [n_edges, 2]
    #     V_face_lin = Us_face[:, self.V_dims]  # shape = [n_edges, 2]
    #     # rho_face_lin = Us_face[:, self.rho_dim]    # shape = [n_edges]
    #
    #     # phi = dS_face * V_face
    #     phi = (E_props.normals_main * V_face_lin).sum(-1)
    #     # # Upwinding
    #     #beta, upwind = self._upwind_coef(V_face_lin)
    #     # # 5. Corrected face value: rho_face = (1 - beta) * u_upwind + beta * u_face_lin
    #     # rho_centroid = rho[E_props.edge_to_tri_main]  # [n_edges_, 2]
    #     # TODO: TEMP TEST
    #     #rho_face_cor = (1 - beta) * rho_centroid[self.edge_idx, upwind] + beta * rho_face_lin  # [n_edges]
    #     # 6. Compute div(rhoV) = phi * u_face_cor
    #     flux_uV = phi #* rho_face_cor       # [n_edges]
    #
    #     return flux_uV
    #
    # def _bc_fluxes_(self):
    #     """ Flux = rho_face * phi
    #         phi = n_face dot V_f.
    #     """
    #     E_props = self.E_props
    #
    #     # Flux = U_face * phi
    #     # phi = normal dot Vf on face. Vf is the nearest cell value.
    #     Us_face = E_props.U_face[E_props.bc_edge_mask]      # shape = [n_bc_edges, 3]
    #     V_face_lin = Us_face[:, self.V_dims]                          # shape = [n_bc_edges, 2]
    #     # TODO: TEMP TEST
    #     #rho_face = Us_face[:, self.rho_dim] * 0 + 1      # shape = [n_bc_edges]
    #
    #     phi = (E_props.normals_bc * V_face_lin).sum(-1)
    #     bc_fluxes = phi #* rho_face
    #
    #     return bc_fluxes


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
        self.proj_mat = E_props.V_insertion_matrix

    def edge_fluxes(self):
        # fluxes = torch.zeros(self.E_props.n_edges, self.E_props.n_component, device=self.device)

        rho_faces = self.E_props.rho_faces        # shape = [n_edges, edges=2, n_comp=1]
        normals = self.E_props.normals             # shape = [n_edges, 1, 2]

        rho_faces = rho_faces.mean(dim=1)  # shape = [n_edges, 1]
        rho_n = rho_faces * normals                 # shape = [n_edges, 2, 2]

        fluxes = rho_n

        # fluxes[:, self.V_dims] = H
        # fluxes_flat = fluxes.flatten()

        fluxes_flat = self.proj_mat @ fluxes.flatten()
        return fluxes_flat


class KTDiffusion(FVMEdgeFunc):
    """ Diffusion term from K-T solver """
    E_props: FVMEdgeInfo

    def __init__(self, E_props: FVMEdgeInfo, device="cpu"):
        self.device = device
        self.E_props = E_props


    def edge_fluxes(self, momentums, a):
        # fluxes = torch.zeros(self.E_props.n_edges, self.E_props.n_component, device=self.device)

        U = self.E_props.U_face     # shape = [n_edges, 2, n_comp=3]
        fluxes = (a/2)  * (U[:, 0] - U[:, 1]) * self.E_props.edge_len  # shape = [n_edges, n_comp=3]
        fluxes_flat = fluxes.flatten()


        return fluxes_flat


class FVMEquation:
    mesh: FVMMesh
    E_props: FVMEdgeInfo
    edges: FVMEdgeFunc
    cells: FVMCells
    n_comp: int
    areas: torch.Tensor  # shape = (n_cells)
    tri_to_edge: torch.Tensor  # shape = (n_cells, 3)
    tri_edge_sign: torch.Tensor  # shape = (n_cells, 3)

    def __init__(self, mesh: FVMMesh, n_comp, bc_tag, us_init=None, device="cuda"):
        self.device = device
        self.mesh = mesh
        self.n_comp = n_comp

        E_props = FVMEdgeInfo(mesh, n_comp, bc_tag, device=device)
        self.cells = FVMCells(mesh.n_cells, n_comp, us_init, device=device)
        self.E_props = E_props

        # Cell divergence calcs
        self.tri_to_edge = mesh.tri_to_edge
        self.tri_edge_sign = mesh.tri_edge_signs.unsqueeze(-1).to(device)
        self.areas = mesh.areas.to(device)

        self.P_advect = AdvectScalar(E_props, V_dims=[0, 1], rho_dim=2, device=device)
        self.P_force = Density(E_props, p_dim=2, V_dims=[0, 1], device=device)
        self.U_advect = AdvectVector(E_props, V_dims=[0, 1], rho_dim=2, device=device)
        self.U_visc = Viscosity(E_props, mu=0.001, V_dims=[0, 1], device=device)
        self.KT_diff = KTDiffusion(E_props, device=device)

        # Matrix for converting edge fluxes to cell divergence
        self.flux_mat = self.build_flux_mat(self.tri_to_edge, -self.tri_edge_sign, mesh.n_edges, device=device)

        t_solver = ExplMidpoint(self.cells, 0.005, 9001, self)
        t_solver.solve()

    def build_flux_mat(self, tri_to_edge, tri_edge_sign, n_edges, device=None, dtype=torch.float32):
        """
        Build the incidence matrix T of shape (n_tri, n_edges).
        For each triangle i and local edge j, we set:
            T[i, tri_to_edge[i, j]] = tri_edge_sign[i, j].
        """
        # 1. Build the incidence matrix T.
        n_tri, n_local = tri_to_edge.shape  # n_local is typically 3.
        T = torch.zeros(n_tri, n_edges, device=device, dtype=dtype)
        for i in range(n_tri):
            for j in range(n_local):
                edge_idx = tri_to_edge[i, j]
                T[i, edge_idx] = tri_edge_sign[i, j]


        # 2. Build the diagonal area inverse matrix A_inv.
        A_inv = torch.diag(1.0 / self.areas)  # shape: (n_tri, n_tri)
        # 3. Combine to form D = A_inv @ T.
        D = A_inv @ T  # shape: (n_tri, n_edges)
        # 4. "Lift" D to act on the full fluxes (all components) using the Kronecker product.
        #    We want M = D ⊗ I_{n_component}, which has shape (n_tri*n_component, n_edges*n_component)
        I_comp = torch.eye(self.n_comp, device=device)
        flux_mat = torch.kron(D, I_comp)

        return flux_mat.to_sparse_csr()


    def _flux_to_div(self, fluxes):
        """ Compute cell divergence using fluxes.
            fluxes.shape = (n_edges, N_component)

            du/dt = -div(flux) = -sum_i (sign_i * flux_i)
        """
        # Vectorised version
        # fluxes = fluxes.view(-1, self.n_comp)  # shape: (n_edges, n_component)
        # tri_fluxes = fluxes[self.tri_to_edge]  # shape: [n_cells, 3, n_component]
        # divergence = torch.sum(-self.tri_edge_sign * tri_fluxes, dim=1).squeeze() / self.areas.unsqueeze(-1)     # shape = [n_cells, N_component]

        # Matrix version
        fluxes_flat = fluxes    # shape: (n_edges * n_component,)
        divergence_flat = torch.mv(self.flux_mat, fluxes_flat)  # shape: (n_cells * n_component,)
        divergence = divergence_flat.view(-1, self.n_comp)  # shape: (n_cells, n_component)
        return divergence

    #@torch.compile()
    def forward(self, primatives, momentum, i=None):
        """ primatives.shape = (n_cells, n_component) """

        E_props = self.E_props

        E_props.precompute_shared(primatives)


        fluxes = self.P_force.edge_fluxes()
        fluxes += self.U_advect.edge_fluxes()
        fluxes += self.P_advect.edge_fluxes()
        fluxes += self.U_visc.edge_fluxes(primatives)
        fluxes += self.KT_diff.edge_fluxes(primatives, 1)
        divergence = self._flux_to_div(fluxes)

        return divergence

    def plot_flux(self, fluxes, title="Fluxes", show_index=False):
        plot_edges(self.mesh.vertices.cpu(), self.mesh.edges.cpu(), title=title, color=fluxes, show_index=show_index)


    def plot_cells(self, values, title="Cell Values", convert=False, show_index=False):
        if convert:
            momentum_x, momentum_y, density = values[:, 0], values[:, 1], values[:, 2]
            u_x, u_y = momentum_x, momentum_y
            #u_x, u_y = momentum_x / density, momentum_y / density
            density = (density - 1)
            values = torch.stack([u_x, u_y, density], dim=1)
        plot_points(self.mesh.centroids.cpu(), values.T, show_index=show_index, title=title)#, lims=[-0.01, 0.01])



