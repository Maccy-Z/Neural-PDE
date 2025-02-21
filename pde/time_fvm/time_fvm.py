import torch
from abc import ABC, abstractmethod
from codetiming import Timer

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
        d_dot_gradU = (E_props.cell_disps.unsqueeze(-1) * upwind_grad).sum(dim=1)  # [n_edges, 2]
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

        self.proj_mat = create_insertion_matrix(E_props.n_edges, E_props.n_component, [rho_dim], device=device).to_sparse_csr()

    def edge_fluxes(self, Us):
        """ Compute flux for each edge.
            us.shape = (n_cells, n_component)
        """

        #fluxes_all = torch.zeros(self.E_props.n_edges, self.E_props.n_component, device=self.device)

        V_faces = self.E_props.U_face[:, :, self.V_dims]        # shape = [n_edges, edges=2, n_comp=2]
        rho_faces = self.E_props.U_face[:, :, self.rho_dim]        # shape = [n_edges, edges=2, dims=1]

        face_diffs = rho_faces[:, 0] - rho_faces[:, 1]  # shape = [n_edges, n_comp=1]
        normals = self.E_props.normals.unsqueeze(1)
        fluxes = (V_faces * normals).sum(dim=-1)        # shape = [n_edges, edges=2]
        fluxes = fluxes.mean(dim=1) - face_diffs/2 # shape = [n_edges]
        #fluxes_all[:, self.rho_dim] = fluxes

        fluxes_all = self.proj_mat @ fluxes.flatten()

        return fluxes_all#.flatten()

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
        self.proj_mat = create_insertion_matrix(E_props.n_edges, E_props.n_component, V_dims, device=device).to_sparse_csr()



    def edge_fluxes(self):
        #fluxes = torch.zeros(self.E_props.n_edges, self.E_props.n_component, device=self.device)

        V_faces = self.E_props.U_face[:, :, self.V_dims]        # shape = [n_edges, edges=2, n_comp=2]
        rho_faces = self.E_props.U_face[:, :, self.p_dim].unsqueeze(dim=-1)        # shape = [n_edges, edges=2, dims=1]

        normals = self.E_props.normals.unsqueeze(1)             # shape = [n_edges, 1, 2]
        rho_n = rho_faces * normals
        rho_n = rho_n.mean(dim=1)  # shape = [n_edges, 2]

        face_diffs = V_faces[:, 0] - V_faces[:, 1]  # shape = [n_edges, n_comp=2]
        H = rho_n - face_diffs/2
        # fluxes[:, self.V_dims] = H
        # fluxes_flat = fluxes.flatten()

        fluxes_flat = self.proj_mat @ H.flatten()
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
        #self.U_advect = AdvectVector(E_props, V_dims=[0, 1], rho_dim=2, device=device)
        #self.U_visc = Viscosity(E_props, mu=0.0, dims=[0, 1], device=device)

        # 1. Build the incidence matrix T.
        T = self.build_triangle_edge_incidence(self.tri_to_edge, -self.tri_edge_sign, mesh.n_edges, device=device)
        # 2. Build the diagonal area inverse matrix A_inv.
        A_inv = torch.diag(1.0 / self.areas)  # shape: (n_tri, n_tri)
        # 3. Combine to form D = A_inv @ T.
        D = A_inv @ T  # shape: (n_tri, n_edges)
        # 4. "Lift" D to act on the full fluxes (all components) using the Kronecker product.
        #    We want M = D ⊗ I_{n_component}, which has shape (n_tri*n_component, n_edges*n_component)
        I_comp = torch.eye(n_comp, device=device)
        self.flux_mat = torch.kron(D, I_comp)

        t_solver = ExplMidpoint(self.cells, 0.001, 9001, self)
        t_solver.solve()

    def build_triangle_edge_incidence(self, tri_to_edge, tri_edge_sign, n_edges, device=None, dtype=torch.float32):
        """
        Build the incidence matrix T of shape (n_tri, n_edges).
        For each triangle i and local edge j, we set:
            T[i, tri_to_edge[i, j]] = tri_edge_sign[i, j].
        """
        n_tri, n_local = tri_to_edge.shape  # n_local is typically 3.
        T = torch.zeros(n_tri, n_edges, device=device, dtype=dtype)
        for i in range(n_tri):
            for j in range(n_local):
                edge_idx = tri_to_edge[i, j]
                T[i, edge_idx] = tri_edge_sign[i, j]
        return T


    def _flux_to_div(self, fluxes):
        """ Compute cell divergence using fluxes.
            fluxes.shape = (n_edges, N_component)

            du/dt = -div(flux) = -sum_i (sign_i * flux_i)
        """
        # Vectorised version
        # tri_fluxes = fluxes[self.tri_to_edge]  # shape: [n_cells, 3, n_component]
        # divergence = torch.sum(self.tri_edge_sign * tri_fluxes, dim=1).squeeze() / self.areas.unsqueeze(-1)     # shape = [n_cells, N_component]

        fluxes_flat = fluxes    # shape: (n_edges * n_component,)
        divergence_flat = self.flux_mat @ fluxes_flat  # shape: (n_cells * n_component,)
        divergence = divergence_flat.view(-1, self.n_comp)  # shape: (n_cells, n_component)

        return divergence


    def forward(self, primatives, momentum, i=None):
        """ primatives.shape = (n_cells, n_component) """

        E_props = self.E_props

        E_props.precompute_shared(primatives)

        fluxes = self.P_advect.edge_fluxes(primatives)

        # fluxes += self.U_advect.edge_fluxes(momentum)
        # fluxes += self.U_visc.edge_fluxes()
        fluxes += self.P_force.edge_fluxes()

        divergence = self._flux_to_div(fluxes)


        return divergence

    def plot_flux(self, fluxes, title="Fluxes", convert=False, show_index=True):
        plot_edges(self.mesh.vertices.cpu(), self.mesh.edges.cpu(), title=title, color=fluxes, show_index=show_index)

    def plot_cells(self, values, title="Cell Values", convert=False):
        if convert:
            momentum_x, momentum_y, density = values[:, 0], values[:, 1], values[:, 2]
            u_x, u_y = momentum_x, momentum_y
            #u_x, u_y = momentum_x / density, momentum_y / density
            density = (density - 1)
            values = torch.stack([u_x, u_y, density], dim=1)
        plot_points(self.mesh.centroids.cpu(), values.T, show_index=True, title=title)#, lims=[-0.01, 0.01])


