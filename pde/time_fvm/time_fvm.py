import torch
from abc import ABC, abstractmethod
from codetiming import Timer
from cprint import c_print

from pde.graph_grid.graph_utils import plot_points, plot_interp_graph, plot_edges, plot_interp
from pde.time_fvm.fvm_mesh import FVMMesh
from pde.time_fvm.edge_process import FVMEdgeInfo, create_selection_matrix, invert_selection_matrix
from pde.time_fvm.t_solvers import FVMCells, Euler, ExplMidpoint, Heuns


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
        pass


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
    """ Viscous forces: div(mu grad(u)) = sum_f grad(u) * mu_f * l_f"""
    E_props: FVMEdgeInfo
    V_dims: list[int]
    def __init__(self, E_props: FVMEdgeInfo, V_dims, mu=0.01, device="cpu"):
        self.device = device
        self.E_props = E_props

        self.V_dims = V_dims
        self.mu = mu
        edge_len_mu = E_props.edge_len * self.mu
        proj_mat = E_props.V_insertion_matrix # create_insertion_matrix(E_props.n_edges, E_props.n_component, [0, 1], device=device).to_sparse_csr()


        """ SPM TESTING """
        n_edges, n_comp = E_props.n_edges, E_props.n_component
        edge_len_mu = edge_len_mu.repeat(1, 2)
        visc_mat = create_selection_matrix(n_blocks=n_edges, block_size=n_comp, selected_dims=V_dims, weights=-edge_len_mu).to_sparse_csr().cuda()

        proj_visc_mat = proj_mat @ visc_mat

        self.A_visc = proj_visc_mat @ self.E_props.A_face_grad
        self.b_visc = proj_visc_mat @ self.E_props.b_face_grad

        del self.E_props.A_face_grad, self.E_props.b_face_grad

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


class AdvectDensity(Advect):
    """ div(rho V).
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

        # V_faces = self.E_props.Vs_faces        # shape = [n_edges, edges=2, n_comp=2]
        # rho_faces = self.E_props.rho_faces     # shape = [n_edges, edges=2, n_comp=1]
        # normals = self.E_props.normals.unsqueeze(1)     # shape = [n_edges, 1, 2]
        #
        # flux = (rho_faces * V_faces * normals).sum(dim=-1)        # shape = [n_edges, edges=2]
        # flux = flux.mean(dim=1)

        V_dot_l = self.E_props.phi                  # shape = [n_edges, 1]
        rho_face = self.E_props.rho_faces.squeeze()               # shape = [n_edges, edges=2]

        flux = V_dot_l * rho_face               # shape = [n_edges, edges=2]
        flux = flux.mean(dim=1)                 # shape = [n_edges]

        fluxes_all[:, self.rho_dim] = flux
        fluxes_flat = fluxes_all.flatten()
        # fluxes_flat = self.proj_mat @ fluxes.flatten()

        return fluxes_flat


class PressureForce(FVMEdgeFunc):
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


    def edge_fluxes(self, a):
        # fluxes = torch.zeros(self.E_props.n_edges, self.E_props.n_component, device=self.device)

        rho = self.E_props.rho_faces
        Vs = self.E_props.Vs_faces      # shape = [n_edges, edges=2, n_comp=2]

        momentum = Vs * rho      # shape = [n_edges, 2, 2]
        As = torch.cat([momentum, rho], dim=2)  # shape = [n_edges, 2, 3]

        # Wavespeed is a + v_max
        Vs = Vs.norm(dim=-1)            # shape = [n_edges, edges=2]
        Vs_max = Vs.max(dim=1, keepdim=True).values   # shape = [n_edges, 1]
        a = a + Vs_max                  # shape = [n_edges, 1]
        fluxes = (a/2)  * (As[:, 0] - As[:, 1]) * self.E_props.edge_len  # shape = [n_edges, n_comp=3]
        fluxes_flat = fluxes.flatten()


        return fluxes_flat


class FVMEquation:
    mesh: FVMMesh
    E_props: FVMEdgeInfo
    edges: FVMEdgeFunc
    cells: FVMCells
    n_comp: int
    areas: torch.Tensor  # shape = (n_cells)
    # tri_to_edge: torch.Tensor  # shape = (n_cells, 3)
    # tri_edge_sign: torch.Tensor  # shape = (n_cells, 3)

    def __init__(self, mesh: FVMMesh, n_comp, bc_tag, us_init=None, device="cuda"):
        self.device = device
        self.mesh = mesh
        self.n_comp = n_comp

        E_props = FVMEdgeInfo(mesh, n_comp, bc_tag, device=device)
        self.cells = FVMCells(mesh.n_cells, n_comp, us_init, device=device)
        self.E_props = E_props

        # Cell divergence calcs
        tri_to_edge = mesh.tri_to_edge
        tri_edge_sign = mesh.tri_edge_signs.unsqueeze(-1) # .to(device)
        self.areas = mesh.areas

        self.rho_advect = AdvectDensity(E_props, V_dims=[0, 1], rho_dim=2, device=device)
        self.P_force = PressureForce(E_props, V_dims=[0, 1], p_dim=2, device=device)
        self.U_advect = AdvectVector(E_props, V_dims=[0, 1], rho_dim=2, device=device)
        self.U_visc = Viscosity(E_props, mu=0.00001, V_dims=[0, 1], device=device)
        self.KT_diff = KTDiffusion(E_props, device=device)

        # Matrix for converting edge fluxes to cell divergence
        self.flux_mat = self.build_flux_mat(tri_to_edge, -tri_edge_sign, mesh.n_edges)

        self.t_solver = Euler(self.cells, 0.0025, 20001, self)

        del self.areas


    def solve(self):
        self.t_solver.solve()

    def build_flux_mat(self, tri_to_edge, tri_edge_sign, n_edges,  dtype=torch.float32):
        """
        Build the incidence matrix T of shape (n_tri, n_edges).
        For each triangle i and local edge j, we set:
            T[i, tri_to_edge[i, j]] = tri_edge_sign[i, j].
        """
        c_print("Constricting flux matrix", color="bright_green")
        # # 1. Build the incidence matrix T.
        # n_tri, n_local = tri_to_edge.shape  # n_local is typically 3.
        # T = torch.zeros(n_tri, n_edges, device="cpu", dtype=dtype)
        # for i in range(n_tri):
        #     for j in range(n_local):
        #         edge_idx = tri_to_edge[i, j]
        #         T[i, edge_idx] = tri_edge_sign[i, j]
        # # 2. Build the diagonal area inverse matrix A_inv.
        # A_inv = torch.diag(1.0 / self.areas.cpu())  # shape: (n_tri, n_tri)
        # # 3. Combine to form D = A_inv @ T.
        # D = A_inv @ T  # shape: (n_tri, n_edges)
        # del T

        # I_comp = torch.eye(self.n_comp, device="cpu")
        # flux_mat = torch.kron(D.to_dense(), I_comp)
        # Assume tri_to_edge and tri_edge_sign are torch tensors of shape (n_tri, n_local),
        # and that self.areas is a tensor of length n_tri.
        n_tri, n_local = tri_to_edge.shape  # typically, n_local == 3

        # Create row indices: each triangle i contributes n_local entries.
        row_indices = torch.arange(n_tri).unsqueeze(1).expand(n_tri, n_local).reshape(-1)

        # Flatten the edge indices from tri_to_edge for column indices.
        col_indices = tri_to_edge.reshape(-1)
        # Flatten the sign values from tri_edge_sign.
        values = tri_edge_sign.reshape(-1).to(dtype)
        # Compute the inverse areas (A_inv is diagonal) and scale the nonzero values.
        areas_inv = (1.0 / self.areas.cpu()).to(dtype)
        D_values = values * areas_inv[row_indices]
        # Stack row and column indices for the sparse tensor.
        D_indices = torch.stack([row_indices, col_indices])

        # "Lift" D to act on the full fluxes (all components) using the Kronecker product.
        #    We want M = D ⊗ I_{n_component}, which has shape (n_tri*n_component, n_edges*n_component)
        comp_ids = torch.arange(self.n_comp, device="cpu")  # shape: (n_comp,)
        new_rows = D_indices[0].unsqueeze(1) * self.n_comp + comp_ids.unsqueeze(0)  # shape: (nnz, n_comp)
        new_cols = D_indices[1].unsqueeze(1) * self.n_comp + comp_ids.unsqueeze(0)  # shape: (nnz, n_comp)

        # Flatten the new indices.
        new_rows = new_rows.reshape(-1)
        new_cols = new_cols.reshape(-1)
        flux_indices = torch.stack([new_rows, new_cols], dim=0)

        # The values are just the original ones repeated for each component.
        flux_values = D_values.unsqueeze(1).expand(-1, self.n_comp).reshape(-1)

        # Define the shape of the lifted matrix:
        flux_shape = (n_tri * self.n_comp, n_edges * self.n_comp)

        # Construct the sparse flux matrix.
        flux_mat = torch.sparse_coo_tensor(flux_indices, flux_values, size=flux_shape, device="cpu", dtype=dtype).coalesce().cuda().to_sparse_csr()

        return flux_mat


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
    def forward(self, primatives, momentum, t=0):
        """ primatives.shape = (n_cells, n_component) """

        E_props = self.E_props

        E_props.precompute_shared(primatives)

        # d(rho_u)/dt
        fluxes = self.P_force.edge_fluxes()
        fluxes += self.U_advect.edge_fluxes()
        fluxes += self.U_visc.edge_fluxes(primatives)

        # d(rho)/dt
        fluxes += self.rho_advect.edge_fluxes()

        # MUSCL term
        fluxes += self.KT_diff.edge_fluxes(1)

        divergence = self._flux_to_div(fluxes)

        return divergence

    def plot_flux(self, fluxes, title="Fluxes", show_index=False, lims=None):
        plot_edges(self.mesh.vertices.cpu(), self.mesh.edges.cpu(), title=title, color=fluxes, show_index=show_index, lims=lims)

    def plot_cells(self, values, title="Cell Values", show_index=False, lims=None):
        plot_points(self.mesh.centroids.cpu(), values.T, show_index=show_index, title=title, lims=lims)

    def plot_interp(self, values, title="Cell Values", lims=None, resolution=1000):
        plot_interp(self.mesh.vertices, values.T, self.mesh.triangles, title=title, lims=lims, resolution=resolution)

        # Example vertex coordinates (arrays of x and y positions)
        # import numpy as np
        # import matplotlib.pyplot as plt
        # import matplotlib.tri as tri
        #
        # Xs = self.mesh.vertices.cpu().numpy()
        # triangles = self.mesh.triangles.cpu().numpy()
        #
        # x, y = Xs[:, 0], Xs[:, 1]
        #
        # # x = np.array([0, 1, 2, 0.5, 1.5])
        # # y = np.array([0, 0, 0, 1, 1])
        #
        # # Example triangles defined by indices into the x, y arrays.
        # # Each row is a triangle (three vertex indices)
        # # triangles = np.array([
        # #     [0, 1, 3],
        # #     [1, 4, 3],
        # #     [1, 2, 4]
        # # ])
        #
        # # Example values at the vertices (for coloring the mesh)
        # # z_faces = np.array([0.2, 0.8, 0.5])
        # z_faces = values[:, 0].cpu().numpy()#np.random.randn(triangles.shape[0])
        # print(f'{values.shape = }, {z_faces.shape}')
        # # Create the Triangulation object
        # mesh = tri.Triangulation(x, y, triangles)
        #
        # plt.figure(figsize=(8, 6))
        #
        # # Optionally, plot the triangle edges to visualize the mesh structure
        # plt.triplot(mesh, color='black', lw=0.8)
        #
        # # Plot the mesh with colors defined by the face values.
        # # Because z_faces has one value per triangle, each triangle gets a uniform color.
        # plt.tripcolor(mesh, facecolors=z_faces, edgecolors='none', cmap='viridis', shading='flat')
        #
        # # Add a colorbar to show the mapping from z-values to colors
        # plt.colorbar()
        #
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.title('2D Mesh Plot with Face-based Values')
        # plt.show()
        # exit(7)
