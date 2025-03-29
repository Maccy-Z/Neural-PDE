import torch
from abc import ABC, abstractmethod
from codetiming import Timer
from cprint import c_print
import math

from pde.graph_grid.graph_utils import plot_points, plot_interp_graph, plot_edges, plot_interp
from pde.time_fvm.fvm_mesh import FVMMesh
from pde.time_fvm.edge_process import FVMEdgeInfo, create_selection_matrix
from pde.time_fvm.t_solvers import FVMCells
from pde.time_fvm.integrators import Euler, Adams2, RK2_SSP, RK3_SSP4, Adams3PC, Butcher, Adams4PC, Heuns, ExplMidpoint, Heuns, RK3_SSP, RK2_SSP3, RK2_SSP4
from pde.time_fvm.config_fvm import ConfigFVM
from pde.time_fvm.sparse_utils import SparseReshapeMM

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

    #@abstractmethod
    def edge_fluxes(self, Us, *args):
        """ Compute flux for each edge
        """
        pass


class Adevction(FVMEdgeFunc):
    """ out_i = div(rho V V_i) for velocity V, i = {x, y}
        dims: Which dimensions of Us are advected.
    """
    E_props: FVMEdgeInfo
    V_dims: list[int]
    rho_dim: int
    M_holder: SparseReshapeMM

    def __init__(self, E_props: FVMEdgeInfo, flux_mat: torch.Tensor, V_dims: list[int], rho_dim:int, device="cpu"):
        self.device = device
        self.E_props = E_props
        self.flux_mat = flux_mat

        self.V_dims = V_dims
        self.rho_dim = rho_dim

        #self._build_jacobian(diag=True)

    # @torch.compile()
    def edge_fluxes(self):
        """ rho * U @ V.T @ n = rho V * phi
            f_x = rho V_x * phi
            f_y = rho V_y * phi
            f_rho = rho * phi
        """
        E_props = self.E_props
        rho_faces = E_props.rho_faces # shape = [n_edges, edges=2, n_comp=1]
        # V_faces = E_props.Vs_faces  # shape = [n_edges, edges=2, n_comp=2]
        phi = E_props.phi           # Linear interpolation of convection vector = (v_faces dot normal). shape = [n_edges, edges=2]
        mom_f = E_props.mom_faces
        # mom_f = rho_faces * V_faces         # shape = [n_edges, edges=2, n_comp=2]

        Us_f = torch.cat([mom_f, rho_faces], dim=-1)  # shape = [n_edges, edges=2, n_comp=3]
        advec_flux = Us_f * phi.unsqueeze(-1)           # shape = [n_edges, edges=2, n_comp=3]
        advec_flux = advec_flux.mean(dim=1)              # shape = [n_edges, n_comp=3]
        advec_flux = advec_flux.flatten()

        # Us_f = Us_f.permute(0, 2, 1).reshape(-1, 2)        # shape = [n_comp*n_edges, edges=2]
        # phi_f = phi.repeat_interleave(3, dim=0)     # shape = [3*n_edges, edges=2]

        # advec_flux = Us_f * phi_f               # shape = [3*n_edges, edges=2]
        # advec_flux = advec_flux.mean(dim=-1)        # shape = [3*n_edges]

        return advec_flux


    def _build_F_A(self, A):
        """ Construct matrix M_lik = F_li @ A_ik for each edge i. Done in a way ready for reshaping.
            F is the flux matrix, A is the gradient matrix.

            A.shape = [3*n_edges, 3*n_cells]
            F.shape = [3*n_cells, 3*n_edges]

            return.shape = [3*n_cells * 3*n_cells, 3*n_edges]
        """
        E_props = self.E_props
        n_edges, n_cells = E_props.n_edges, E_props.n_cells
        F = self.flux_mat.to_sparse()#.cpu()

        F_indices = F._indices()  # shape: (2, nnz_F): [l_indices; i_indices]
        F_values = F._values()  # shape: (nnz_F,)
        A_indices = A._indices()  # shape: (2, nnz_G): [i_indices; k_indices]
        A_values = A._values()  # shape: (nnz_G,)

        # # Number of nonzero elements:
        # nnz_F = F_indices.shape[1]
        # nnz_G = A_indices.shape[1]

        # Broadcast F_indices[1] (i indices) and G0_indices[0] (i indices) to compare:
        f_i = F_indices[1].unsqueeze(1)  # shape: (nnz_F, 1)
        g_i = A_indices[0].unsqueeze(0)  # shape: (1, nnz_G)
        match_mask = (f_i == g_i)  # shape: (nnz_F, nnz_G)

        # Get the indices of the matching pairs. Do on CPU since GPU has size limit
        match_mask = match_mask.cpu()
        f_match_idx, g_match_idx = torch.nonzero(match_mask, as_tuple=True)
        f_match_idx, g_match_idx = f_match_idx.to(self.device), g_match_idx.to(self.device)

        # For each matching pair, get:
        # - l from F (row index of F)
        # - i from F (column index of F, also matching the row index of G0)
        # - k from G0 (column index of G0)
        # - corresponding values f_val and g_val
        l_vals = F_indices[0][f_match_idx]  # shape: (num_matches,)
        i_vals = F_indices[1][f_match_idx]  # shape: (num_matches,)
        f_vals = F_values[f_match_idx]  # shape: (num_matches,)
        k_vals = A_indices[1][g_match_idx]  # shape: (num_matches,)
        g_vals = A_values[g_match_idx]  # shape: (num_matches,)

        # Now compute the flattened row indices for M:
        # Here, we assume the flattened row is defined as: new_row = l * (3*n_cells) + k
        new_rows = l_vals * (3 * n_cells) + k_vals  # shape: (num_matches,)
        new_cols = i_vals  # shape: (num_matches,)
        new_vals = f_vals * g_vals  # shape: (num_matches,)

        # Build the sparse matrix M (of shape (L*K, i), here K is 3*n_cells)
        M_shape = ((3*n_cells )* (3 * n_cells), 3*n_edges)
        M_indices = torch.stack([new_rows, new_cols])
        M = torch.sparse_coo_tensor(M_indices, new_vals, M_shape)

        return M


    def _build_jacobian(self, diag):
        """
            U_cell = [interleave(mom_x | mom_y | rho)], shape = [3*n_cell]
            U_face = A(U_cell)              shape = [3*n_edges, 2], second dim is Left / Right side of face.
            d(U_f_i, L/R)/d(U_c_j) = :
                                0 if cell_j doesnt have face_i  - Or computing wrong mom_x, mom_y, rho component.
                                1 if face_i is on cell_j and cell_j is on L/R side.
            For jacobian:
                advec = sum_j 1/2 Uf_ij phi_ij
                dadvec/dUc_k = 1/2 sum_j phi_ij dUf_ij/dUc_k
                             = 1/2 sum_j phi_ij J_ijk
                             -> A_L @ phi_iL | A_L @ phi_iR
            Divergence:
                D = F @ advec
                dD/dUc_k = F @ dadvec/dUc_k
                         = F @ A_L @ phi_iL | F @ A_L @ phi_iR
                         = M @ [phi_iL | phi_iR]
        """
        E_props = self.E_props
        n_edges, n_cells = E_props.n_edges, E_props.n_cells
        A_left, A_right = self.E_props.dUf_dUc

        """ Method 2: Sparse matrix precompute """
        M_L = self._build_F_A(A_left)
        M_R = self._build_F_A(A_right)
        # Join M = [M_L | M_R]
        M_R_indices = M_R._indices().clone()
        M_R_indices[1:] += 3*n_edges
        M_indices = torch.cat([M_L._indices(), M_R_indices], dim=1)
        new_values = torch.cat([M_L._values(), M_R._values()]) / 2
        new_shape = (3*n_cells*3*n_cells, 6*n_edges)
        M = torch.sparse_coo_tensor(M_indices, new_values, new_shape).coalesce()

        bc_cells = self.E_props.edge_to_tri_bc
        bc_cells = torch.cat([3*bc_cells, 3*bc_cells+1, 3 * bc_cells + 2])
        self.M_holder = SparseReshapeMM(M, 3 * n_cells, diag=diag, device=self.device, zero_rows=bc_cells)


    def calc_jacobian(self):
        E_props = self.E_props
        phi = E_props.phi
        phi_f = phi.repeat_interleave(3, dim=0)  # shape = [3*n_edges, edges=2]
        phi_vec = torch.cat([phi_f[:, 0], phi_f[:, 1]])  # shhape = [6*n_edges]

        # New method
        Jac = self.M_holder.multiply(phi_vec)

        return Jac


class Viscosity(FVMEdgeFunc):
    """ Viscous forces:
            Shear viscosity: div(mu grad(V)) = sum_f grad(V) * mu_f * l_f
            Bulk viscosity: k * grad(div(V))
    """
    E_props: FVMEdgeInfo
    V_dims: list[int]
    def __init__(self, E_props: FVMEdgeInfo, areas, flux_mat, V_dims, cfg: ConfigFVM, device="cpu"):
        self.device = device
        self.E_props = E_props
        self.V_dims = V_dims


        self.mu = cfg.viscosity
        self.mu_b = cfg.visc_bulk


        """ Combine sparse matrices for viscosity: 
                project to velocity component @ viscosity @ face gradient         
        """
        edge_len_mu = E_props.edge_len * self.mu
        proj_mat = E_props.V_insertion_matrix # create_insertion_matrix(E_props.n_edges, E_props.n_component, [0, 1], device=device).to_sparse_csr()

        n_edges, n_comp = E_props.n_edges, E_props.n_component
        edge_len_mu = edge_len_mu.repeat(1, 2)
        visc_mat = create_selection_matrix(n_blocks=n_edges, block_size=n_comp, selected_dims=V_dims, weights=-edge_len_mu).to_sparse_csr().cuda()
        proj_visc_mat = proj_mat @ visc_mat
        self.A_visc = flux_mat @ proj_visc_mat @ self.E_props.A_face_grad
        self.b_visc = flux_mat @ proj_visc_mat @ self.E_props.b_face_grad
        # self.b_visc = self.b_visc.view(-1, 1)

        """ Combine sparse matrix for bulk viscosity """
        idx = torch.arange(n_edges, device=device)
        # Compute the row and column indices for non-zero entries.
        rows = torch.cat([2 * idx, 2 * idx + 1])
        cols = torch.cat([idx, idx])
        indices = torch.stack([rows, cols])  # Shape: [2, 2*n_edges]
        # Compute the corresponding values from the normals
        values = torch.cat([E_props.normals[:, 0], E_props.normals[:, 1]])
        M = torch.sparse_coo_tensor(indices, values, size=(2 * n_edges, n_edges), device=device).to_sparse_csr()

        self.A_visc_bulk = - self.mu_b * flux_mat @ proj_mat @ M

        # Clamp viscosity to k * A / dt
        self.clip_val = cfg.bulk_visc_lim * areas.to(device) / cfg.dt


    #@torch.compile()
    def divergence(self, Us):
        """ Viscosity is limited after computing divergence for stability """
        E_props = self.E_props

        """ Even fuller spm """
        Us_flat = Us.flatten()
        # div_visc = self.A_visc @ Us_flat + self.b_visc
        div_visc = torch.addmv(self.b_visc, self.A_visc, Us_flat)
        div_visc = div_visc.view(-1, 3)         # shape = [n_cells, 3]

        """ Bulk viscosity: gradient = dux/dx + duy/dy, least squares gradient. Use this to compute cell divergence. """
        if self.mu_b > 0:
            div_V_edge = E_props.div_V_faces.mean(dim=1)

            # flux_bulk = div_u_edge * E_props.normals        # shape = [n_edges, 2]
            div_bulk = torch.mv(self.A_visc_bulk, div_V_edge)
            div_bulk = div_bulk.view(-1, 3)         # shape = [n_cells, 3]
            bulk_norm = div_bulk.norm(dim=-1)

            clip_val = self.clip_val
            div_limit = torch.where(bulk_norm > clip_val, clip_val/bulk_norm, 1).unsqueeze(-1)
            div_bulk = div_bulk * div_limit
        else:
            div_bulk = 0
        return div_visc + div_bulk


class PressureForce(FVMEdgeFunc):
    """ Special case. N
        grad(rho) = div(rho I) """
    E_props: FVMEdgeInfo
    p_dim: int
    V_dims: list[int]

    def __init__(self, E_props: FVMEdgeInfo, flux_mat, c2, p_dim: int, V_dims: list[int], device="cpu"):
        self.device = device
        self.E_props = E_props
        self.c2 = c2

        self.p_dim = p_dim
        self.V_dims = V_dims
        self.proj_mat = E_props.V_insertion_matrix
        self.flux_mat = flux_mat


    #     #self._build_jacobian(zero_bc=True, diag_only=True)
    #
    # def _build_jacobian(self, zero_bc, diag_only):
    #     E_props = self.E_props
    #     normals = self.E_props.normals.squeeze()             # shape = [n_edges, 2]
    #
    #     dUfL_dUc, dUfR_duC = E_props.dUf_dUc
    #     dUfm_dUc = 1/2 * (dUfL_dUc + dUfR_duC)
    #
    #     dUfm_dUc = dUfm_dUc.coalesce()
    #
    #     # Compute J = dGf_dUc
    #     J_cols, J_rows, J_vals = [], [], []
    #     n_edges, n_cells = E_props.n_edges, E_props.n_cells
    #     for i in range(3*n_edges):
    #         i_hat = i // 3
    #         if i % 3 == 2:      # Pressure only affects Vx and Vy componenets
    #             continue
    #
    #         dUf_row_dUcj = dUfm_dUc[3 * i_hat + 2].coalesce()
    #
    #         cols = dUf_row_dUcj.indices()[0]
    #         rows = torch.ones_like(cols) * i
    #
    #         J_cols.append(cols)
    #         J_rows.append(rows)
    #         if i % 3 == 0:      # x component
    #             val = normals[i_hat, 0] * dUf_row_dUcj.values()
    #             J_vals.append(val)
    #         if i % 3 == 1:
    #             val = normals[i_hat, 1] * dUf_row_dUcj.values()
    #             J_vals.append(val)
    #
    #     J_cols = torch.cat(J_cols)
    #     J_rows = torch.cat(J_rows)
    #     new_indices = torch.stack([J_rows, J_cols], dim=0)
    #     new_values = torch.cat(J_vals)
    #     dGf_dUc = torch.sparse_coo_tensor(new_indices, new_values, size=[3*n_edges, 3*n_cells]).to(self.device)
    #     J = self.flux_mat.to_sparse_coo() @ dGf_dUc
    #
    #     # Zero out boundary cells
    #     if zero_bc:
    #         bc_cells = self.E_props.edge_to_tri_bc
    #         bc_cells = torch.cat([3*bc_cells, 3*bc_cells+1, 3 * bc_cells + 2])
    #         p_jac_idx, p_jac_values = J.indices(), J.values()
    #         rows = p_jac_idx[0]
    #         bc_mask = torch.isin(rows, bc_cells)
    #
    #         p_jac_idx = p_jac_idx[:, ~bc_mask]
    #         p_jac_values = p_jac_values[~bc_mask]
    #         J = torch.sparse_coo_tensor(p_jac_idx, p_jac_values, size=J.shape).coalesce()
    #
    #     # Only keep diagonal elements
    #     if diag_only:
    #         p_jac_idx, p_jac_values = J.indices(), J.values()
    #
    #         rows, cols = p_jac_idx[0], p_jac_idx[1]
    #         diag_mask = (rows == cols)
    #
    #         p_jac_idx = p_jac_idx[:, diag_mask]
    #         p_jac_values = p_jac_values[diag_mask]
    #         J = torch.sparse_coo_tensor(p_jac_idx, p_jac_values, size=J.shape)
    #
    #     self.Jacobian = J.coalesce()
    #
    #     # print()

    def edge_fluxes(self):

        rho_faces = self.E_props.rho_faces        # shape = [n_edges, edges=2, n_comp=1]
        normals = self.E_props.normals             # shape = [n_edges, 2]

        rho_faces = rho_faces.mean(dim=1)  # shape = [n_edges, 1]
        rho_n = self.c2 * rho_faces * normals                 # shape = [n_edges, 2]


        # fluxes = torch.zeros(self.E_props.n_edges, self.E_props.n_component, device=self.device)
        # fluxes[:, :2] =rho_n
        # fluxes_flat = fluxes.flatten()

        # fluxes_flat = torch.zeros(self.E_props.n_edges * self.E_props.n_component, device=self.device)
        # fluxes_flat[::3] = rho_n[:, 0]
        # fluxes_flat[1::3] = rho_n[:, 1]

        # fluxes = torch.cat([rho_n, torch.zeros((self.E_props.n_edges, 1), device=self.device)] , dim=1)  # shape = [n_edges, 3]
        # fluxes_flat = fluxes.flatten()

        fluxes_flat = self.proj_mat @ rho_n.flatten()
        return fluxes_flat


class KTDiffusion(FVMEdgeFunc):
    """ Diffusion term from K-T solver """
    E_props: FVMEdgeInfo

    def __init__(self, v_factor, E_props: FVMEdgeInfo, device="cpu"):
        self.device = device
        self.v_factor = v_factor
        self.E_props = E_props

    @torch.compile()
    def edge_fluxes(self, c):
        rho_face = self.E_props.rho_faces
        Vs_face = self.E_props.Vs_faces      # shape = [n_edges, edges=2, n_comp=2]
        mom_face = self.E_props.mom_faces

        Us = torch.cat([mom_face, rho_face], dim=2)  # shape = [n_edges, 2, 3]

        # Wavespeed is c + v_max. Clip velocity wavespeed to k*c + v_max
        Vs = Vs_face.norm(dim=-1)            # shape = [n_edges, edges=2]
        Vs_max = Vs.max(dim=1, keepdim=True).values   # shape = [n_edges, 1]

        # a = torch.empty((self.E_props.n_edges, 3), device=self.device)
        # a[:, :2] = self.v_factor * c       # Velocity speed
        # # a[:, :2] = c       # Velocity speed
        # a[:, 2] = c                        # Pressure speed
        # a += Vs_max
        # a = a / 2

        a = torch.tensor([[self.v_factor * c, self.v_factor * c, c]], device=self.device)
        a = a.repeat(self.E_props.n_edges, 1) + Vs_max  # shape = [n_edges, 3]

        fluxes = (a/2) * (Us[:, 0] - Us[:, 1]) * self.E_props.edge_len  # shape = [n_edges, n_comp=3]
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

    def __init__(self, cfg: ConfigFVM, mesh: FVMMesh, n_comp, bc_tag, us_init=None, device="cuda"):
        self.cfg = cfg
        self.device = device
        self.mesh = mesh
        self.n_comp = n_comp

        E_props = FVMEdgeInfo(cfg, mesh, n_comp, bc_tag, device=device)
        self.cells = FVMCells(mesh.n_cells, n_comp, us_init, device=device)
        self.E_props = E_props

        # Cell divergence calcs
        tri_to_edge = mesh.tri_to_edge
        tri_edge_sign = mesh.tri_edge_signs.unsqueeze(-1) # .to(device)
        # Matrix for converting edge fluxes to cell divergence
        self.flux_mat = self.build_flux_mat(tri_to_edge, -tri_edge_sign, mesh.n_edges, mesh.areas)  # shape = [n_cells * n_comp, n_edges * n_comp]
        # Physical parameters
        self.c = cfg.c      # Speed of sound squared

        self.P_force = PressureForce(E_props, flux_mat=self.flux_mat, c2=self.c**2, V_dims=[0, 1], p_dim=2, device=device)
        self.U_advect = Adevction(E_props, flux_mat=self.flux_mat, V_dims=[0, 1], rho_dim=2, device=device)
        self.U_visc = Viscosity(E_props, mesh.areas, flux_mat=self.flux_mat, cfg=cfg, V_dims=[0, 1], device=device)
        self.KT_diff = KTDiffusion(cfg.v_factor, E_props, device=device)


        self.t_solver = RK3_SSP4(self.cells, cfg.dt, cfg.n_iter, self)

        E_props.clear_temp()
        c_print("Done FVMEquation", color="bright_magenta")

    def solve(self):
        self.t_solver.solve()

    def build_flux_mat(self, tri_to_edge, tri_edge_sign, n_edges, areas, dtype=torch.float32):
        """
        Build the incidence matrix T of shape (n_tri, n_edges).
        For each triangle i and local edge j, we set:
            T[i, tri_to_edge[i, j]] = tri_edge_sign[i, j].
        """
        c_print("Constricting flux matrix", color="bright_magenta")
        # and that self.areas is a tensor of length n_tri.
        n_tri, n_local = tri_to_edge.shape  # typically, n_local == 3

        # Create row indices: each triangle i contributes n_local entries.
        row_indices = torch.arange(n_tri).unsqueeze(1).expand(n_tri, n_local).reshape(-1)

        # Flatten the edge indices from tri_to_edge for column indices.
        col_indices = tri_to_edge.reshape(-1)
        # Flatten the sign values from tri_edge_sign.
        values = tri_edge_sign.reshape(-1).to(dtype)
        # Compute the inverse areas (A_inv is diagonal) and scale the nonzero values.
        areas_inv = (1.0 / areas.cpu()).to(dtype)
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
            fluxes.shape = (n_edges * N_component)

            du/dt = -div(flux) = -sum_i (sign_i * flux_i)
        """
        # # Vectorised version
        # fluxes = fluxes.view(-1, self.n_comp)  # shape: (n_edges, n_component)
        # tri_fluxes = fluxes[self.tri_to_edge]  # shape: [n_cells, 3, n_component]
        # divergence = torch.sum(-self.tri_edge_sign * tri_fluxes, dim=1).squeeze() / self.areas.unsqueeze(-1)     # shape = [n_cells, N_component]

        # Matrix version
        divergence_flat = torch.mv(self.flux_mat, fluxes)  # shape: (n_cells * n_component,)
        divergence = divergence_flat.view(-1, self.n_comp)  # shape: (n_cells, n_component)
        return divergence


    def forward(self, primatives, t=0):
        """ primatives.shape = (n_cells, n_component) """

        E_props = self.E_props
        E_props.precompute_shared(primatives)

        # Advection term
        fluxes = self.U_advect.edge_fluxes()
        # Pressure term
        fluxes += self.P_force.edge_fluxes()
        # MUSCL term
        fluxes += self.KT_diff.edge_fluxes(self.c)
        # Compute divergence
        divergence = self._flux_to_div(fluxes)
        # Viscosity is directly from divergence
        divergence += self.U_visc.divergence(primatives)

        return divergence

    # def get_jacobian(self, prims=None, t=0):
    #     E_props = self.E_props
    #     if t == 0:
    #         E_props.precompute_shared(prims)
    #
    #     advec_jac, diag_mask = self.U_advect.calc_jacobian()
    #     p_jac = self.P_force.Jacobian
    #     jacobian = advec_jac.to_sparse_coo() + p_jac
    #     # print(jacobian)
    #     jacobian = jacobian.coalesce()
    #     return jacobian, None


    def plot_flux(self, fluxes, title="Fluxes", show_index=False, lims=None, Xlims=None):
        plot_edges(self.mesh.vertices.cpu(), self.mesh.edges.cpu(), title=title, colors=fluxes, show_index=show_index, lims=lims, Xlims=Xlims)

    def plot_cells(self, values, title="Cell Values", show_index=False, lims=None, Xlims=None):
        plot_points(self.mesh.centroids.cpu(), values.T, show_index=show_index, title=title, lims=lims, Xlims=Xlims)

    def plot_interp(self, values, title="Cell Values", Xlims=None, resolution=2000):
        plot_interp(self.mesh.vertices, values.T, self.mesh.triangles, title=title, Xlims=Xlims, resolution=resolution)

