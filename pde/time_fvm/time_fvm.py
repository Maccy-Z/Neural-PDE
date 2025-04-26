import torch
from abc import ABC
from codetiming import Timer
from cprint import c_print

from pde.graph_grid.graph_utils import plot_points, plot_interp_graph, plot_edges, plot_interp
from pde.time_fvm.fvm_mesh import FVMMesh
from pde.time_fvm.edge_process import FVMEdgeInfo
from pde.time_fvm.t_solvers import FVMCells
from pde.time_fvm.integrators import Euler, RK3_SSP4, Adams3PC, Adams4PC, Butcher_adapt
from pde.time_fvm.config_fvm import ConfigFVM

class PhysicalSetup:
    """ Set physical properties of fluid. """
    E_props: FVMEdgeInfo
    tau: torch.Tensor       # shape = [n_edges, 2, 2]
    P_face: torch.Tensor    # shape = [n_edges, 2, 1]
    c: torch.Tensor         # shape = [n_edges, 2, 1]

    def __init__(self, E_props: FVMEdgeInfo, cfg: ConfigFVM, device="cpu"):
        self.E_props = E_props
        self.device = device

        self.gamma = cfg.gamma
        self.mu = cfg.viscosity
        self.mu_b = cfg.visc_bulk
        self.R = cfg.C_v * (cfg.gamma - 1)
        #self.M = cfg.M
        self.C_v_inv = 1 / cfg.C_v

        # Precompute
        self.eye = self.mu_b * torch.eye(2, device=self.device).unsqueeze(0)

    @torch.compile()
    def state_to_primative(self, state):
        """ Convert """
        momentum, density, Q = state[:, [0, 1]], state[:,[2]], state[:,[3]]

        V = momentum / density
        T = self.C_v_inv * (Q / density - 0.5 * V.norm(dim=1, keepdim=True) ** 2)
        primatives = torch.cat([V, density, T], dim=-1)

        return primatives, state


    def _tau(self):
        """ Compute stress tensor:
                tau = mu * (grad(V) + grad(V).T) + mu_b * div(V) * I
         """
        E_props = self.E_props

        grad_V_t = E_props.grad_V  # shape = [n_edges, dim=2, n_comp=2,]
        div_V_edge = E_props.div_V_faces.mean(dim=1)  # shape = [n_edges]

        bulk_tau = div_V_edge.view(-1, 1, 1) * self.eye
        self.tau = -self.mu * (grad_V_t + grad_V_t.permute(0, 2, 1)) - bulk_tau  # shape = [n_edges, 2, 2]

    def _pressure(self):
        """ Pressure force:
                P = rho * C_v * (gamma - 1) * T = R * rho * T
        """
        E_props = self.E_props
        rho_faces = E_props.rho_faces  # shape = [n_edges, edges=2, n_comp=1]
        T_faces = E_props.T_faces       # shape = [n_edges, edges=2, n_comp=1]

        self.P_face = self.R * rho_faces * T_faces
        self.c = torch.sqrt(self.gamma * self.P_face / rho_faces)  # shape = [n_edges, edges=2, n_comp=1]

        # assert not torch.any(torch.isnan(self.c))

    @torch.compile()
    def update(self):
        # E_props = self.E_props
        #E_props.T_faces = E_props.T_faces.clamp(min=10, max=2000)

        self._tau()
        self._pressure()



class FVMEdgeFunc(ABC):
    device: str

    #@abstractmethod
    def edge_fluxes(self, fluxes=None):
        """ Compute flux for each edge
        """
        pass


class Adevction(FVMEdgeFunc):
    """ out_i = div(rho V V_i) for velocity V, i = {x, y}
        dims: Which dimensions of Us are advected.
    """
    E_props: FVMEdgeInfo

    def __init__(self, E_props: FVMEdgeInfo, phy_setup: PhysicalSetup, cfg: ConfigFVM, device="cpu"):
        self.device = device
        self.E_props = E_props
        self.phy_setup = phy_setup
        # self._build_jacobian(diag=False)

    #@torch.compile()
    def edge_fluxes(self, fluxes=None):
        """ rho * U @ V.T @ n = rho V * phi
            f_x = rho V_x * phi
            f_y = rho V_y * phi
            f_rho = rho * phi
            f_E = (Q+p) * phi
        """
        E_props = self.E_props
        #V_faces = E_props.Vs_faces      # shape = [n_edges, edges=2, n_comp=2]
        rho_faces = E_props.rho_faces # shape = [n_edges, edges=2, n_comp=1]
        #T_faces = E_props.T_faces       # shape = [n_edges, edges=2, n_comp=1]
        phi = E_props.phi           # Linear interpolation of convection vector = (v_faces dot normal). shape = [n_edges, edges=2]
        mom_f = E_props.mom_faces
        Q_faces = E_props.Q_faces # = rho_faces * (1/2  * V_faces.norm(dim=-1, keepdim=True) ** 2 + self.C_v * T_faces)

        Q_p_P = Q_faces + self.phy_setup.P_face
        Us_f = torch.cat([mom_f, rho_faces, Q_p_P], dim=-1)  # shape = [n_edges, edges=2, n_comp=3]
        advec_flux = Us_f * phi.unsqueeze(-1)           # shape = [n_edges, edges=2, n_comp=3]
        advec_flux = advec_flux.mean(dim=1)              # shape = [n_edges, n_comp=3]

        if fluxes is None:
            return advec_flux
        else:
            fluxes += advec_flux


    # def _build_F_A(self, A):
    #     """ Construct matrix M_lik = F_li @ A_ik for each edge i. Done in a way ready for reshaping.
    #         F is the flux matrix, A is the gradient matrix.
    #
    #         A.shape = [3*n_edges, 3*n_cells]
    #         F.shape = [3*n_cells, 3*n_edges]
    #
    #         return.shape = [3*n_cells * 3*n_cells, 3*n_edges]
    #     """
    #     E_props = self.E_props
    #     n_edges, n_cells = E_props.n_edges, E_props.n_cells
    #     F = self.flux_mat.to_sparse()#.cpu()
    #
    #     F_indices = F._indices()  # shape: (2, nnz_F): [l_indices; i_indices]
    #     F_values = F._values()  # shape: (nnz_F,)
    #     A_indices = A._indices()  # shape: (2, nnz_G): [i_indices; k_indices]
    #     A_values = A._values()  # shape: (nnz_G,)
    #
    #     # # Number of nonzero elements:
    #     # nnz_F = F_indices.shape[1]
    #     # nnz_G = A_indices.shape[1]
    #
    #     # Broadcast F_indices[1] (i indices) and G0_indices[0] (i indices) to compare:
    #     f_i = F_indices[1].unsqueeze(1)  # shape: (nnz_F, 1)
    #     g_i = A_indices[0].unsqueeze(0)  # shape: (1, nnz_G)
    #     match_mask = (f_i == g_i)  # shape: (nnz_F, nnz_G)
    #
    #     # Get the indices of the matching pairs. Do on CPU since GPU has size limit
    #     match_mask = match_mask.cpu()
    #     f_match_idx, g_match_idx = torch.nonzero(match_mask, as_tuple=True)
    #     f_match_idx, g_match_idx = f_match_idx.to(self.device), g_match_idx.to(self.device)
    #
    #     # For each matching pair, get:
    #     # - l from F (row index of F)
    #     # - i from F (column index of F, also matching the row index of G0)
    #     # - k from G0 (column index of G0)
    #     # - corresponding values f_val and g_val
    #     l_vals = F_indices[0][f_match_idx]  # shape: (num_matches,)
    #     i_vals = F_indices[1][f_match_idx]  # shape: (num_matches,)
    #     f_vals = F_values[f_match_idx]  # shape: (num_matches,)
    #     k_vals = A_indices[1][g_match_idx]  # shape: (num_matches,)
    #     g_vals = A_values[g_match_idx]  # shape: (num_matches,)
    #
    #     # Now compute the flattened row indices for M:
    #     # Here, we assume the flattened row is defined as: new_row = l * (3*n_cells) + k
    #     new_rows = l_vals * (3 * n_cells) + k_vals  # shape: (num_matches,)
    #     new_cols = i_vals  # shape: (num_matches,)
    #     new_vals = f_vals * g_vals  # shape: (num_matches,)
    #
    #     # Build the sparse matrix M (of shape (L*K, i), here K is 3*n_cells)
    #     M_shape = ((3*n_cells )* (3 * n_cells), 3*n_edges)
    #     M_indices = torch.stack([new_rows, new_cols])
    #     M = torch.sparse_coo_tensor(M_indices, new_vals, M_shape)
    #
    #     return M
    #
    #
    # def _build_jacobian(self, diag):
    #     """
    #         U_cell = [interleave(mom_x | mom_y | rho)], shape = [3*n_cell]
    #         U_face = A(U_cell)              shape = [3*n_edges, 2], second dim is Left / Right side of face.
    #         d(U_f_i, L/R)/d(U_c_j) = :
    #                             0 if cell_j doesnt have face_i  - Or computing wrong mom_x, mom_y, rho component.
    #                             1 if face_i is on cell_j and cell_j is on L/R side.
    #         For jacobian:
    #             advec = sum_j 1/2 Uf_ij phi_ij
    #             dadvec/dUc_k = 1/2 sum_j phi_ij dUf_ij/dUc_k
    #                          = 1/2 sum_j phi_ij J_ijk
    #                          -> A_L @ phi_iL | A_L @ phi_iR
    #         Divergence:
    #             D = F @ advec
    #             dD/dUc_k = F @ dadvec/dUc_k
    #                      = F @ A_L @ phi_iL | F @ A_L @ phi_iR
    #                      = M @ [phi_iL | phi_iR]
    #     """
    #     E_props = self.E_props
    #     n_edges, n_cells = E_props.n_edges, E_props.n_cells
    #     A_left, A_right = self.E_props.dUf_dUc
    #
    #     """ Method 2: Sparse matrix precompute """
    #     M_L = self._build_F_A(A_left)
    #     M_R = self._build_F_A(A_right)
    #     # Join M = [M_L | M_R]
    #     M_R_indices = M_R._indices().clone()
    #     M_R_indices[1:] += 3*n_edges
    #     M_indices = torch.cat([M_L._indices(), M_R_indices], dim=1)
    #     new_values = torch.cat([M_L._values(), M_R._values()]) / 2
    #     new_shape = (3*n_cells*3*n_cells, 6*n_edges)
    #     M = torch.sparse_coo_tensor(M_indices, new_values, new_shape).coalesce()
    #
    #     bc_cells = self.E_props.edge_to_tri_bc
    #     bc_cells = torch.cat([3*bc_cells, 3*bc_cells+1, 3 * bc_cells + 2])
    #     self.M_holder = SparseReshapeMM(M, 3 * n_cells, diag=diag, device=self.device, zero_rows=bc_cells)
    #
    #
    # def calc_jacobian(self):
    #     E_props = self.E_props
    #     phi = E_props.phi
    #     phi_f = phi.repeat_interleave(3, dim=0)  # shape = [3*n_edges, edges=2]
    #     phi_vec = torch.cat([phi_f[:, 0], phi_f[:, 1]])  # shhape = [6*n_edges]
    #
    #     # New method
    #     Jac = self.M_holder.multiply(phi_vec)
    #
    #     return Jac


class Viscosity(FVMEdgeFunc):
    """ Viscous forces:
            Shear viscosity: div(mu grad(V)) = sum_f grad(V) * mu_f * l_f
            Bulk viscosity: k * grad(div(V))
    """
    E_props: FVMEdgeInfo
    def __init__(self, E_props: FVMEdgeInfo, stress_calc: PhysicalSetup, flux_mat, device="cpu"):
        self.device = device
        self.E_props = E_props
        self.stress_calc = stress_calc

        #proj_mat = E_props.V_insertion_matrix # create_insertion_matrix(E_props.n_edges, E_props.n_component, [0, 1], device=device).to_sparse_csr()
        # self.visc_mat = flux_mat @ proj_mat

    # def divergence(self):
    #     """ Viscosity is limited after computing divergence for stability """
    #     E_props = self.E_props
    #
    #     tau = self.stress_calc.tau
    #     F = (tau * E_props.normals.unsqueeze(-1)).sum(dim=-2)  # shape = [n_edges, 2]
    #
    #     div_visc = self.visc_mat @ F.flatten()
    #     div_visc = div_visc.view(-1, E_props.n_comp)         # shape = [n_cells, n_comp]
    #
    #     return div_visc

    def edge_fluxes(self, fluxes=None):
        E_props = self.E_props

        tau = self.stress_calc.tau
        F = (tau * E_props.normals.unsqueeze(-1)).sum(dim=-2)  # shape = [n_edges, 2]

        if fluxes is None:
            fluxes = torch.zeros(self.E_props.n_edges, self.E_props.n_comp, device=self.device)
            fluxes[:, :2] = F
            return fluxes
        else:
            fluxes[:, :2] += F

    #@torch.compile()
    # n_edges, n_comp = E_props.n_edges, E_props.n_comp
    # edge_len_mu = edge_len_mu.repeat(1, 2)
    # visc_mat = create_selection_matrix(n_blocks=n_edges, block_size=n_comp, selected_dims=V_dims, weights=-edge_len_mu).to_sparse_csr().cuda()
    # proj_visc_mat = proj_mat @ visc_mat
    # self.A_visc = flux_mat @ proj_visc_mat @ self.E_props.A_face_grad
    # self.b_visc = flux_mat @ proj_visc_mat @ self.E_props.b_face_grad
    # # self.b_visc = self.b_visc.view(-1, 1)
    #
    # """ Combine sparse matrix for bulk viscosity """
    # idx = torch.arange(n_edges, device=device)
    # # Compute the row and column indices for non-zero entries.
    # rows = torch.cat([2 * idx, 2 * idx + 1])
    # cols = torch.cat([idx, idx])
    # indices = torch.stack([rows, cols])  # Shape: [2, 2*n_edges]
    # # Compute the corresponding values from the normals
    # values = torch.cat([E_props.normals[:, 0], E_props.normals[:, 1]])
    # M = torch.sparse_coo_tensor(indices, values, size=(2 * n_edges, n_edges), device=device).to_sparse_csr()
    #
    # self.A_visc_bulk = - self.mu_b * flux_mat @ proj_mat @ M
    #
    # # Clamp viscosity to k * A / dt
    # self.clip_val = cfg.bulk_visc_lim * areas.to(device) / cfg.dt

    # def divergence(self, Us):
    #     """ Viscosity is limited after computing divergence for stability """
    #     E_props = self.E_props
    #
    #     """ Even fuller spm """
    #     Us_flat = Us.flatten()
    #     # div_visc = self.A_visc @ Us_flat + self.b_visc
    #     div_visc = torch.addmv(self.b_visc, self.A_visc, Us_flat)
    #     div_visc = div_visc.view(-1, E_props.n_comp)         # shape = [n_cells, n_comp]
    #
    #     """ Bulk viscosity: gradient = dux/dx + duy/dy, least squares gradient. Use this to compute cell divergence. """
    #     if self.mu_b > 0:
    #         div_V_edge = E_props.div_V_faces.mean(dim=1)
    #
    #         # flux_bulk = div_u_edge * E_props.normals        # shape = [n_edges, 2]
    #         div_bulk = torch.mv(self.A_visc_bulk, div_V_edge)
    #         div_bulk = div_bulk.view(-1, E_props.n_comp)         # shape = [n_cells, n_comp]
    #
    #         # Clipping bulk viscosity
    #         bulk_norm = div_bulk.norm(dim=-1)
    #         clip_val = self.clip_val
    #         div_limit = torch.where(bulk_norm > clip_val, clip_val/bulk_norm, 1).unsqueeze(-1)
    #         div_bulk = div_bulk * div_limit
    #     else:
    #         div_bulk = 0
    #     return div_visc + div_bulk


class Heating(FVMEdgeFunc):
    """ Viscous heating term: div(tau V) = sum_f tau_f * V_f * n_f
        Thermal conductivity term:  div(grad(T)) = sum(grad(T) * n_f)
    """
    E_props: FVMEdgeInfo
    def __init__(self, E_props: FVMEdgeInfo, stress_calc:PhysicalSetup, cfg: ConfigFVM, device="cpu"):
        self.E_props = E_props
        self.stress_calc = stress_calc

        self.kappa = cfg.thermal_cond
        self.device = device

    @torch.compile()
    def edge_fluxes(self, fluxes=None):
        E_props = self.E_props
        normals = E_props.normals       # shape = [n_edges, 2]
        V_face = E_props.Vs_faces       # shape = [n_edges, edges=2, n_comp=2]

        tau = self.stress_calc.tau

        V_face = V_face.mean(dim=1)     # shape = [n_edges, 2]
        heating = (tau * V_face.unsqueeze(1) * normals.unsqueeze(-1)).sum(dim=(-1, -2))

        """ Thermal conductivity:
                div(grad(T)) = sum(grad(T) * n_f)
        """
        grad_T_n = E_props.grad_T_n     # shape = [n_edges]
        heating -= self.kappa * grad_T_n * E_props.edge_len.squeeze()

        if fluxes is None:
            fluxes = torch.zeros(self.E_props.n_edges, self.E_props.n_comp, device=self.device)
            fluxes[:, 3] = heating
            return fluxes
        else:
            fluxes[:, 3] += heating


class PressureForce(FVMEdgeFunc):
    """ Special case. N
        grad(rho) = div(rho I) """
    E_props: FVMEdgeInfo

    def __init__(self, E_props: FVMEdgeInfo, phy_setup: PhysicalSetup, device="cpu"):
        self.device = device
        self.E_props = E_props
        self.phy_setup = phy_setup
        #self.proj_mat = E_props.V_insertion_matrix
        #self.flux_mat = flux_mat

        # self._build_jacobian(zero_bc=True, diag_only=False)

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

    def edge_fluxes(self, fluxes=None):
        normals = self.E_props.normals             # shape = [n_edges, 2]
        P_face = self.phy_setup.P_face      # shape = [n_edges, edges=2, n_comp=1]

        P_face = P_face.mean(dim=1)  # shape = [n_edges, 1]
        P_n = P_face * normals                 # shape = [n_edges, 2]

        if fluxes is None:
            fluxes = torch.zeros(self.E_props.n_edges, self.E_props.n_comp, device=self.device)
            fluxes[:, :2] = P_n
            return fluxes # _flat
        else:
            fluxes[:, :2] += P_n


class KTDiffusion(FVMEdgeFunc):
    """ Diffusion term from K-T solver """
    E_props: FVMEdgeInfo

    def __init__(self, v_factor, phy_setup: PhysicalSetup, E_props: FVMEdgeInfo, device="cpu"):
        self.device = device
        self.v_factor = v_factor
        self.E_props = E_props
        self.phy_setup = phy_setup


    @torch.compile()
    def edge_fluxes(self, dt):
        E_props = self.E_props
        rho_face = E_props.rho_faces
        Vs_face = E_props.Vs_faces      # shape = [n_edges, edges=2, n_comp=2]
        Q_face = E_props.Q_faces   # shape = [n_edges, edges=2, n_comp=1]
        mom_face = E_props.mom_faces

        Us = torch.cat([mom_face, rho_face, Q_face], dim=2)  # shape = [n_edges, 2, n_comp]

        # Wavespeed is c + v_max. Clip velocity wavespeed to k*c + v_max
        Vs = Vs_face.norm(dim=-1)            # shape = [n_edges, edges=2]
        Vs_max = Vs.max(dim=1, keepdim=True).values   # shape = [n_edges, 1]
        c = self.phy_setup.c.max(dim=1).values  # shape = [n_edges, 1]
        # a = torch.tensor([[self.v_factor * c, self.v_factor * c, c, c]], device=self.device)
        # a = a.repeat(self.E_props.n_edges, 1) + Vs_max  # shape = [n_edges, n_comp]
        a = Vs_max + c

        # Maximum diffusion distance is a * dt/2 < tri_height -> a < 2 * tri_height / dt
        # Assume tri_height = k * edge_len / 2
        edge_len = E_props.edge_len
        # a = a.clamp(max=0.75 * edge_len / dt)  # shape = [n_edges, 1]
        kt_fluxes = (a/2) * (Us[:, 0] - Us[:, 1]) * edge_len  # shape = [n_edges, n_comp]

        return kt_fluxes #* 0.25


class FVMEquation:
    mesh: FVMMesh
    E_props: FVMEdgeInfo
    edges: FVMEdgeFunc
    cells: FVMCells
    n_comp: int

    def __init__(self, cfg: ConfigFVM, mesh: FVMMesh, n_comp, bc_tag, us_init=None, device="cuda"):
        self.cfg = cfg
        self.device = device
        self.mesh = mesh
        self.n_comp = n_comp

        E_props = FVMEdgeInfo(cfg, mesh, n_comp, bc_tag, device=device)
        # Physical parameters
        self.phy_setup = PhysicalSetup(E_props, cfg=cfg, device=device)

        self.cells = FVMCells(mesh.n_cells, n_comp, init_val=us_init, phys_setup=self.phy_setup, device=device)
        self.E_props = E_props

        # Matrix for converting edge fluxes to cell divergence
        self.flux_mat = self.build_flux_mat(mesh.tri_to_edge, -mesh.tri_edge_signs, mesh.n_edges, mesh.areas)  # shape = [n_cells, n_edge]


        self.P_force = PressureForce(E_props, self.phy_setup, device=device)
        self.U_advect = Adevction(E_props, self.phy_setup, cfg=cfg, device=device)
        self.U_visc = Viscosity(E_props, self.phy_setup, flux_mat=self.flux_mat, device=device)
        self.Heat = Heating(E_props, self.phy_setup, cfg=cfg, device=device)
        self.KT_diff = KTDiffusion(cfg.v_factor, self.phy_setup, E_props, device=device)

        # self.t_solver = Adams4PC(self.cells, cfg.dt, cfg.n_iter, self)
        self.t_solver = Butcher_adapt(self.cells, cfg.dt, cfg.n_iter, self, name="RK3_SSP4")

        E_props.clear_temp()
        c_print("Done FVMEquation", color="bright_magenta")


    def solve(self):
        self.t_solver.solve()


    def forward(self, primatives, dt, t):
        """ primatives.shape = (n_cells, n_component) """
        E_props = self.E_props
        E_props.precompute_shared(primatives, dt)

        self.phy_setup.update()

        # Advection term
        fluxes = self.U_advect.edge_fluxes()
        # Pressure term
        self.P_force.edge_fluxes(fluxes)
        # Viscosity term
        self.U_visc.edge_fluxes(fluxes)
        # Heating term
        self.Heat.edge_fluxes(fluxes)
        # MUSCL term
        fluxes += self.KT_diff.edge_fluxes(dt)
        # Compute divergence
        divergence = self._flux_to_div(fluxes)

        # For plotting
        self.divergence = divergence

        #
        # self.pressure_flux = self.P_force.edge_fluxes()
        # self.pressure_div = self._flux_to_div(self.pressure_flux)
        # self.advect_flux = self.U_advect.edge_fluxes()
        # self.advect_div = self._flux_to_div(self.advect_flux)
        # self.kt_flux = self.KT_diff.edge_fluxes(dt)
        # self.kt_div = self._flux_to_div(self.kt_flux)
        # self.divergence = divergence
        # self.heat_flux = self.Heat.edge_fluxes()
        # self.heat_div = self._flux_to_div(self.heat_flux)
        # self.visc_flux = self.U_visc.edge_fluxes()
        # self.visc_div = self._flux_to_div(self.visc_flux)

        return divergence

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

        D_shape = [n_tri, n_edges]
        flux_mat = torch.sparse_coo_tensor(D_indices, D_values, size=D_shape, device="cpu", dtype=dtype).coalesce().cuda().to_sparse_csr()

        return flux_mat


    def _flux_to_div(self, fluxes):
        """ Compute cell divergence using fluxes.
            fluxes.shape = (n_edges * N_component)

            du/dt = -div(flux) = -sum_i (sign_i * flux_i)
        """
        # Matrix version
        divergence = torch.mm(self.flux_mat, fluxes)  # shape: (n_cells * n_component,)
        return divergence


    def plot_flux(self, fluxes, title="Fluxes", show_index=False, lims=None, Xlims=None):
        plot_edges(self.mesh.vertices.cpu(), self.mesh.edges.cpu(), title=title, colors=fluxes, show_index=show_index, lims=lims, Xlims=Xlims)

    def plot_cells(self, values, title="Cell Values", show_index=False, lims=None, Xlims=None):
        plot_points(self.mesh.centroids.cpu(), values.T, show_index=show_index, title=title, lims=lims, Xlims=Xlims)

    def plot_interp(self, values, title="Cell Values", Xlims=None, resolution=2000):
        plot_interp(self.mesh.vertices, values.T, self.mesh.triangles, title=title, Xlims=Xlims, resolution=resolution)

    def pretty_plot(self, primatives, Xlims=None, title=None):
        Vx, Vy, rho, T = primatives[:, 0], primatives[:, 1], primatives[:, 2], primatives[:, 3]

        P = self.phy_setup.R * rho * T
        c = torch.sqrt(P / rho)

        Mx, My = Vx / c, Vy / c
        M_num = torch.sqrt(Mx ** 2 + My ** 2)

        plot_vals = torch.stack([P, M_num, self.divergence[:, 3] ], dim=0)

        title = [f"Pressure: {title}", f"Mach number: {title}", f'Heating: {title}']
        plot_interp(self.mesh.vertices, plot_vals, self.mesh.triangles, title=title, Xlims=Xlims)


