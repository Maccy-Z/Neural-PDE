import torch
from abc import ABC
from cprint import c_print

from pde.graph_grid.graph_utils import plot_points, plot_edges, plot_interp
from fvm_mesh import FVMMesh
from edge_process import FVMEdgeInfo
from t_solvers import FVMCells
from integrators import Butcher_adapt
from config_fvm import ConfigFVM
from sparse_utils import to_csr

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
        self.S_const = cfg.S_const
        self.R = cfg.C_v * (cfg.gamma - 1)
        self.C_v_inv = 1 / cfg.C_v


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

        T = E_props.T_faces       # shape = [n_edges, edges=2, n_comp=1]
        grad_V_t = E_props.grad_V  # shape = [n_edges, dim=2, n_comp=2,]
        div_V_edge = E_props.div_V_faces.mean(dim=1)  # shape = [n_edges]

        # Viscosity = mu * (T/T0)^(3/2) * (T0 + S) / (T + S)
        mu = self.mu * (T / 273)**1.5 * (273 + self.S_const) / (T + self.S_const)  # shape = [n_edges, edges=2, n_comp=1]
        # Bulk viscosity: Proportional to T^2
        mu_b = self.mu_b * T ** 2 / 77824

        eye = mu_b * torch.eye(2, device=self.device).unsqueeze(0)
        bulk_tau = div_V_edge.view(-1, 1, 1) * eye
        self.tau = -mu * (grad_V_t + grad_V_t.permute(0, 2, 1)) - bulk_tau  # shape = [n_edges, 2, 2]

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

    #@torch.compile()
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

    #@torch.compile()
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


    #@torch.compile()
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
        self.t_solver = Butcher_adapt(self.cells, cfg.dt, cfg.n_iter, self, name="RK3_SSP6")

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
        flux_mat = torch.sparse_coo_tensor(D_indices, D_values, size=D_shape, device="cpu", dtype=dtype).coalesce()# .cuda().to_sparse_csr()
        flux_mat = to_csr(flux_mat, self.device)
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


