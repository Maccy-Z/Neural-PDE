from __future__ import annotations
from typing import TYPE_CHECKING
import torch
import math

from time_fvm.config_fvm import ConfigFVM
from time_fvm.sparse_utils import to_csr
if TYPE_CHECKING:
    from time_fvm.edge_process import FVMEdgeInfo


class FarfieldBC:
    set_bc_U_face: callable

    def __init__(self, cfg: ConfigFVM, farfield_mask, farfield_normals):
        self.cfg = cfg
        self.exit_cfg = cfg.exit_cfg

        self.farfield_mask = farfield_mask
        self.farfield_normals = farfield_normals
        self.farfield_idx = torch.where(farfield_mask)[0]

        self.R = self.cfg.R
        self.gamma = self.cfg.gamma

        v_far = self.exit_cfg.v_far
        rho_far = self.exit_cfg.rho_far
        T_far = self.exit_cfg.T_far
        P_far = rho_far * T_far * self.R
        a_far = math.sqrt(self.gamma * self.R * T_far)


        if self.exit_cfg.mode == "decay":
            self.set_bc_U_face = self.__decay
            self.tau = 1 / self.exit_cfg.decay_tau
            self.decay_beta = self.exit_cfg.decay_beta
            self.rho_far = rho_far
            self.v_far = v_far
            self.factor = 1
        elif self.exit_cfg.mode == "farfield":
            self.set_bc_U_face = self.__farfield_neuman
            self.factor = 1
            self.R_far = v_far - 2 * a_far / (self.gamma - 1)
        elif self.exit_cfg.mode == "farfield_blended":
            self.set_bc_U_face = self.__farfield_blended
            self.factor = 1
            self.R_m_far = v_far - 2 * a_far / (self.gamma - 1)
            self.R_p_far = v_far + 2 * a_far / (self.gamma - 1)
            self.S_far = P_far / (rho_far ** self.gamma)
        elif self.exit_cfg.mode == "adaptive":
            self.set_bc_U_face = self.__adaptive
            self.dR_m = 0
            self.R_m = v_far - 2 * a_far / (self.gamma - 1)
            self.P_far = P_far
            self.tau = self.exit_cfg.decay_tau
            self.decay_beta = self.exit_cfg.decay_beta
        elif self.exit_cfg.mode == "interior":
            self.beta = 1 - 1 / self.exit_cfg.beta_tau
            self.set_bc_U_face = self.__interior

    def __decay(self, U_face, Us_bc_cells, dt):
        """" Combine two methods to decay boundary:
                U_charachteristic = f * rho_interior * torch.exp(vx_interior - self.v_far)

                f estimates the natural farfield value rho* exp(-v_far)

            Interpolate using moving state f(S):
                S = (1-tau) * S + tau  * dUdt
                U_face = f(S) * U_charachteristic + (1 - f(S)) * U_decay

        """
        V = Us_bc_cells[:, :2]                                          # shape = [n_ff_edge, 2]
        V_n = (V * self.farfield_normals).sum(dim=1, keepdim=True)      # shape = [n_ff_edge, 1]
        tau = dt * self.tau

        rho_bc = self.factor * self.rho_far * torch.exp((V_n - self.v_far)/self.c)

        d_factor = (self.rho_far - rho_bc) + self.decay_beta * (1 - self.factor)
        self.factor = self.factor + tau * d_factor

        # rho_bc = rho_bc.clamp(min=0.3, max=1.4)
        U_face[self.farfield_mask] =  rho_bc

    def __adaptive(self, U_face, Us_bc_cells, dt):
        """ Compressible farfield:
                R+ = u + 2a/(gamma - 1)
                R- = u - 2a/(gamma - 1)
                S = P / rho^gamma
            On exit, we have:
                R+ = R+_int
                R- = R-_inf
                S = S_int

            Adapt V_far.
        """
        gm1 = self.gamma - 1

        V = Us_bc_cells[:, [0, 1]]                                    # shape = [n_ff_edge, 2]
        rho_int = Us_bc_cells[:, 2]                                   # shape = [n_ff_edge]
        T_int = Us_bc_cells[:, 3]                                     # shape = [n_ff_edge]

        # Parallel and tangential velocity
        V_n = -(V * self.farfield_normals).sum(dim=1)      # shape = [n_ff_edge]
        V_t = V + V_n.unsqueeze(-1) * self.farfield_normals

        # Incoming farfield:
        R_m = self.R_m + self.dR_m    # Rm = v_far - 2 * a_far/(gamma - 1)

        # Outgoing (extrapolate from internal) :
        a_int = torch.sqrt(self.gamma * self.R * T_int)
        R_p = V_n + 2 * a_int / gm1       #  R+ = R+_int = V_n + 2 * a_in / (gamma-1)
        # S_p = self.R * T_int * rho_int ** (-gm1)

        # Boundary values: a_bc = (gamma-1)/4 * (R+ - R-)
        a_bc_2 = (gm1/4 * (R_p - R_m)) ** 2
        V_n_bc = 1/2 * (R_p + R_m)
        T_bc = a_bc_2 / (self.gamma * self.R)
        rho_bc = rho_int * (T_bc / T_int) ** (1/gm1) #(a_bc_2 / (self.gamma * S_p)) ** (1 / gm1)

        V_bc = V_t - V_n_bc.unsqueeze(-1) * self.farfield_normals

        U_face_farfield = torch.cat([V_bc, rho_bc.unsqueeze(-1), T_bc.unsqueeze(-1)], dim=-1)
        U_face[self.farfield_mask] = U_face_farfield

        # Adapt farfield conditions. Decay pressure exponentially and update R-:
        P_bc = rho_bc * self.R * T_bc
        P_new = P_bc + dt / self.tau * (self.P_far - P_bc)
        m = self.gamma / gm1
        B = (gm1 ** 2 / (16 * self.gamma)) ** m * (self.R * T_int) ** (-1/gm1) * rho_int
        R_new = R_p - (P_new / B) ** (0.5/m)
        self.dR_m = R_new - self.R_m - dt / self.tau * self.decay_beta * self.dR_m


        # self.v_far = self.v_far + dt / self.tau * (V_n - self.v_far)

        # print(f'{self.v_far = }')

    def __farfield(self, U_face, Us_bc_cells, dt):
        """ Compressible farfield:
                R+ = u + 2a/(gamma - 1)
                R- = u - 2a/(gamma - 1)
                S = P / rho^gamma
            On exit, we have:
                R+ = R+_int
                R- = R-_inf
                S = S_int
        """
        gm1 = self.gamma - 1

        V = Us_bc_cells[:, [0, 1]]                                    # shape = [n_ff_edge, 2]
        rho_int = Us_bc_cells[:, 2]                                   # shape = [n_ff_edge]
        T_int = Us_bc_cells[:, 3]                                     # shape = [n_ff_edge]

        # Parallel and tangential velocity
        V_n = (V * self.farfield_normals).sum(dim=1)      # shape = [n_ff_edge]
        V_t = V - V_n.unsqueeze(-1) * self.farfield_normals

        # Incoming farfield:
        R_m = self.R_far #self.v_far - 2 * self.a_far / gm1    # Rm = v_far - 2 * a_far/(gamma - 1)

        # Outgoing (extrapolate from internal) :
        a_int = torch.sqrt(self.gamma * self.R * T_int)
        R_p = V_n + 2 * a_int / gm1       #  R+ = R+_int = V_n + 2 * a_in / (gamma-1)
        S_p = self.R * T_int * rho_int ** (-gm1)

        # Boundary values: a_bc = (gamma-1)/4 * (R+ - R-)
        V_n_bc = 1/2 * (R_p + R_m)
        a_bc_2 = (gm1/4 * (R_p - R_m)) ** 2
        rho_bc = (a_bc_2 / (self.gamma * S_p)) ** (1 / gm1)
        T_bc = a_bc_2 / (self.gamma * self.R)

        V_bc = V_t + V_n_bc.unsqueeze(-1) * self.farfield_normals

        U_face_farfield = torch.cat([V_bc, rho_bc.unsqueeze(-1), T_bc.unsqueeze(-1)], dim=-1)
        U_face[self.farfield_mask] = U_face_farfield

    def __farfield_blended(self, U_face, Us_bc_cells, dt=None):
        """Set farfield boundary conditions using blended characteristic approach.

        This function implements a smooth, blended farfield boundary condition that
        automatically transitions between inflow and outflow using Riemann invariants
        and entropy. The blending prevents spurious reflections at the boundary by
        smoothly interpolating between interior and farfield states based on local
        flow direction.

        Theory - Riemann Invariants:
        ----------------------------
        For inviscid compressible flow, three characteristic quantities are transported
        along characteristic curves:
        1. Left-running Riemann invariant (characteristic speed: u - a):
           R⁻ = u - 2a/(γ-1)
           where u is normal velocity, a is speed of sound
        2. Entropy invariant (characteristic speed: u):
           S = p/ρ^γ = RT/ρ^(γ-1)
           constant along streamlines for isentropic flow
        3. Right-running Riemann invariant (characteristic speed: u + a):
           R⁺ = u + 2a/(γ-1)

        Blending Strategy:
        ------------------
        The method smoothly interpolates each invariant based on the sign and magnitude
        of its characteristic speed λ = u ± a:

        For each invariant i with characteristic speed λᵢ:
        - If λᵢ >> 0 (strongly outgoing): use interior value (information flows outward)
        - If λᵢ << 0 (strongly incoming): use farfield value (information flows inward)
        - If λᵢ ≈ 0 (near-sonic): blend smoothly between interior and farfield
        Blending function:
            αᵢ = 0.5 * (1 - tanh(λᵢ/c))

        where c is a characteristic speed scale (mean interior sound speed).
        This gives:
        - α → 1 when λ << 0 (use farfield value)
        - α → 0 when λ >> 0 (use interior value)
        - α ≈ 0.5 when λ ≈ 0 (equal blend)

        Blended invariants:
            R⁻_bc = α₁ * R⁻_far + (1-α₁) * R⁻_int
            S_bc  = α₂ * S_far  + (1-α₂) * S_int
            R⁺_bc = α₃ * R⁺_far + (1-α₃) * R⁺_int

        Reconstruction:
        ---------------
        Once the blended invariants are computed, the primitive variables are
        reconstructed:

        1. Normal velocity:
           u_bc = (R⁺_bc + R⁻_bc) / 2
        2. Speed of sound:
           a_bc = (γ-1)/4 * (R⁺_bc - R⁻_bc)
        3. Density (from entropy and speed of sound):
           ρ_bc = (a²_bc / (γ * S_bc))^(1/(γ-1))
        4. Temperature (from equation of state):
           T_bc = a²_bc / (γ * R)
        5. Velocity vector (preserve tangential component):
           V_bc = V_t + u_bc * n
           where V_t is tangential velocity from interior

        Parameters:
        -----------
        U_face : torch.Tensor, shape [n_edges_bc, n_comp]
            Boundary face values to be modified in place
        Us_bc_cells : torch.Tensor, shape [n_farfield_edges, n_comp]
            Interior cell values adjacent to farfield edges [V_x, V_y, ρ, T]
        dt : float, optional
            Time step (unused, kept for interface compatibility)

        Notes:
        ------
        - The tangential velocity component is always preserved from interior
        - The blending scale is O(a), making the transition region sonic-scale
        - Farfield values R⁻_far, R⁺_far, S_far are precomputed from exit conditions
        - This approach is stable for both subsonic and supersonic flows
        - The method reduces to pure extrapolation for supersonic outflow
        - The method reduces to pure farfield prescription for supersonic inflow
        """
        gm1 = self.gamma - 1

        V = Us_bc_cells[:, [0, 1]]                                    # shape = [n_ff_edge, 2]
        rho_int = Us_bc_cells[:, 2]                                   # shape = [n_ff_edge]
        T_int = Us_bc_cells[:, 3]                                     # shape = [n_ff_edge]

        # Parallel and tangential velocity
        V_n = (V * self.farfield_normals).sum(dim=1)      # shape = [n_ff_edge]
        V_t = V - V_n.unsqueeze(-1) * self.farfield_normals

        # Interior invariants:
        a_int = torch.sqrt(self.gamma * self.R * T_int)
        R_m_int = V_n - 2 * a_int / gm1       #  1
        S_int = self.R * T_int * rho_int ** (-gm1)        # 2
        R_p_int = V_n + 2 * a_int / gm1       #  3

        # Interpolation values (assuming c = a_int). Transition smoothly on scale O(c)
        c = a_int.mean()
        v_hat, a_hat = 0.1*V_n / c, 0.1*a_int / c
        lambda1 = v_hat - a_hat
        alpha1 = 0.5 * (1 - torch.tanh(lambda1))
        lambda2 = v_hat
        alpha2 = 0.5 * (1 - torch.tanh(lambda2))
        lambda3 = v_hat + a_hat
        alpha3 = 0.5 * (1 - torch.tanh(lambda3))

        # Set boundary invariants
        R_m_bc = alpha1 * self.R_m_far + (1 - alpha1) * R_m_int
        S_bc = alpha2 * self.S_far + (1 - alpha2) * S_int
        R_p_bc = alpha3 * self.R_p_far + (1 - alpha3) * R_p_int

        # Construct boundary values
        V_n_bc = 1/2 * (R_m_bc + R_p_bc)
        a_bc_2 = (gm1/4 * (R_p_bc - R_m_bc)) ** 2
        rho_bc = (a_bc_2 / (self.gamma * S_bc)) ** (1 / gm1)
        T_bc = a_bc_2 / (self.gamma * self.R)

        V_bc = V_t + V_n_bc.unsqueeze(-1) * self.farfield_normals

        U_face_farfield = torch.cat([V_bc, rho_bc.unsqueeze(-1), T_bc.unsqueeze(-1)], dim=-1)
        U_face[self.farfield_idx] = U_face_farfield

    def __farfield_neuman(self, U_face, Us_bc_cells, dt):
        """Neuman velocity BC """
        V = Us_bc_cells[:, [0, 1]]                                          # shape = [n_ff_edge, 2]
        rho_int = Us_bc_cells[:, 2]                                   # shape = [n_ff_edge]
        T_int = Us_bc_cells[:, 3]                                     # shape = [n_ff_edge]

        # # a_int = math.sqrt(self.gamma * self.R * T_int)  # shape = [n_ff_edge]
        a_int_2 = self.gamma * self.R * T_int
        a_bc_2 = a_int_2

        # Boundary entropy: S- = rho_int^(1-gamma) R T_int
        S_m = rho_int ** (1 - self.gamma) * self.R * T_int
        # rho_b = (a^2/gamma * S) ^(1/(gamma-1))
        rho_bc = (a_bc_2 / (self.gamma * S_m)) ** (1/(self.gamma - 1))
        # T_bc = a^2 / (gamma * R)
        T_bc = a_bc_2 / (self.gamma * self.cfg.R)

        U_face[self.farfield_mask, 2] = rho_bc
        U_face[self.farfield_mask, 3] = T_bc

    def __farfield_isothermal(self, U_face, Us_bc_cells, dt):
        vx_interior = Us_bc_cells[:, 0]
        rho_bc = self.rho_far * torch.exp(vx_interior - self.v_far)
        U_face[self.farfield_mask, 2] = rho_bc

    def __interior(self, U_face, Us_bc_cells, dt):
        vx_interior = Us_bc_cells[:, 0]
        rho_interior = Us_bc_cells[:, 2]
        beta = dt * self.beta

        U_face[self.farfield_mask] = torch.where((vx_interior < 0),
                             rho_interior * torch.exp(vx_interior),
                             (beta * rho_interior + (1-beta) * self.rho_far),
                             )

    def set_bc_U_face(self, U_face, Us_bc_cells):
        """ Set U_face.
            U_face: shape = [n_edges, n_comp], all boundary edges. Set value in place, given by mask.
            Us_bc_cells: shape = [n_bc_edges, n_comp], cell values at boundary edges.
        """
        raise NotImplementedError


class InletBC:
    inlet_mask: torch.Tensor        # shape = [n_edges_bc]
    inlet_normals: torch.Tensor     # shape = [n_inlet_edges, 2]
    def __init__(self, cfg: ConfigFVM, inlet_mask, inlet_normals):
        """ Inlet boundary condition.
            inlet_normals must point inward.
        """
        self.cfg = cfg
        self.inlet_mask = inlet_mask
        self.inlet_normals = inlet_normals

        self.C_v = cfg.C_v
        self.R = self.cfg.R
        self.gamma = self.cfg.gamma

        # Set stagnation condition to be similar to natural parameters.
        bc_cfg = cfg.inlet_cfg
        T_nat = bc_cfg.T_nat
        rho_nat = bc_cfg.rho_nat
        V_x_nat = bc_cfg.V_x_nat
        M_nat = V_x_nat / math.sqrt(self.gamma * self.R * T_nat)
        p_nat = rho_nat * self.R * T_nat
        self.p_0 = p_nat * (1 + (self.gamma - 1)/2 * M_nat ** 2) ** (self.gamma / (self.gamma - 1))
        self.T_0 = T_nat * (1 + (self.gamma - 1)/2 * M_nat ** 2)

        print(f'Typical sound speed: c = {math.sqrt(self.gamma * self.R * T_nat):.2g}')

    def set_bc_U_face(self, U_face, Us_bc_cells, dt):
        """Set inlet boundary conditions using total pressure and total temperature.

        Subsonic inlet boundary conditions based on isentropic
        flow relations. Given stagnation (total) conditions p_0 and T_0, we compute
        the boundary state by combining interior flow.
                -------
        For isentropic flow, the following relations hold:

        1. Pressure ratio (isentropic relation):
           p_0/p = (1 + (γ-1)/2 * M²)^(γ/(γ-1))
        2. Temperature ratio (isentropic relation):
           T_0/T = 1 + (γ-1)/2 * M²
        3. Speed of sound:
           a = √(γ * R * T)
        4. Mach number definition:
           M = V/a

        Parameters:
        U_face : torch.Tensor, shape [n_edges_bc, n_comp]
            Boundary face values to be modified in place
        Us_bc_cells : torch.Tensor, shape [n_inlet_edges, n_comp]
            Interior cell values adjacent to inlet edges [V_x, V_y, ρ, T]
        dt : float
            Time step (unused, kept for interface compatibility)

        Notes:
        - p and T are computed from interior cell values, p=p_int, T=T_int
        - Backflow prevention: p_ratio is clamped to be ≥ 1 + ε to ensure M_bc ≥ 0
        - The tangential velocity component is preserved from the interior flow
        """

        V_int = Us_bc_cells[:, [0, 1]]  # shape = [n_inlet_edge, 2]
        rho_int = Us_bc_cells[:, 2]  # shape = [n_inlet_edge]
        T_int = Us_bc_cells[:, 3]  # shape = [n_inlet_edge]

        # Parallel and tangential velocity
        V_n_int = (V_int * self.inlet_normals).sum(dim=1)      # shape = [n_ff_edge]
        V_t_int = (V_int - V_n_int.unsqueeze(-1) * self.inlet_normals).norm(dim=-1)   # shape = [n_inlet_edge, 2]

        # Interior values
        p_int = rho_int * self.R * T_int
        p_ratio = self.p_0 / p_int
        p_ratio.clamp_(min=1 + 1e-7)  # Ensure no backflow.

        # Boundary values
        M_bc = torch.sqrt((2/(self.gamma - 1)) * (p_ratio ** ((self.gamma - 1)/self.gamma) - 1))
        T_bc = self.T_0 / (1 + (self.gamma - 1)/2 * M_bc ** 2)
        a_bc = torch.sqrt(self.gamma * self.R * T_bc)

        # Inlet velocity
        V_t = V_t_int
        V_n = M_bc * a_bc # Keep normal velocity from interior
        # Convert to x-y component. Note normals point outward, so reverse sign
        inlet_normals = self.inlet_normals
        V_x = V_n * inlet_normals[:, 0] - V_t * inlet_normals[:, 1]
        V_y = V_n * inlet_normals[:, 1] + V_t * inlet_normals[:, 0]

        rho_bc = p_int / (self.R * T_bc)


        U_face_farfield = torch.stack([V_x, V_y, rho_bc, T_bc], dim=-1)
        U_face[self.inlet_mask] = U_face_farfield



class BoundarySetter:
    """ Non-orthogonal correction for Neumann BCs."""
    n_comp: int
    n_edges_bc: int

    grad_comps: torch.Tensor          # shape = [n_neum_edges, 2, 1]
    where_neum: tuple[torch.Tensor]      # shape = [2][n_neum_edges, 2]

    # Matrices for general face values
    A_bc: torch.Tensor
    b_bc: torch.Tensor

    # Farfield boundary condition
    use_farfield: bool
    farfield_calc: FarfieldBC
    exit_cell2edge: torch.Tensor    # shape = (n_cells, 2)  # Exit edge for each cell

    # Inlet boundary condition
    use_inlet: bool
    inlet_calc: InletBC
    inlet_cell2edge: torch.Tensor    # shape = (n_cells, 2)  # Inlet edge for each cell

    def __init__(self, E_props: FVMEdgeInfo):
        self.use_farfield, self.use_inlet = False, False

        self.n_comp = E_props.n_comp
        self.n_edges_bc = E_props.n_edges_bc
        tri_to_edge = E_props.tri_to_edge.view(-1, 3)

        # Flatten out all Neumann BCs and index according to order where_neum_all[0]
        neum_mask_all = torch.zeros_like(E_props.bc_edge_mask)
        neum_mask_all = neum_mask_all.unsqueeze(-1).repeat(1, 4)
        neum_mask_all[E_props.bc_edge_mask] = E_props.neumann_mask
        where_neum_all = torch.where(neum_mask_all)
        where_neum = {'edge': where_neum_all[0], 'comp': where_neum_all[1]}  # shape = [n_neum_edges, 2]
        # Mapping from boundary id to boundary edge id
        self.where_neum = torch.where(E_props.neumann_mask)

        # Mapping from boundary edge to cell
        bc_edge_to_tri = torch.zeros_like(E_props.bc_edge_mask).long()
        bc_edge_to_tri[E_props.bc_edge_mask] = E_props.edge_to_tri_bc

        # Cells corresponding to Neumann BC
        self.neum_cells = bc_edge_to_tri[where_neum_all[0]]  # shape = [n_neum_edges], which cells have neuman BCs
        where_neum['cells'] = self.neum_cells
        # Edge within cell corresponding to Neumann BC
        tri_edge_num = (where_neum['edge'].unsqueeze(-1).repeat(1, 3) == tri_to_edge[where_neum['cells']])
        tri_edge_id = torch.where(tri_edge_num)[1]
        where_neum['tri_edge_id'] = tri_edge_id

        # Which component of gradient is needed for Neumann BC
        self.grad_comps = where_neum['comp'].unsqueeze(1).repeat(1, 2).unsqueeze(2)     # shape = [n_neum_edges, 2, 1]
        # Normal vector of edges
        n_hats = E_props.normals_hat.squeeze()[where_neum['edge']]  # shape = [n_neum_edges, 2]
        # Displacement from centroid to edge
        cent_to_edge = E_props.cent_to_edge_disp[where_neum['cells']].squeeze()  # shape = [n_neum_edge, 3, 2]
        r = cent_to_edge[torch.arange(cent_to_edge.shape[0]), where_neum['tri_edge_id']]  # shape = [n_neum_edge, 2]
        # Normal component of r
        d = n_hats * (r * n_hats).sum(dim=1, keepdim=True)  # shape = [n_neum_edge, 2]
        # Parallel component of r
        self.l = r - d

        A_bc, b_bc = self._build_spm_face_vals(E_props)
        self.A_bc, self.b_bc = A_bc, b_bc

    def set_face_values(self, Us, cell_grads=None, dt=None):
        """Compute and return boundary face values from cell values.

        Uses the precomputed sparse operator to map flattened cell values to
        boundary face values, then applies non-orthogonal correction and
        farfield adjustments if enabled.
        """
        Us_flat = Us.flatten()
        # Final U_face in flattened form.a
        U_face_flat = torch.mv(self.A_bc, Us_flat) + self.b_bc      # shape = [n_edges_bc * n_comp]
        # Reshape back to (n_edges_bc, n_comp)
        U_face = U_face_flat.view(self.n_edges_bc, self.n_comp)

        if cell_grads is not None:
            self._non_orthogonal_correction(U_face, cell_grads)

        if self.use_farfield:
            self.farfield_calc.set_bc_U_face(U_face, Us[self.exit_cell2edge], dt)

        if self.use_inlet:
            self.inlet_calc.set_bc_U_face(U_face, Us[self.inlet_cell2edge], dt)

        return U_face

    def _non_orthogonal_correction(self, U_face, cell_grads):
        """
        Use previous gradient for non-orthogonal correction.

        U_face.shape = [n_bc_faces, n_comp]
        cell_grads.shape = [n_cells, 2, n_comp]

        r = centroid to midpoint.
        d = normal component of r
        U_f = U_0 + d * dUdn + (r-d) grad(U)
        """
        grads = torch.gather(cell_grads[self.neum_cells], 2, self.grad_comps).squeeze()  # shape = [n_neum_edge, 2]

        dU = (grads * self.l).sum(dim=1)  # shape = [n_neum_edge]
        U_face[self.where_neum[0], self.where_neum[1]] += dU

    def _build_spm_face_vals(self, E_props):
        """ Compute bc edge values using sparse matrix multiplication. """

        device = E_props.device
        n_bc = E_props.n_edges_bc  # number of boundary edges
        n_comp = E_props.n_comp
        n_cells = E_props.n_cells

        # Total number of flattened BC rows.
        N = n_bc * n_comp

        # Create flattened indices for the boundary rows and the corresponding component.
        # Each boundary edge gives rise to n_comp rows.
        bc_rows = torch.arange(n_bc, device=device).unsqueeze(1).expand(n_bc, n_comp).reshape(-1)
        comp_idx = torch.arange(n_comp, device=device).unsqueeze(0).expand(n_bc, n_comp).reshape(-1)

        # Reshape the condition masks to a flat vector of length N.
        dirich_mask = E_props.dirich_mask.reshape(-1)  # For Dirichlet conditions.
        neum_mask = E_props.neumann_mask.reshape(-1)  # For Neumann conditions.

        # --- Build sparse matrix A ---
        # For Neumann entries, we want to extract the cell value from Us.
        # For each Neumann row, the corresponding column in Us (flattened) is given by:
        #   col = self.edge_to_tri_bc[ edge_index ] * n_comp + component
        neum_indices = torch.nonzero(neum_mask, as_tuple=False).squeeze(1)  # indices where Neumann is True.
        A_rows = neum_indices
        # bc_rows[neum_indices] gives the corresponding boundary edge for each flattened row.
        A_cols = E_props.edge_to_tri_bc[bc_rows[neum_indices]] * n_comp + comp_idx[neum_indices]
        A_vals = torch.ones_like(A_rows, dtype=torch.float32, device=device)

        size_A = (N, n_cells * n_comp)
        A_bc = torch.sparse_coo_tensor(torch.stack([A_rows, A_cols], dim=0), A_vals, size=size_A)# .coalesce().to_sparse_csr()
        A_bc = to_csr(A_bc, device=device)
        # Build the offset vector b.
        b_bc = torch.empty(N, device=device, dtype=torch.float32)
        # For Dirichlet entries, the prescribed value should override any extracted value.
        b_bc[dirich_mask] = E_props.dirich_val
        # For Neumann entries, add the offset computed from the edge distance.
        # Here, we select the proper component value from self.neumann_val using comp_idx.
        b_bc[neum_mask] = E_props.neumann_val[comp_idx[neum_mask]] * E_props.edge_dists_bc.flatten()[neum_mask]

        return A_bc, b_bc


    def init_farfield(self, cfg: ConfigFVM, farfield_mask, exit_cell2edge, farfield_normals):
        self.use_farfield = True
        self.farfield_calc = FarfieldBC(cfg, farfield_mask, farfield_normals)
        self.exit_cell2edge = exit_cell2edge

    def init_inlet(self, cfg: ConfigFVM, inlet_mask, inlet_cell2edge, inlet_normals):
        self.use_inlet = True
        self.inlet_calc = InletBC(cfg, inlet_mask, inlet_normals)
        self.inlet_cell2edge = inlet_cell2edge

