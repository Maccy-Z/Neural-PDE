import math
from cprint import c_print
import torch

from time_fvm.fvm_mesh import FVMMesh
from time_fvm.fvm_store import Edge, EdgeBCTypes
from time_fvm.config_fvm import ConfigFVM
from time_fvm.sparse_utils import lift_sparse_matrix, combine_edge_operators, to_csr
from time_fvm.edge_boundary import BoundarySetter


class SlopeLimiter:
    def __init__(self, areas, cfg: ConfigFVM):
        lim_p = cfg.lim_p
        K = cfg.lim_K
        areas = areas.view(-1, 1, 1)

        self.eps_p = (K * areas ** 0.5) ** (lim_p+1)

        if lim_p == 1:
            self._limit = self.p1
        elif lim_p == 2:
            self._limit = self.p2
        elif lim_p == 3:
            self._limit = self.p3
        elif lim_p == 4:
            self._limit = self.p4
        elif lim_p == 5:
            self._limit = self.p5
        else:
            raise NotImplementedError(f"Limiter of order {lim_p} is not implemented.")

    def p1(self, delta, dU):
        """ BJ limiter"""
        dU = 2 * ((dU > 0).float() - 0.5) * (dU.abs() + 1e-8)
        r = delta / dU  # shape = [n_cells, neigh=3, n_comp]
        phi = torch.clamp(r, min=0., max=1)  # shape = [n_cells, neigh=3, n_comp]
        return phi

    def p2(self, delta, dU):
        """ Venkatakrishnan limiter"""
        phi = (delta ** 2 + self.eps_p + 2 * delta * dU) / (delta ** 2 + 2 * dU ** 2 + delta * dU + self.eps_p)
        return phi

    def p3(self, delta, dU):
        """ 3rd order limiter - https://arc.aiaa.org/doi/epdf/10.2514/6.2022-1374"""
        a = delta.abs()
        b = dU.abs()

        a_eps = a ** 3 + self.eps_p
        S = 4 * b ** 2
        phi = (a_eps + a * S) / (a_eps + b * (delta ** 2 + S))
        phi = torch.where(a < 2 * b, phi, 1)
        return phi

    def p4(self, delta, dU):
        """ 4th order limiter """
        a = delta.abs()
        b = dU.abs()
        a_eps = a ** 4 + self.eps_p
        S = 2 * b * (a ** 2 - 2 * b * (a - 2 * b))
        phi = (a_eps + a * S) / (a_eps + b * (delta ** 3 + S))
        phi = torch.where(a < 2 * b, phi, 1)
        return phi

    def p5(self, delta, dU):
        """ 5th order limiter """
        a = delta.abs()
        b = dU.abs()
        a_eps = a ** 5 + self.eps_p
        S = 8 * b ** 2 * (a ** 2 - 2 * b * (a - b))
        phi = (a_eps + a * S) / (a_eps + b * (delta ** 4 + S))

        phi = torch.where(a < 2 * b, phi, 1)
        return phi

    def limit(self, delta, dU):
        """ delta: maximum allowed values
            dU: Predicted value from lstsq gradient
        """
        phi = self._limit(delta, dU)    # shape = [n_cells, neigh=3, n_comp]
        # Cell wide clamping
        phi = torch.min(phi, dim=1, keepdim=True).values  # shape = [n_cells, neigh=1, n_comp]

        return phi


class FVMEdgeInfo:
    device: str
    n_edges: int
    n_cells: int
    n_comp: int
    slope_limiter: SlopeLimiter

    # Shared
    edge_len: torch.Tensor  # shape = (n_edges, 1)
    normals: torch.Tensor  # shape = (n_edges, 2)
    normals_hat: torch.Tensor  # shape = (n_edges, 2, 1)
    X_orthog: torch.Tensor      # shape = (n_edges, 2, 1)
    cell_disps: torch.Tensor        # shape = (n_edges, 2)

    # Main mesh
    edge_to_tri_main: torch.Tensor  # shape = [n_edges_m, 2], ordered so triangle parallel to edge normal comes last, antiparallel first.
    cell_dist_proj: torch.Tensor  # shape = (n_edges_m)
    tri_edge_signs: torch.Tensor  # shape = (3 * n_cells)
    tri_to_edge: torch.Tensor  # shape = (3 * n_cells)
    cent_to_edge_disp: torch.Tensor  # shape = (n_cells, 3, 2)  # Displacement vector between centroid to edge, for every edge

    # Boundary condition
    n_edges_bc: int             # Number of boundary edges
    bc_edge_mask: torch.Tensor  # shape = (n_edges)
    bc_locations: torch.Tensor  # shape = (n_edges_bc)  # int version of bc_edge_mask
    dirich_mask: torch.Tensor  # shape = (n_edges, n_comp)
    neumann_mask: torch.Tensor  # shape = (n_edges, n_comp)
    edge_to_tri_bc: torch.Tensor  # shape = (n_edges_bc)
    bc_edge_side: torch.Tensor  # shape = (n_edges_bc, 2)  # Side of the edge for each boundary edge
    boundary_setter: BoundarySetter

    # Gradients
    G_mats: torch.Tensor  # shape = (2*n_cells, n_cells)  Gradient matrix for every cell
    edge_dists_bc: torch.Tensor  # shape = (n_bc_edges, n_comp)     Distance between cell centroids, for every edge_bc
    neigh_combine: torch.Tensor # shape = (n_cell, 2). Used for masking neighbors of cell incl boundary edges, in format [Us, U_face_bc]

    # Temporary Variables
    grad_V: torch.Tensor        # shape = (n_edges, {dx,dy}, {vx,vy})  Gradient at V_faces
    Vs_faces: torch.Tensor  # shape = (n_edges, 2, 2)  Face values
    rho_faces: torch.Tensor  # shape = (n_edges, 2, 1)  Face values
    T_faces: torch.Tensor # shape = (n_edges, 2, 1)  Face values
    grad_T_n: torch.Tensor  # shape = (n_edges)         Face temperature gradient
    div_V_faces: torch.Tensor  # shape = (n_edges, 2)  Divergence of V_faces
    mom_faces: torch.Tensor     # shape = (n_edges, 2, 2)  Face values
    Q_faces: torch.Tensor       # shape = (n_edges, 2, 1)  Face energy values
    phi: torch.Tensor  # shape = (n_edges, 1)  Face values = V_faces dot normals. After averaging over faces.
    cell_grads: torch.Tensor = None # shape = (n_cells, 2, n_comp)  Gradient at cells. Used for boundary setter as None.
    bc_type_str: list[str]         # BC types for each bc edge. Used for saving mesh.


    def __init__(self, cfg: ConfigFVM, mesh: FVMMesh, n_comp, bc_tags, device="cpu"):
        self.device = device
        self.cfg = cfg
        self.mesh = mesh
        self.n_edges = mesh.n_edges
        self.n_cells = mesh.n_cells
        self.n_comp = n_comp
        self.slope_limiter = SlopeLimiter(mesh.areas.to(device), cfg)


        self.C_v = cfg.C_v

        self.edge_to_tri_main = mesh.edge_to_tri_main.to(device)
        self.cent_to_edge_disp = mesh.cent_to_edge_disp.to(device).unsqueeze(-1)
        self.tri_edge_signs = (-self.mesh.tri_edge_signs + 1 / 2).to(torch.int32).view(3*self.n_cells).to(device)
        self.tri_to_edge = mesh.tri_to_edge.view(3*self.n_cells).to(device)

        self.edge_to_tri_bc = mesh.edge_to_tri_bc.to(device)
        self.bc_edge_mask = mesh.bc_edge_mask.to(device)

        (cell_disps, edge_dists_bc, G_mats, neigh_combine, edge_to_tri_comb) = mesh.cell_grad_stuff
        self.edge_dists_bc = edge_dists_bc.to(device).unsqueeze(-1).expand(-1, self.n_comp)
        G_mats = torch.cat([G_mats[0], G_mats[1]], dim=0)
        self.G_mats = to_csr(G_mats, device) #torch.sparse_csr_tensor(G_mats.crow_indices().to(torch.int32), G_mats.col_indices().to(torch.int32), G_mats.values(), G_mats.size())

        self.neigh_combine = neigh_combine.to(device)
        self.cell_disps = cell_disps.to(device)
        self.normals = mesh.normals.to(device)
        self.edge_len = torch.norm(self.normals, dim=1).to(device).unsqueeze(-1)
        normal_hat = self.normals / self.edge_len
        self.normals_hat = normal_hat.unsqueeze(-1)
        # Non orthogonal correction
        cell_disps = torch.full((self.n_edges, 2), float("nan"), device=device)
        cell_disps[~self.bc_edge_mask] = self.cell_disps
        d_cos_theta = (normal_hat * cell_disps).sum(dim=1)
        self.cell_dist_proj = d_cos_theta
        X_orthog = cell_disps / d_cos_theta.unsqueeze(-1) - normal_hat
        X_orthog[self.bc_edge_mask] = 0
        self.X_orthog = X_orthog.unsqueeze(dim=-1)

        # Create indexing masks between main and boundary edges
        # Step 1: Create a boolean tensor tracking which face of each edge is assigned.
        face_assigned = torch.zeros((self.n_edges, 2), dtype=torch.bool, device=self.device)
        face_assigned[self.tri_to_edge, self.tri_edge_signs] = True
        # Step 2: For each boundary edge, find which side (face) is not assigned, with the index (0 or 1) of the unset face
        assigned_boundary = face_assigned[self.bc_edge_mask]  # shape: (n_boundary_edges, 2)
        self.bc_edge_side = (~assigned_boundary).float().argmax(dim=1).int()
        self.bc_locations = torch.where(self.bc_edge_mask)[0].int()
        self._init_bc(bc_tags)
        c_print(f'_init_bc done', color="magenta")

        self._build_spm_face_grads()
        c_print(f'_build_spm_face_grads done', color="magenta")
        c_print(f'Complete init FVMEdgeInfo', color="magenta")


    def clear_temp(self):
        del self.edge_dists_bc, self.cell_dist_proj, self.edge_to_tri_main, self.dirich_val, self.neumann_val, self.cell_disps
        del self.dirich_mask, self.neumann_mask , self.bc_edge_mask

        torch.cuda.empty_cache()
        c_print(f'Deleted temp variables', color="magenta")


    def _init_bc(self, bc_tags: dict[int, Edge]):
        self.n_edges_bc = self.bc_edge_mask.sum().item()

        bc_type_str = []
        dirich_mask, neumann_mask = [], []
        dirich_val, neumann_val = [], []
        farfield_mask, inlet_mask = [], []
        for bc_idx, e_type in bc_tags.items():
            dirich_mask.append(e_type.dirichlet())
            neumann_mask.append(e_type.neumann())
            dirich_val.append(e_type.U)
            neumann_val.append(e_type.dUdn)

            farfield_mask.append(e_type.farfield())
            inlet_mask.append(e_type.inlet())

            bc_type_str.append(e_type.tag)

        self.bc_type_str = bc_type_str
        self.dirich_mask, self.neumann_mask = torch.tensor(dirich_mask, device=self.device), torch.tensor(neumann_mask, device=self.device)
        dirich_val, neumann_val = torch.tensor(dirich_val, dtype=torch.float32, device=self.device), torch.tensor(neumann_val, dtype=torch.float32, device=self.device)
        self.dirich_val = dirich_val[self.dirich_mask]
        self.neumann_val = neumann_val[self.neumann_mask]

        # Farfield and inlet boundary conditions
        farfield_mask = torch.tensor(farfield_mask, device=self.device)     # shape = (n_edges_bc)
        use_farfield = torch.any(farfield_mask).item()

        inlet_mask = torch.tensor(inlet_mask, device=self.device)     # shape = (n_edges_bc)
        use_inlet = torch.any(inlet_mask).item()

        assert self.dirich_mask.shape[0] == self.bc_edge_mask.sum(), f'Wrong mask shape'
        assert self.neumann_mask.shape[0] == self.bc_edge_mask.sum(), f'Wrong mask shape'
        assert farfield_mask.shape[0] == self.bc_edge_mask.sum(), f'Wrong mask shape'
        assert inlet_mask.shape[0] == self.bc_edge_mask.sum(), f'Wrong mask shape'

        self.boundary_setter = BoundarySetter(self)
        if use_farfield:
            exit_cell2edge = self.edge_to_tri_bc[farfield_mask]
            # Normal for farfield edges, pointing outward always
            ff_edge_sign = 2 * (self.bc_edge_side[farfield_mask] - 0.5)
            ff_edge_normals = self.normals_hat.squeeze()[self.bc_edge_mask][farfield_mask]
            ff_edge_normals = ff_edge_normals * ff_edge_sign.unsqueeze(-1)
            self.boundary_setter.init_farfield(self.cfg, farfield_mask, exit_cell2edge, ff_edge_normals)
        if use_inlet:
            inlet_cell2edge = self.edge_to_tri_bc[inlet_mask]
            # Directed normal for inlet edges. Points outward always
            inlet_edge_sign = -2 * (self.bc_edge_side[inlet_mask] - 0.5)
            inlet_edge_normals = self.normals_hat.squeeze()[self.bc_edge_mask][inlet_mask]
            inlet_edge_normals = inlet_edge_normals * inlet_edge_sign.unsqueeze(-1)
            self.boundary_setter.init_inlet(self.cfg, inlet_mask, inlet_cell2edge, inlet_edge_normals)

    def precompute_shared(self, Us, dt):
        """ Precompute shared values that are used multiple times later.
            Us.shape = [n_cells, n_component] """
        U_face_bc = self._bc_face_vals(Us, dt)      # shape = [n_edges_bc, n_comp]
        Us_cell_face = torch.cat([Us, U_face_bc])        # shape = [n_cells + n_edges_bc, n_comp]
        cell_grads = self._cell_grads(Us_cell_face) # shape = [n_cells, 2, n_comp]
        grad_faces_n = self._face_grads(Us)        # shape = [n_faces, n_comp]

        # Compute limited face values,
        Us_face, phi_lim = self._limit_face_vals(Us, U_face_bc, cell_grads)   # Us_face.shape = [n_cells, 3, n_comp], phi_lim.shape =  [n_cells, 1, n_comp]
        Us_face = Us_face.view(3 * self.n_cells, self.n_comp)
        cell_grads = cell_grads * phi_lim

        # Face and cell gradients of velocity and temperature
        grad_F_dn = grad_faces_n[:, [0, 1, 3]]   # shape = [n_faces, 3]
        grad_F = cell_grads[:, :, [0, 1, 3]].reshape(self.n_cells, 6)      # shape = [n_cells, {dx, dy} * {vx, vy, T}]
        grad_F_flat = torch.repeat_interleave(grad_F, 3, dim=0, output_size=3*self.n_cells) # [dvx/dx, dvy/dx, dvx/dy, dvy/dy, dT/dx, dT/dy]
        grad_F_bc = grad_F[self.edge_to_tri_bc]

        # Limited cell divergence
        div_V = cell_grads[:, 0, 0] + cell_grads[:, 1, 1]        # shape = [n_cells]
        div_V_bc = div_V[self.edge_to_tri_bc].unsqueeze(-1)     # shape = [n_bc_edges, 1]
        div_V = torch.repeat_interleave(div_V, 3, output_size=3*self.n_cells).unsqueeze(-1)            # shape = [3*n_cells, 1]

        # Prepare projection from cell to edges - (slow step so vectorise over all components)
        cell_values = torch.cat([Us_face, div_V, grad_F_flat], dim=-1)        # shape = [3*n_cells, n_comp+7]
        cell_values_bc = torch.cat([U_face_bc, div_V_bc, grad_F_bc], dim=1)      # shape = [n_edges_bc, n_comp+7]

        # Project to left and right face values
        U_face_all = torch.empty((self.n_edges, 2, self.n_comp + 7), device=self.device)    # [momx, momy, rho, Q, div_V, face_grad X 4]
        U_face_all[self.tri_to_edge, self.tri_edge_signs] = cell_values
        U_face_all[self.bc_locations, self.bc_edge_side] = cell_values_bc

        # Decompose components back
        self.Vs_faces = U_face_all[:, :, :2]  # shape = [n_edges, edges=2, n_comp=2]
        self.rho_faces = U_face_all[:, :, [2]]  # shape = [n_edges, edges=2, dims=1]
        self.T_faces = U_face_all[:, :, [3]]    # shape = [n_edges, edges=2, dims=1]
        self.div_V_faces = U_face_all[:, :, 4]  # shape = [n_edges, edges=2]

        # Conserved quantities
        self.mom_faces = self.Vs_faces * self.rho_faces  # shape = [n_edges, edges=2, dims=2]
        self.Q_faces = self.rho_faces * (1/2  * self.Vs_faces.norm(dim=-1, keepdim=True) ** 2 + self.C_v * self.T_faces)
        self.phi = (self.Vs_faces * self.normals.unsqueeze(1)).sum(dim=-1) # shape = [n_edges, edges=2, ]

        # Non-orthogonal correction for face gradient. Assume lstsq gradient is mean of left and right cell.
        grad_F_lstsq = U_face_all[:, :, 5:11].view(self.n_edges, 2, 2, 3)   # shape = [n_edges, edges=2, {x, y}, {vx, vy, T}]
        grad_F_lstsq = grad_F_lstsq.mean(dim=1)   # shape = [n_edges, {x, y}, {vx, vy, T}]
        dFdn_correct = grad_F_dn - (grad_F_lstsq * self.X_orthog).sum(dim=1)      # shape = [n_edges, 3]
        # Replace normal part of gradient with face gradient
        grad_F_dot_n = (grad_F_lstsq * self.normals_hat).sum(dim=1, keepdim=True)       # shape = [n_edges, 1, 3]
        grad_F_para = (dFdn_correct.unsqueeze(dim=1) - grad_F_dot_n) * self.normals_hat       # shape = [n_edges, 2, 3]
        grad_F = grad_F_lstsq + grad_F_para             # shape = [n_edges, 2, 3]
        self.grad_V = grad_F[:, :, :2]
        self.grad_T_n = dFdn_correct[:, 2]      # shape = [n_edges]

        self.cell_grads = cell_grads

    def _limit_face_vals(self, Us, U_face_bc, cell_grads):
        """ Limited B-J scheme for cell to face interpolation.
        """
        U_cent = Us.unsqueeze(1)        # shape = [n_cells, 1, n_comp]
        Us_cell_face = torch.cat([Us, U_face_bc])        # shape = [n_cells + n_edges_bc, n_comp]
        Us_neigh = Us_cell_face[self.neigh_combine]  # shape = [n_cells, neigh=3, n_comp]

        # Uncorrected update
        grads = cell_grads.unsqueeze(1)     # shape = [n_cells, 1, dims=2, n_comp]
        dU = (grads * self.cent_to_edge_disp).sum(dim=2)  # shape = [n_cells, neigh=3, n_comp]

        # Select limiting neighbor values and compute gradient limiter
        U_cent_neigh = torch.cat([U_cent, Us_neigh], dim=1)             # shape = [n_cells, 4, n_comp]
        U_upper = torch.max(U_cent_neigh,dim=1, keepdim=True)[0] - U_cent      # shape = [n_cells, 1, n_comp]
        U_lower = torch.min(U_cent_neigh,dim=1, keepdim=True)[0] - U_cent

        numerator = torch.where(dU > 0, U_upper, U_lower) # shape = [n_cells, neigh=3, n_comp]
        phi_lim = self.slope_limiter.limit(numerator, dU)           # shape = [n_cells, neigh=3, n_comp]
        Us_face = U_cent + phi_lim * dU      # shape = [n_cells, neigh=3, n_comp]

        return Us_face, phi_lim

    def _bc_face_vals(self, Us, dt):
        """ Boundary face values.
            Us.shape = [n_cells, n_comp]
            return.shape: [n_edges_bc, n_comp]

         """
        # U_face = torch.empty((self.n_edges_bc, self.n_comp), device=self.device)
        # # Boundary edges
        # u_centroid_bc = Us[self.edge_to_tri_bc] # shape = [n_bc_edges, n_comp]
        # # Dirichlet
        # U_face[self.dirich_mask] = self.dirich_val
        # # Neumann
        # U_cent_bc_neum = u_centroid_bc[self.neumann_mask]        # shape = [n_neum_edges]
        # U_face_neum = U_cent_bc_neum + self.neumann_val * self.edge_dists_bc[self.neumann_mask]
        # U_face[self.neumann_mask] = U_face_neum

        U_face = self.boundary_setter.set_face_values(Us, self.cell_grads, dt)


        return  U_face

    def _cell_grads(self, Us_cell_face):
        """ Vectorised gradient computation
            Gradient = G @ (u_neigh - u_cell)
            Us_cell_face.shape = (n_cells+n_edges_bc, N_component)
            Returns: Gradient matrix of shape (n_cells, 2, N_component)
        """
        combined_grad = torch.sparse.mm(self.G_mats, Us_cell_face)  # combined_grad.shape == [2 * n_cells, N_component]
        cell_grads = combined_grad.view(2, self.n_cells, -1).permute(1, 0, 2)    # shape = [n_cells, 2, N_component]
        return cell_grads

    def _face_grads(self, Us):
        """ n . grad(U) on faces.
            Us.shape = (n_cells, N_component)
            Returns: shape = [n_edges, N_component]

            Non-orthogonal correction: n . grad(U)_f = C du + (n - C d) . grad(U)_f
            NOTE: Must be called after _cell_grads() to ensure up to date cell_grads
        """
        #
        # On faces
        # U_centroid = Us[self.edge_to_tri_main]      # shape = [n_edges, 2, N_component]
        # dU = U_centroid[:, 1] - U_centroid[:, 0]        # shape = [n_edges, N_component]
        # print(f'{U_centroid[494] = }')
        # print(f'{dU[494] = }')
        #
        """ NEW, non-orthogonal correction """
        # normals_hat = self.normals_main / torch.norm(self.normals_main, dim=1).unsqueeze(-1)
        # C = 1 / (normals_hat * self.cell_disps).sum(dim=1, keepdim=True)
        #
        # grad_impl = C * dU
        # """ NEw - cell corrected """
        # # Interpolate cell gradients to face
        # grad_U_main = self.cell_grads[self.edge_to_tri_main]  # shape = [n_edges, 2, d_dims=2, N_component]
        # w = self.edge_to_tri_w.unsqueeze(-1).unsqueeze(-1)  # shape: [n_edges, 2, 1, 1]
        # grad_U_face = (w * grad_U_main).sum(dim=1)  # shape: [n_edges, 2, n_component]
        #
        # corr_expl = (normals_hat - C * self.cell_disps)#.unsqueeze(1) * grad_U_face
        # grad_expl = (corr_expl.unsqueeze(-1) * grad_U_face).sum(dim=1)       # shape = [n_edges, N_component]
        #
        # dUdn_face_m_new = grad_impl + grad_expl
        # dUdn_face[~self.bc_edge_mask] = dUdn_face_m_new
        # # # #9


        """ OLD """
        # U_centroid = Us[self.edge_to_tri_main]      # shape = [n_edges, 2, N_component]
        # dU = U_centroid[:, 1] - U_centroid[:, 0]        # shape = [n_edges, N_component]
        # dUdn_face = torch.empty((self.n_edges, self.n_comp), device=self.device)
        # dUdn_face_m = dU / self.cell_dist.unsqueeze(-1)       # shape = [n_edges, N_component]
        # # print(f'{dUdn_face.shape = }, {self.bc_edge_mask.shape = }')
        # dUdn_face[~self.bc_edge_mask] = dUdn_face_m
        # #
        # # On boundary. Either u or du/dn is given
        # # Neumann: n.grad(u) = du/dn
        # dUdn_face[self.neum_all] = self.neumann_val
        #
        # # Dirichlet: n.grad(u) = 1/d * (u_bc - u)
        # u_centroid_bc = Us[self.edge_to_tri_bc]  # shape = [n_bc_edges, N_component]
        # U_cent_bc_dir = u_centroid_bc[self.dirich_mask]     # shape = [n_dirich_edges]
        # edge_dists = self.edge_dists_bc[self.dirich_mask]    # shape = [n_dirich_edges]
        # dudn_face_bc_dir = (self.dirich_val - U_cent_bc_dir) / edge_dists
        # dUdn_face[self.dirich_all] = dudn_face_bc_dir

        # """ SPARSE"""
        Us_flat = Us.flatten()
        dUdn_face_flat = torch.mv(self.A_face_grad, Us_flat) + self.b_face_grad
        dUdn_face = dUdn_face_flat.view(self.n_edges, self.n_comp)

        return dUdn_face

    def _build_spm_face_grads(self):
        """Build sparse operators to compute normal face gradients from cell values.

        Constructs the sparse matrix/operator and offset vector used to compute
        dU/dn on each face from cell-centered values and boundary conditions.
        The routine:
          - builds A_face for interior faces (differences / distances),
          - lifts it to component-wise form,
          - builds boundary contributions for Dirichlet and Neumann faces,
          - assembles the final lifted sparse operator self.A_face_grad and offset self.b_face_grad.

        Outputs:
          - self.A_face_grad : sparse matrix for mapping flattened cell values to face normal gradients
                              (shape corresponds to [n_edges * n_comp, n_cells * n_comp])
          - self.b_face_grad : offset vector for boundary contributions (shape [n_edges * n_comp])

        Implementation preserves existing behavior and shapes used elsewhere in the class.
        """
        n_edges = self.edge_to_tri_main.shape[0]  # number of faces (edges)
        n_cells = self.n_cells
        n_bc = self.n_edges_bc  # number of boundary edges
        n_comp = self.n_comp  # number of components

        """ Main faces"""
        # For each face, we have two contributions.
        # Create row indices: each face i gives two rows (one per contribution).
        rows = torch.arange(n_edges, device=self.device).repeat_interleave(2)

        # Flatten the cell indices from self.edge_to_tri_main.
        cols = self.edge_to_tri_main.reshape(-1)

        # We want, for each face i, to assign:
        #   - For the first cell (cols entry from self.edge_to_tri_main[i, 0]): -1/d_i
        #   - For the second cell (cols entry from self.edge_to_tri_main[i, 1]): +1/d_i
        #
        # To do this, we first repeat the cell distances for each face:
        cell_dist_rep = self.cell_dist_proj[~self.bc_edge_mask].repeat_interleave(2)  # shape (2*n_edges,)
        # Create a vector with the appropriate signs: first -1 then +1 for each face.
        face_signs = torch.tensor([-1, 1], device=self.device, dtype=torch.float32).repeat(n_edges)
        # Now compute the nonzero values.
        vals = face_signs / cell_dist_rep

        # Build the sparse matrix A_face.
        A_face = torch.sparse_coo_tensor(
            torch.stack([rows, cols]),
            vals,
            size=(n_edges, n_cells))

        A_face_grad_main = lift_sparse_matrix(A_face, self.n_comp)

        """ Boundary faces """
        # --- Prepare flattened indices for boundary rows ---
        # Each boundary edge gives n_comp rows.
        bc_rows = torch.arange(n_bc, device=self.device).unsqueeze(1).expand(n_bc, n_comp).reshape(-1)
        # Also record the component index for each entry.
        comp_idx = torch.arange(n_comp, device=self.device).unsqueeze(0).expand(n_bc, n_comp).reshape(-1)

        # Flatten the condition masks.
        dirich_mask_flat = self.dirich_mask.reshape(-1)  # True where gradient BC is given as Dirichlet
        neum_mask_flat = self.neumann_mask.reshape(-1)  # True where gradient BC is Neumann
        # Identify the flattened rows corresponding to Dirichlet gradient entries, with edge index .
        dirich_rows = torch.nonzero(dirich_mask_flat, as_tuple=False).squeeze(1)
        dirich_edge_idx = bc_rows[dirich_rows]
        # Neumann
        neum_rows_all = torch.nonzero(neum_mask_flat, as_tuple=False).squeeze(1)
        neum_comp = comp_idx[neum_rows_all]

        # --- Build the sparse matrix A_grad_bc ---
        # For Dirichlet entries, we want:
        #   coefficient = -1 / edge_dists_bc[edge]  at the column corresponding to
        #   cell = self.edge_to_tri_bc[edge] and component c.
        # For boundary edge i and component c, the cell value is at: col = self.edge_to_tri_bc[i] * n_comp + c
        cols = self.edge_to_tri_bc[dirich_edge_idx] * n_comp + comp_idx[dirich_rows]
        # The coefficient for each Dirichlet entry is -1/edge_dists_bc (for the corresponding boundary edge).
        vals = -1.0 / self.edge_dists_bc[dirich_edge_idx, 0]  # shape: (n_dirich_entries,)
        # The size of the lifted matrix is (n_bc*n_comp, n_cells*n_comp)
        size_grad = (n_bc * n_comp, n_cells * n_comp)
        indices = torch.stack([dirich_rows, cols], dim=0)
        A_grad_bc = torch.sparse_coo_tensor(indices, vals, size=size_grad)

        # --- Build the offset vector b_grad_bc ---
        # For Dirichlet entries:
        #   b = (dirich_val)/edge_dists_bc (applied componentwise)
        # For Neumann entries:
        #   b = neumann_val (applied componentwise)
        b_grad = torch.zeros(n_bc * n_comp, dtype=torch.float32, device=self.device)
        # Handle Dirichlet gradient entries:
        b_grad[dirich_rows] = self.dirich_val / self.edge_dists_bc[dirich_edge_idx, 0]
        # Handle Neumann entries:
        b_grad[neum_rows_all] = self.neumann_val[neum_comp]

        # print("Starting combine_edge_operators")
        self.A_face_grad, self.b_face_grad = combine_edge_operators(A_face_grad_main, A_grad_bc, b_grad, self.bc_edge_mask, self.n_edges, self.n_cells, self.n_comp, self.device)

    def _build_dUf_dUc(self):
        """ Build sparse matrix for dU_f/dU_c - gradient of face val w.r.t. cell values.
            U_cell = [interleave(mom_x | mom_y | rho)], shape = [3*n_cell]
            U_face = A(U_cell)              shape = [3*n_edges, 2], second dim is Left / Right side of face.
            d(U_f_i, L/R)/d(U_c_j) = :
                                0 if cell_j doesnt have face_i  - Or computing wrong mom_x, mom_y, rho component.
                                1 if face_i is on cell_j and cell_j is on L/R side.
            Returns.shape = 2 * [3*n_edges, 3*n_cells]
        """

        n_edges, n_cells = self.n_edges, self.n_cells
        rows_left = []
        cols_left = []
        rows_right = []
        cols_right = []
        # Loop over each edge and each component
        for e, adj_cell in self.mesh.edge_to_tri.items():
            if adj_cell.numel() == 2:  # Only interior edges have gradient
                for comp in range(self.n_comp):
                    i = 3 * e + comp
                    # Left side: contribution from the left cell.
                    rows_left.append(i)
                    cols_left.append(3 * int(adj_cell[0].item()) + comp)
                    # Right side: contribution from the right cell.
                    rows_right.append(i)
                    cols_right.append(3 * int(adj_cell[1].item()) + comp)

        # Convert indices to tensors.
        rows_left = torch.tensor(rows_left, dtype=torch.long)
        cols_left = torch.tensor(cols_left, dtype=torch.long)
        rows_right = torch.tensor(rows_right, dtype=torch.long)
        cols_right = torch.tensor(cols_right, dtype=torch.long)

        # Create the constant skeleton sparse matrices (values are 1).
        indices_left = torch.stack([rows_left, cols_left], dim=0)
        indices_right = torch.stack([rows_right, cols_right], dim=0)
        ones_left = torch.ones(rows_left.size(0))
        ones_right = torch.ones(rows_right.size(0))

        dUfL_dUc = torch.sparse_coo_tensor(indices_left, ones_left, size=(3 * n_edges, 3 * n_cells), dtype=torch.int16)
        dUfR_dUc = torch.sparse_coo_tensor(indices_right, ones_right, size=(3 * n_edges, 3 * n_cells), dtype=torch.int16)

        #dUfL_dUc, dUfR_dUc = dUfL_dUc.to_sparse_csr(), dUfR_dUc.to_sparse_csr()
        dUfL_dUc, dUfR_dUc = dUfL_dUc.cuda(non_blocking=True), dUfR_dUc.cuda(non_blocking=True)
        return dUfL_dUc, dUfR_dUc

