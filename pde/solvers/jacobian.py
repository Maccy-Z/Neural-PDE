import torch
import torch.func as func

from pde.config import JacMode
from pde.BaseU import UBase
from pde.cartesian_grid.U_grid import USplitGrid, UNormalGrid
from pde.cartesian_grid.PDE_Grad import PDEForward
from pde.utils import get_split_indices, clamp
from pde.BasePDEGrad import PDEFwdBase
from pde.graph_grid.PDE_Grad import PDEForward as PDEForwardGraph
from pde.utils_sparse import CSRSummer, CSRRowMultiplier, CSRTransposer, CsrBuilder, CSRConcatenator, plot_sparsity, CSRPermuter
from pde.graph_grid.U_graph import UGraph

def get_jac_calc(us_grid: UBase, pde_forward: PDEFwdBase, cfg):
    if cfg.jac_mode == JacMode.DENSE:
        jacob_calc = JacobCalc(us_grid, pde_forward)
    elif cfg.jac_mode == JacMode.GRAPH:
        jacob_calc = GraphJacobCalc(us_grid, pde_forward)
    elif cfg.jac_mode == JacMode.SPLIT:
        jacob_calc = SplitJacobCalc(us_grid, pde_forward, num_blocks=cfg.num_blocks)
    elif cfg.jac_mode == JacMode.SPARSE:
        jacob_calc = SparseJacobCalc(us_grid, pde_forward, num_blocks=cfg.num_blocks)
    else:
        raise ValueError(f"Invalid jac_mode {cfg.jac_mode = }. Must be 'dense' or 'split'")
    return jacob_calc

class JacobCalc:
    """
        Calculate Jacobian of PDE.
    """
    def __init__(self, sol_grid: UBase, pde_fwd:PDEFwdBase):
        self.sol_grid = sol_grid
        self.pde_fwd = pde_fwd

        self.device = sol_grid.device

    def jacobian(self, aux_input=None):
        """ Returns Jacobain of function. Return shape: [N_pde, N_u_grad].
            PDEs are indexed by pde_mask and Us are indexed by us_grad_mask. 2D coordinates are stacked into 1D here.
            Columns for each function. Row for each u value.
         """
        us, us_grad_mask, pde_mask = self.sol_grid.get_us_mask()

        subgrid = UNormalGrid(self.sol_grid, us_grad_mask, pde_mask)

        us_grad = subgrid.get_us_grad()
        jacobian, residuals = torch.func.jacfwd(self.pde_fwd.residuals, has_aux=True, argnums=0)(us_grad, subgrid)  # N equations, N+2 Us, jacob.shape: [N^d, N^d]
        # jacobian, residuals = torch.func.jacrev(self.pde_func.residuals, has_aux=True, argnums=0)(us_grad, self.sol_grid)  # N equations, N+2 Us, jacob.shape: [N^d, N^d]

        return jacobian, residuals

    def jacob_transpose(self):
        jacobian, _ = self.jacobian()
        return jacobian.T

    def residauls(self):
        raise NotImplementedError


class GraphJacobCalc(JacobCalc):
    """ Use pde_func's inbuilt method to calculate jacobian"""
    def __init__(self, u_graph: UGraph, pde_fwd:PDEForwardGraph):
        super().__init__(u_graph, pde_fwd)
        self.u_graph = u_graph
        self.N_pdes = u_graph.N_pdes
        self.N_us_grad = u_graph.N_us_grad
        self.N_component = u_graph.N_component
        self.N_deriv = u_graph.N_deriv

        self.pde_fwd = pde_fwd
        self.deriv_calc = u_graph.deriv_calc
        self.resid_jac_val = torch.func.vmap(torch.func.jacrev(self.pde_fwd.residuals, has_aux=True, argnums=0))

        # Precompute transforms with jacobian structure
        deriv_jac_list = self.u_graph.deriv_calc.jacobian()
        self.row_multipliers = [CSRRowMultiplier(spm, check_sparsity=True) for spm in deriv_jac_list]
        self.csr_summer = CSRSummer(deriv_jac_list, check_sparsity=True)

        dummy_jac = self.csr_summer.blank_csr()
        self.transposer = CSRTransposer(dummy_jac, check_sparsity=True)

        if u_graph.neumann_mode:
            self.deriv_calc_bc = u_graph.deriv_calc_bc
            self.concatenator = CSRConcatenator(dummy_jac, self.deriv_calc_bc.jac_mat)

            # dummy_jac_full = self.concatenator.blank_csr()
            # self.permuter = CSRPermuter(u_graph.row_perm, dummy_jac_full)

    @torch.no_grad()        # Gradient explicity handled.
    def jacobian(self, pde_aux_input=None):
        """
            Compute jacobian dR/dU = dR/dD * dD/dU.

            us_grad.shape = [N_u_grad]. Gradients of trained u values.
            dR/dU.shape = [N_pde, N_u_grad]
            dR/dD.shape = [N_pde, N_derivs]
            dD/dU.shape = [N_pde, N_derivs, N_u_grad]

            dR_i/dU_j = sum_k dR_i/dD_k * dD_ik/dU_j

            Vector derivatives are handled as batches of [N_comp, N_pde], then merged into a column-concatenated vector [N_comp*N_pde]. Components are grouped together.
        """
        us_all, _ = self.u_graph.get_all_us_Xs()  # Shape = [N_total, N_comp].
        _, Xs = self.u_graph.get_us_Xs_pde()  # Shape = [N_total, 2].

        # 1) Finite differences D. shape = [N_pde, N_derivs, N_components]
        grads_dict = self.deriv_calc.derivative(us_all)  # shape = [N_pde, N_comp]. Derivative removes boundary points.
        u_dus = torch.stack(list(grads_dict.values()), dim=1)    # shape = [N_pde, N_derivs, N_component]

        # 2) dD/dU. shape = [N_derivs, N_u_grad]
        dDdU = self.deriv_calc.jacobian() # shape = [N_derivs][N_pde_, N_u_grad_]

        # 3) dR/dD. shape = [N_pde*N_comp, N_derivs*N_comp] = [N_pde_, N_derivs_]
        dRdD, residuals = self.resid_jac_val(u_dus, Xs) if (pde_aux_input is None) else self.resid_jac_val(u_dus, Xs, pde_aux_input) # [N_pde, N_component, N_deriv, N_component]
                                                                                                                    # residuals.shape = [N_pde, N_component]
        dRdD = dRdD.permute(1, 0, 3, 2).reshape(self.N_pdes*self.N_component, (self.N_deriv+1)*self.N_component)   # [N_pde_, N_deriv_]
        residuals = residuals.T.reshape(self.N_pdes*self.N_component)    # [N_pde_]

        # 4.1) Take product over i: dD_ik/dU_j * dR_i/dD_k . shape = [N_deriv_][N_pde_, N_u_grad_]
        partials = []
        for d in range(self.N_component*(self.N_deriv+1)):
            prod = self.row_multipliers[d].mul(dDdU[d], dRdD[:, d])   # shape = [N_pde_, N_u_grad_]
            partials.append(prod)

        # 4.2) Sum over k: sum_k partials_ijk
        jacobian = self.csr_summer.sum(partials)

        if self.u_graph.neumann_mode:
            # 5.1) Neumann boundary conditions: R = grad_n(u) - constant
            bc_deriv_pred = self.deriv_calc_bc.derivative(us_all)  # shape = [N_bc_derivs, N_comp]
            bc_deriv_true = self.u_graph.deriv_val
            bc_residuals = bc_deriv_pred - bc_deriv_true
            # 5.2_ Neumann jacobian: dR/dD = 1, so select corresponding rows of jacobian.
            bc_deriv_jac = self.deriv_calc_bc.jac_mat       # shape = [N_bc_derivs_, N_u_grad_]

            # 6) Concatenate on jacobian and residuals
            # residuals = [p0_0, p1_0, ..., p0_1, p1_1, ..., b0_0, b0_1, ..., b0_1, b_1_1, ...]
            residuals = torch.cat([residuals, bc_residuals])        # shape = [N_pde_+N_bc_]
            jacobian = self.concatenator.cat(jacobian, bc_deriv_jac)  # shape = [N_pde_+N_bc_, N_total_]


        return jacobian, residuals

    def jacob_transpose(self):
        jacobian, _ = self.jacobian()
        return self.transposer.transpose(jacobian)

    def residuals(self, aux_input=None):
        us_all, _ = self.u_graph.get_all_us_Xs()  # Shape = [N_total].
        _, Xs = self.u_graph.get_us_Xs_pde()  # Shape = [N_total, 2].

        # 1) Finite differences D. shape = [N_pde, N_derivs]
        grads_dict = self.deriv_calc.derivative(us_all)  # shape = [N_pde]. Derivative removes boundary points.
        u_dus = torch.stack(list(grads_dict.values()), dim=-1).permute(0, 2, 1)    # shape = [N_pde, N_derivs, N_component]   # shape = [N_pde, N_derivs]

        if aux_input is None:
            residuals, _ = func.vmap(self.pde_fwd.residuals)(u_dus, Xs)
        else:
            residuals, _ = func.vmap(self.pde_fwd.residuals)(u_dus, Xs, aux_input)

        # 2) Neumann BCs
        if self.u_graph.neumann_mode:
            bc_deriv_pred = self.deriv_calc_bc.derivative(us_all)
            bc_deriv_true = self.u_graph.deriv_val
            bc_residuals = bc_deriv_pred - bc_deriv_true

            residuals = torch.cat([residuals.flatten(), bc_residuals])  # shape = [N_pde_+N_bc_]
        return residuals


class SplitJacobCalc(JacobCalc):
    """
        Compute Jacobian in split blocks for efficiency.
    """
    def __init__(self, sol_grid: UBase, pde_func:PDEForward, num_blocks: int,):
        self.sol_grid = sol_grid
        self.pde_func = pde_func
        self.device = sol_grid.device

        # Sparsity pattern of Jacobian and splitting. Always fixed
        self.jacob_shape = self.sol_grid.N_us_grad  # == len(torch.nonzero(us_grad_mask))
        self.num_blocks = num_blocks
        self.split_idxs = get_split_indices(self.jacob_shape, num_blocks)
        self.block_size = self.jacob_shape // self.num_blocks
        self.us_stride = (self.sol_grid.N[1] + 1).item() # Stride of us in grid. Second element of N is the number of points in x direction. Overestimate.

    def jacobian(self):
        # Build up Jacobian from sections of PDE and us for efficiency.
        jacob_shape = self.jacob_shape
        us, us_grad_mask, pde_mask = self.sol_grid.get_us_mask()

        # Rectangular blocks of Jacobian between [xmin, xmax] (pde) and [ymin, ymax] (us)
        # Make new masks for pde and us for each block.
        jacobian = torch.zeros((jacob_shape, jacob_shape), device=self.device)
        residuals = torch.zeros(jacob_shape, device=self.device)

        # Indices of non-zero elements in pde_mask and us_grad_mask
        pde_true_idx, us_grad_idx = self.sol_grid.mask_nonzero_idx()

        for xmin, xmax in self.split_idxs:      # (pde_min, pde_max)
            # Diagonal blocks of Jacobian
            ymin = clamp(xmin - self.us_stride, 0, jacob_shape)         # (us_min, us_max)
            ymax = clamp(xmin + self.block_size + self.us_stride, 0, jacob_shape)

            pde_slice = slice(xmin, xmax)
            us_slice = slice(ymin, ymax)

            # Subset of pde equations
            want_pde_idx = pde_true_idx[pde_slice]
            pde_submask = torch.zeros_like(pde_mask)
            pde_submask[want_pde_idx[:, 0], want_pde_idx[:, 1]] = True

            # Subset of us
            want_us_idx = us_grad_idx[us_slice]
            us_grad_submask = torch.zeros_like(us_grad_mask)
            us_grad_submask[want_us_idx[:, 0], want_us_idx[:, 1]] = True

            # Further clip region PDE is calculated to around pde_mask and us_grad_mask to avoid unnecessary calculations
            want_us_idx = want_us_idx.cpu() # Faster on CPU
            a, b = torch.min(want_us_idx, dim=0)[0], torch.max(want_us_idx, dim=0)[0]
            a, b = a - 1, b + 1  # Add one point of padding. This *should* always be enough depending on bc and pde_mask
            a = torch.clamp(a, min=0)
            us_region_mask = (slice(a[0], b[0] + 1), slice(a[1], b[1] + 1))

            subgrid = USplitGrid(self.sol_grid, us_region_mask, us_grad_submask, pde_submask)

            # Get Jacobian
            us_grad = subgrid.get_us_grad()  # us[region][grad_mask]
            jacob, resid = torch.func.jacfwd(self.pde_func.residuals, has_aux=True, argnums=0)(us_grad, subgrid)  # jacob.shape: [block_size, block_size+stride]

            # Fill in calculated parts
            jacobian[pde_slice, us_slice] = jacob
            residuals[pde_slice] = resid.flatten()

        return jacobian, residuals


class SparseJacobCalc(JacobCalc):
    """
        Compute Jacobian in split blocks for efficiency.
    """
    def __init__(self, sol_grid: UBase, pde_func:PDEForward, num_blocks: int,):
        self.sol_grid = sol_grid
        self.pde_func = pde_func
        self.device = sol_grid.device

        # Sparsity pattern of Jacobian and splitting. Always fixed
        self.jacob_shape = self.sol_grid.N_us_grad  # == len(torch.nonzero(us_grad_mask))
        self.num_blocks = num_blocks
        self.split_idxs = get_split_indices(self.jacob_shape, num_blocks)
        self.block_size = self.jacob_shape // self.num_blocks
        self.us_stride = (self.sol_grid.N[1] + 1).item() # Stride of us in grid. Second element of N is the number of points in x direction. Overestimate.

        self.csr_jac_builder = CsrBuilder(self.jacob_shape, self.jacob_shape, device=self.device)

    def jacobian(self):
        # Build up Jacobian from sections of PDE and us for efficiency.
        jacob_shape = self.jacob_shape
        us, us_grad_mask, pde_mask = self.sol_grid.get_us_mask()

        # Rectangular blocks of Jacobian between [xmin, xmax] (pde) and [ymin, ymax] (us)
        # Make new masks for pde and us for each block.
        residuals = torch.zeros(jacob_shape, device=self.device)

        # Indices of non-zero elements in pde_mask and us_grad_mask
        pde_true_idx, us_grad_idx = self.sol_grid.mask_nonzero_idx()

        for xmin, xmax in self.split_idxs:      # (pde_min, pde_max)
            # Diagonal blocks of Jacobian
            ymin = clamp(xmin - self.us_stride, 0, jacob_shape)         # (us_min, us_max)
            ymax = clamp(xmin + self.block_size + self.us_stride, 0, jacob_shape)

            pde_slice = slice(xmin, xmax)
            us_slice = slice(ymin, ymax)

            # Subset of pde equations
            want_pde_idx = pde_true_idx[pde_slice]
            pde_submask = torch.zeros_like(pde_mask)
            pde_submask[want_pde_idx[:, 0], want_pde_idx[:, 1]] = True

            # Subset of us
            want_us_idx = us_grad_idx[us_slice]
            us_grad_submask = torch.zeros_like(us_grad_mask)
            us_grad_submask[want_us_idx[:, 0], want_us_idx[:, 1]] = True

            # Further clip region PDE is calculated to around pde_mask and us_grad_mask to avoid unnecessary calculations
            want_us_idx = want_us_idx.cpu() # Faster on CPU
            a, b = torch.min(want_us_idx, dim=0)[0], torch.max(want_us_idx, dim=0)[0]
            a, b = a - 1, b + 1  # Add one point of padding. This *should* always be enough depending on bc and pde_mask
            a = torch.clamp(a, min=0)
            us_region_mask = (slice(a[0], b[0] + 1), slice(a[1], b[1] + 1))

            subgrid = USplitGrid(self.sol_grid, us_region_mask, us_grad_submask, pde_submask)

            # Get Jacobian
            us_grad = subgrid.get_us_grad()  # us[region][grad_mask]
            jacob, resid = torch.func.jacfwd(self.pde_func.residuals, has_aux=True, argnums=0)(us_grad, subgrid)  # jacob.shape: [block_size, block_size+stride]

            # Fill in calculated parts
            self.csr_jac_builder.add_block(jacob, xmin, ymin)
            residuals[pde_slice] = resid.flatten()

        jacobian = self.csr_jac_builder.build()

        self.csr_jac_builder.reset()
        return jacobian, residuals


