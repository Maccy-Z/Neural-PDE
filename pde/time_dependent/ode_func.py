import torch
from cprint import c_print

from pde.config import Config
from pde.time_dependent.U_time_graph import UGraphTime
from pde.time_dependent.time_cfg import ConfigTime
from pde.NeuralPDE_Graph import NeuralPDEGraph
from pde.graph_grid.U_graph import UGraph
from pde.graph_grid.graph_utils import plot_interp_graph
from pde.pdes.PDEs import PressureNS

class ExplicitNS:
    def __init__(self, u_graph_T: UGraphTime, u_graph_PDE: UGraph, cfg: Config, cfg_T: ConfigTime):
        self.cfg = cfg
        self.cfg_T = cfg_T
        self.device = cfg.DEVICE
        self.dtype = torch.float32
        self.u_graph_T = u_graph_T
        self.p_graph_PDE = u_graph_PDE

        self.p_graph_PDE.set_eval_deriv_calc(torch.ones_like(u_graph_PDE.pde_mask).bool())

        # Subgraph for intermediate state V_star
        self.v_star_graph = u_graph_T.get_subgraph(all_grads=True)
        self.dp_graph = u_graph_PDE.copy() # u_graph_PDE.get_subgraph() #

        # Solve pressure equation
        pde_fn_inner = PressureNS(cfg, device=cfg.DEVICE)

        self.p_solver = NeuralPDEGraph(pde_fn_inner, self.dp_graph, cfg)

        self.mu = 0.05
        self.rho = 1
        self.dt = cfg_T.dt

        # The PDE is solved on a different graph from time derivatives.
        self.P_pde_mask = u_graph_PDE.pde_mask
        self.P_updt_mask = u_graph_PDE.updt_mask
        self.V_star_mask = self.v_star_graph.pde_mask

        self.v_deriv_calc = u_graph_T.deriv_calc
        self.v_s_deriv_calc = self.v_star_graph.deriv_calc
        self.p_deriv_calc = u_graph_PDE.deriv_calc_eval

        self._set_grad_I()

    def _set_grad_I(self):
        indicator = 1 - self.p_graph_PDE.neumann_mask.float().unsqueeze(-1)
        grad_I = self.p_graph_PDE.deriv_calc_eval.derivative(indicator)
        grad_I_x = grad_I[(1, 0)][self.P_pde_mask]
        grad_I_y = grad_I[(0, 1)][self.P_pde_mask]
        self.grad_I = (grad_I_x, grad_I_y)

    def _v_star(self, dvdxs, d2vdx2s, dpdxs, vs):
        # Viscosity = mu laplacian(v)
        viscosity = self.mu * d2vdx2s.sum(dim=-2)     # shape = [N_us_grad, 2]
        # Convective = -(v . grad)v
        vs_expand = vs.unsqueeze(-1)        # shape = [N_us_grad, 2, 1]
        product = vs_expand * dvdxs         # shape = [N_us_grad, 2, 2]
        convective = - product.sum(dim=-1)    # shape = [N_us_grad, 2]
        # Pressure = -1/rho grad(p)
        pressure = -1 / self.rho * dpdxs
        # Uncorrected velocity update
        dv_star = convective + pressure + viscosity
        v_star = vs + dv_star * self.cfg_T.dt

        return v_star

    def _get_derivs(self, vs, ps):
        """ Compute derivatives of velocity and pressure. """
        u_dus = self.v_deriv_calc.derivative(vs)
        vs = u_dus[0, 0]  # shape = [N_us_grad, 2]. Without boundary points
        dvdxs = torch.stack([u_dus[(1, 0)], u_dus[(0, 1)]], dim=1)
        d2vdx2s = torch.stack([u_dus[(2, 0)], u_dus[(0, 2)]], dim=1)

        p_dps = self.p_deriv_calc.derivative(ps)
        ps = p_dps[(0, 0)][self.V_star_mask, -1]  # shape = [N_us_grad, 1]
        dpdxs = torch.stack([p_dps[(1, 0)], p_dps[(0, 1)]], dim=2)[self.V_star_mask, 0]   # shape [N_us_grad, 2]
        d2pdx2s = torch.stack([p_dps[(2, 0)], p_dps[(0, 2)]], dim=2)[self.V_star_mask, 0]
        return vs, dvdxs, d2vdx2s, ps, dpdxs, d2pdx2s

    def _compute_w(self, div_v_s, grad_dP):
        # Compute update factor: w = div_u (u) dot div_u(grad_p(dP)) / div_u(grad_p(dP))^2
        grad_dP[~self.V_star_mask] = 0
        grad2_dp = self.v_s_deriv_calc.derivative(grad_dP, get_orders=[(1, 0), (0, 1)])
        laplac_dp = grad2_dp[(1, 0)][..., 0] + grad2_dp[(0, 1)][..., 1]
        laplac_dp[~self.V_star_mask] = 0
        laplac_dp *= self.dt / self.rho
        w = torch.dot(div_v_s, laplac_dp) / torch.dot(laplac_dp, laplac_dp)
        return w

    def solve(self, t, step_num=0):
        """ u_dus: dict[degree, torch.Tensor]. shape = [N_deriv][N_us_grad, N_comp]
            Returns: new values of velocity and pressure at variable nodes at time t.
        """
        self.v_star_graph.reset()
        us_old, _ = self.u_graph_T.get_all_us_Xs()
        p_old, _X = self.p_graph_PDE.get_all_us_Xs()
        us_old, p_old = us_old.clone(), p_old.clone()
        p_old = p_old * .8
        Dp = torch.zeros_like(p_old)

        # Initialise v_star with bounadry conditions
        self.v_star_graph._us = self.u_graph_T._us.clone()
        self.v_star_graph.set_bc()

        # Compute momentum equation estimate
        vs, dvdxs, d2vdx2s, ps, dpdxs, _ = self._get_derivs(us_old, p_old)
        # V_star = V + dt * (1/mu laplacian(v) - (v . grad)v - 1/rho grad(p))
        v_star_updt = self._v_star(dvdxs, d2vdx2s, dpdxs, vs)

        # Solve pressure equation multiple times
        # P_n+1 = P_n + DP_i = P_n + sum_i dp_i
        iters = 1 if step_num >= 8 else 1
        for i in range(iters):
            self.dp_graph.reset()
            self.v_star_graph.set_grid(v_star_updt)
            self.v_star_graph.set_bc()

            # Pressure correction: laplacian(dP) = rho/dt div(v_star)
            v_star_grad = self.v_star_graph.get_grads(get_orders=[(1, 0), (0, 1)])
            div_v_s = v_star_grad[(1, 0)][..., 0] + v_star_grad[(0, 1)][..., 1]
            # Solve pressure equation
            div_v_s[~self.V_star_mask] = 0
            div_v_s_ = div_v_s * self.rho / self.cfg_T.dt
            div_v_s_ = div_v_s_[self.P_pde_mask]
            self.p_solver.forward_solve([div_v_s_, self.grad_I[0], self.grad_I[1]])
            dp_i = self.dp_graph.get_all_us_Xs()[0]   # shape = [N_ps, 1]

            # Update velocity: V_star_i+1 = V_star_i - w * dt/rho grad(dp_i)
            dP_grads = self.p_deriv_calc.derivative(dp_i, get_orders=[(1, 0), (0, 1)])
            grad_dP = torch.cat([dP_grads[(1, 0)], dP_grads[(0, 1)]], dim=1)
            # Compute update factor: w = div_u (u) dot div_u(grad_p(dP)) / div_u(grad_p(dP))^2
            w = self._compute_w(div_v_s, grad_dP) * 0 + 1
            # V_star_i+1 = V_star_i - w * dt/rho grad(dp_i)
            v_star_updt = v_star_updt - w * self.dt / self.rho * grad_dP[self.V_star_mask]

            # Update pressure:
            Dp += w * dp_i

            if i >= 1:
                p_orthog = dp_i - p_old / (p_old.norm() + 1e-4) * dp_i.norm()
                dPo_grads = self.p_deriv_calc.derivative(p_orthog, get_orders=[(1, 0), (0, 1)])
                grad_dPo = torch.cat([dPo_grads[(1, 0)], dPo_grads[(0, 1)]], dim=1)
                # Compute update factor: w = div_u (u) dot div_u(grad_p(dP)) / div_u(grad_p(dP))^2
                _w = self._compute_w(div_v_s, grad_dPo)
                # _w = -1
                v_star_updt = v_star_updt - _w * self.dt / self.rho * grad_dPo[self.V_star_mask]
                Dp += _w * p_orthog
                print(f'{_w = }, norm: {(_w * p_orthog).norm().item():.4g}')

                # Eval
                if step_num == 79:
                    plot_interp_graph(_X, p_old, title=f"p_old- Step {i}")
                    plot_interp_graph(_X, -w * p_orthog, title=f"P orthog - Step {i}")
                    plot_interp_graph(_X, Dp, title=f"dp i - Step {i}")


            self.v_star_graph.set_grid(v_star_updt)
            self.v_star_graph.set_bc()

            v_star_updt_ = self.v_star_graph.get_all_us_Xs()[0]
            v_updt_grad = self.v_s_deriv_calc.derivative(v_star_updt_, get_orders=[(1, 0), (0, 1)])
            div_v_s_updt = v_updt_grad[(1, 0)][..., 0] + v_updt_grad[(0, 1)][..., 1]
            div_v_s_updt[~self.V_star_mask] = 0
            div_v_s_updt = div_v_s_updt #* self.rho / self.cfg_T.dt
            print(f'{step_num}: Divergence: {div_v_s_updt.abs().mean().item():.4g}, w: {w.item():.4g}')

            # if step_num > 10:
            #     k = div_v_s_updt[self.V_star_mask].unsqueeze(-1)
            #     f_k = (k/10).abs().clamp(0, 1)
            #     v_star_updt = v_star_updt * (1 -  f_k)
            #
            #
            # self.v_star_graph.set_grid(v_star_updt)
            # self.v_star_graph.set_bc()
            #
            # v_star_updt_ = self.v_star_graph.get_all_us_Xs()[0]
            # v_updt_grad = self.v_s_deriv_calc.derivative(v_star_updt_, get_orders=[(1, 0), (0, 1)])
            # div_v_s_updt = v_updt_grad[(1, 0)][..., 0] + v_updt_grad[(0, 1)][..., 1]
            # div_v_s_updt[~self.V_star_mask] = 0
            # div_v_s_updt = div_v_s_updt #* self.rho / self.cfg_T.dt
            # print(f'{step_num}: Divergence: {div_v_s_updt.norm().item():.4g}, w: {w.item():.4g}')


        p_new = p_old + Dp
        p_new = p_new[self.P_updt_mask]
        self.p_graph_PDE.set_grid(p_new)

        self.v_star_graph.set_grid(v_star_updt)
        self.v_star_graph.set_bc()
        us_new = self.v_star_graph.get_all_us_Xs()[0]
        #us_new = [v_star_updt[:, 0], v_star_updt[:, 1]]

        """ Plotting """
        # torch.save({"updt_mask": self.P_updt_mask, "Xs": _X}, "graph.pth")
        if step_num==79:
            plot_graph = self.v_star_graph
            _X = self.v_star_graph.get_all_us_Xs()[1]

            # plot_interp_graph(_X, indicator.squeeze(), title=f"Neumann mask- Step {t}")
            # exit(9)
            # plot_interp_graph(_X, self.V_star_mask, title=f"PDE mask- Step {t}")
            # # Init divergence:
            # div_plot = torch.full_like(_X[:, 0], torch.nan)
            # div_plot[self.P_pde_mask] = -div_v_s_
            # plot_interp_graph(_X, div_plot, title=f"Initial divergence Step {t:.3g}")
            # # Laplacian new
            # resid = torch.full_like(_X[:, 0], torch.nan)
            # dP_grads = self.p_graph_PDE.get_grads()
            # laplacian_new = dP_grads["laplacian"].squeeze()
            # resid[self.P_pde_mask] = laplacian_new
            # resid = resid.abs()
            # plot_interp_graph(_X, resid, title=f"laplacian new- Step {t:.3g}")
            #
            # # Laplacian Old
            # resid = torch.full_like(_X[:, 0], torch.nan)
            # dP_grads = self.dp_graph.get_grads()
            # laplacian = dP_grads[(2, 0)].squeeze() + dP_grads[(0, 2)].squeeze()
            # resid[self.P_pde_mask] = laplacian
            # resid = resid.abs()
            # plot_interp_graph(_X, resid, title=f"laplacian old- Step {t:.3g}")

            # # P PDE residual
            # resid = torch.zeros(_X.shape[0], device=self.device)
            # P_resid = self.P_solver.newton_solver.residuals.abs()
            # resid[self.P_updt_mask] = P_resid
            # plot_interp_graph(_X, resid, title=f"PDE Residual- Step {t:.3g}")

            # self.v_star_graph.set_grid(v_star)
            # _u, _X = self.v_star_graph.get_all_us_Xs()
            # plot_interp_graph(_X, _u[:, 0], title=f"Vx star- Step {t}")
            # plot_interp_graph(_X, _u[:, 1], title=f"Vy star- Step {t}")

            # # Plotting P
            # self.p_graph_PDE.set_grid(p_new)
            # _p, _X = self.p_graph_PDE.get_all_us_Xs()
            # plot_interp_graph(_X, _p[:, 0], title=f"P new- Step {t}")
            # #
            # # P grad
            # dP_grads = self.p_graph_PDE.get_eval_grads()
            # grad_dP_ = torch.cat([dP_grads[(1, 0)], dP_grads[(0, 1)]], dim=1)
            # # grad_dP_[self.P_pde_mask] = grad_dP
            # plot_interp_graph(_X, grad_dP_[:, 0], title=f"grad dP_x- Step {t}")
            # plot_interp_graph(_X, grad_dP_[:, 1], title=f"grad dP_y- Step {t}")


            # plot_interp_graph(_X, div_v_s_est, title=f"Estimated divergence- Step {t}")
            # plot_interp_graph(_X, est_laplace - true_laplace, title=f"Residual laplacian- Step {t}")
            # #Plotting V
            plot_graph.set_grid(v_star_updt)
            _u, _ = plot_graph.get_all_us_Xs()
            plot_interp_graph(_X, _u[:, 0], title=f"Vx- Step {t:.3g}")
            plot_interp_graph(_X, _u[:, 1], title=f"Vy- Step {t:.3g}")

            # # Viscous force
            # plot_graph.set_grid(v_star_updt)
            # v_star_grad = plot_graph.get_grads()
            # div_v_s = v_star_grad[(0, 2)][..., 0] + v_star_grad[(2, 0)][..., 0]
            # div_v_s[~self.V_star_mask] = torch.nan
            # plot_interp_graph(_X, div_v_s, title=f"Viscosity X - Step {t}")

            # Final divergence
            plot_graph.set_grid(v_star_updt)
            v_star_grad = plot_graph.get_grads()
            div_v_s = v_star_grad[(1, 0)][..., 0] + v_star_grad[(0, 1)][..., 1]
            div_v_s[~self.V_star_mask] = torch.nan
            plot_interp_graph(_X, div_v_s, title=f"Final divergence- Step {t}")
            #
            exit(4)
        return us_new

class SemiExplNS:
    def __init__(self, u_graph_T: UGraphTime, u_graph_PDE: UGraph, cfg: Config, cfg_T: ConfigTime):
        self.cfg = cfg
        self.cfg_T = cfg_T
        self.device = cfg.DEVICE
        self.dtype = torch.float32
        self.u_graph_T = u_graph_T
        self.p_graph_PDE = u_graph_PDE

        self.p_graph_PDE.set_eval_deriv_calc(torch.ones_like(u_graph_PDE.pde_mask).bool())

        # Subgraph for intermediate state V_star
        self.v_star_graph = u_graph_T.get_subgraph(all_grads=True)
        self.dp_graph = u_graph_PDE.copy() # u_graph_PDE.get_subgraph() #

        # Solve pressure equation
        pde_fn_inner = PressureNS(cfg, device=cfg.DEVICE)

        self.p_solver = NeuralPDEGraph(pde_fn_inner, self.dp_graph, cfg)

        self.mu = 0.1
        self.rho = 1
        self.dt = cfg_T.dt

        # The PDE is solved on a different graph from time derivatives.
        self.P_pde_mask = u_graph_PDE.pde_mask
        self.P_updt_mask = u_graph_PDE.updt_mask
        self.V_star_mask = self.v_star_graph.pde_mask

        self.v_deriv_calc = u_graph_T.deriv_calc
        self.v_s_deriv_calc = self.v_star_graph.deriv_calc
        self.p_deriv_calc = u_graph_PDE.deriv_calc_eval

        self._set_grad_I()

    def _set_grad_I(self):
        indicator = 1 - self.p_graph_PDE.neumann_mask.float().unsqueeze(-1)
        grad_I = self.p_graph_PDE.deriv_calc_eval.derivative(indicator)
        grad_I_x = grad_I[(1, 0)][self.P_pde_mask]
        grad_I_y = grad_I[(0, 1)][self.P_pde_mask]
        self.grad_I = (grad_I_x, grad_I_y)

    def _viscosity(self, v_dvs):
        d2vdx2s = torch.stack([v_dvs[(2, 0)], v_dvs[(0, 2)]], dim=1)
        viscosity = self.mu * d2vdx2s.sum(dim=-2)  # shape = [N_us_grad, 2]
        return viscosity

    def _get_dus(self, vs):
        """ Compute derivatives of velocity on v_star_mask"""
        v_dvs = self.v_deriv_calc.derivative(vs)
        vs = v_dvs[0, 0]  # shape = [N_us_grad, 2]. Without boundary points
        dvdxs = torch.stack([v_dvs[(1, 0)], v_dvs[(0, 1)]], dim=1)
        d2vdx2s = torch.stack([v_dvs[(2, 0)], v_dvs[(0, 2)]], dim=1)
        return vs, dvdxs, d2vdx2s

    def _get_dps(self, ps):
        """ Compute derivatives of pressure on v_star_mask"""
        p_dps = self.p_deriv_calc.derivative(ps)
        ps = p_dps[(0, 0)][self.V_star_mask, -1]  # shape = [N_us_grad, 1]
        dpdxs = torch.stack([p_dps[(1, 0)], p_dps[(0, 1)]], dim=2)[self.V_star_mask, 0]   # shape [N_us_grad, 2]
        d2pdx2s = torch.stack([p_dps[(2, 0)], p_dps[(0, 2)]], dim=2)[self.V_star_mask, 0]

        return ps, dpdxs, d2pdx2s

    def _v_star(self, dvdxs, dpdxs, vs):
        # Convective = -(v . grad)v
        vs_expand = vs.unsqueeze(-1)        # shape = [N_us_grad, 2, 1]
        product = vs_expand * dvdxs         # shape = [N_us_grad, 2, 2]
        convective = - product.sum(dim=-1)    # shape = [N_us_grad, 2]
        # Pressure = -1/rho grad(p)
        pressure = -1 / self.rho * dpdxs
        # Uncorrected velocity update
        dv_star = pressure + convective
        v_star = vs + dv_star * self.dt
        return v_star

    def solve(self, t, step_num=0):
        """ u_dus: dict[degree, torch.Tensor]. shape = [N_deriv][N_us_grad, N_comp]
            Returns: new values of velocity and pressure at variable nodes at time t.
        """
        self.u_graph_T.set_bc()

        self.v_star_graph.reset()
        us_old, _ = self.u_graph_T.get_all_us_Xs()
        p_old, _X = self.p_graph_PDE.get_all_us_Xs()
        us_old, p_old = us_old.clone(), p_old.clone()
        p_old = p_old * 0.
        Dp = torch.zeros_like(p_old)

        p_deriv_calc = self.p_graph_PDE.deriv_calc_eval

        # Initialise v_star with bounadry conditions
        self.v_star_graph._us = self.u_graph_T._us.clone()

        # 1) Compute initial momentum equation estimate
        (vs, dvdxs, d2vdx2s), (ps, dpdxs, d2pdx2s) = self._get_dus(us_old), self._get_dps(p_old)
        # V_star = V + dt * ((v . grad)v + 1/rho grad(p))
        v_i = self._v_star(dvdxs, dpdxs, vs)

        # Solve implicit update with viscosity and pressure
        # P_n+1 = P_n + DP_i = P_n + sum_i dp_i
        v_0 = v_i.clone()
        sum_grad_dp = torch.zeros_like(dpdxs)
        iters = 3 if step_num < 2 else 1
        print()
        for i in range(iters):
            self.dp_graph.reset()
            self.v_star_graph.set_grid(v_i)
            self.v_star_graph.set_bc()

            # 2) u_hat_i+1 = u_0 + dt * viscosity(u_i) - dt/rho grad(p_i)
            v_i_grads = self.v_star_graph.get_grads()
            if step_num < 1:
                v_star = v_0 + self.dt * (self._viscosity(v_i_grads)[self.V_star_mask] - 1. / self.rho * sum_grad_dp)
            else:
                v_j0 = v_0 + self.dt * (- 1. / self.rho * sum_grad_dp)
                v_star = v_j0.clone()
                for j in range(20):
                    self.v_star_graph.set_grid(v_star)
                    v_star = self.v_star_graph.get_all_us_Xs()[0]
                    v_st_grads = self.v_s_deriv_calc.derivative(v_star, get_orders=[(2, 0), (0, 2)])
                    v_star = v_j0 + self.dt * self._viscosity(v_st_grads)[self.V_star_mask]

            print("MAX", v_star.max())

            self.v_star_graph.set_grid(v_star)
            self.v_star_graph.set_bc()

            # 3) Pressure correction: laplacian(dP) = rho/dt div(v_star)
            v_star_grad = self.v_star_graph.get_grads()
            div_v_s = v_star_grad[(1, 0)][..., 0] + v_star_grad[(0, 1)][..., 1]
            div_v_s[~self.V_star_mask] = 0
            div_v_s_ = div_v_s * self.rho / self.dt
            div_v_s_ = div_v_s_[self.P_pde_mask]
            self.p_solver.forward_solve([div_v_s_, self.grad_I[0], self.grad_I[1]])
            c_print(f'{div_v_s_.norm() = }', color='green')
            # 4) Update pressure:
            dp_i = self.dp_graph.get_all_us_Xs()[0]   # shape = [N_ps, 1]

            # 5) Update velocity: V_star_i+1 = V_star_i - w * dt/rho grad(dp_i)
            dPi_grads = p_deriv_calc.derivative(dp_i)
            grad_dPi = torch.cat([dPi_grads[(1, 0)], dPi_grads[(0, 1)]], dim=1)
            # 5.1) Compute update factor: w = div_u(u) dot div_u(grad_p(dP)) / div_u(grad_p(dP))^2
            grad_dPi[~self.V_star_mask] = 0
            grad2_dp = self.v_s_deriv_calc.derivative(grad_dPi)
            laplac_dp = grad2_dp[(1, 0)][..., 0] + grad2_dp[(0, 1)][..., 1]
            laplac_dp[~self.V_star_mask] = 0
            laplac_dp *= self.dt / self.rho
            w = torch.dot(div_v_s, laplac_dp) / torch.dot(laplac_dp, laplac_dp)
            # 5.2) Update pressure:
            dp_grads = p_deriv_calc.derivative(dp_i)
            grad_dp_i = torch.cat([dp_grads[(1, 0)], dp_grads[(0, 1)]], dim=1)[self.V_star_mask]
            sum_grad_dp += w * grad_dp_i
            Dp += w * dp_i
            # 5.3) Update velocity: V_star_i+1 = V_star_i - w * dt/rho grad(dp_i)
            v_i = v_star - w * self.dt / self.rho * grad_dp_i  #+ v_star_updt#[self.V_star_mask]

            # Eval
            self.v_star_graph.set_grid(v_i)
            self.v_star_graph.set_bc()

            v_star_grad = self.v_star_graph.get_grads()
            div_v_s = v_star_grad[(1, 0)][..., 0] + v_star_grad[(0, 1)][..., 1]
            div_v_s[~self.V_star_mask] = 0
            div_v_s_ = div_v_s * self.rho / self.dt
            div_v_s_ = div_v_s_[self.P_pde_mask]
            print(f'{step_num} {i}: {div_v_s_.norm() = }, {w = }')


        if step_num > 0:
            p_new = p_old + Dp
        else:
            p_new = p_old

        p_new = p_new[self.P_updt_mask]
        self.p_graph_PDE.set_grid(p_new)
        # _p = self.p_graph_PDE.get_all_us_Xs()[0]
        # plot_interp_graph(_X, _p, title=f"P new- Step {t}")

        us_new = [v_i[:, 0], v_i[:, 1]]

        """ Plotting """
        if step_num==10:
            plot_graph = self.v_star_graph
            _X = self.v_star_graph.get_all_us_Xs()[1]

            # self.v_star_graph.set_grid(v_star)
            # _u, _X = self.v_star_graph.get_all_us_Xs()
            # plot_interp_graph(_X, _u[:, 0], title=f"Vx star- Step {t}")
            # plot_interp_graph(_X, _u[:, 1], title=f"Vy star- Step {t}")

            # Plotting P
            self.p_graph_PDE.set_grid(p_new)
            _p, _X = self.p_graph_PDE.get_all_us_Xs()
            plot_interp_graph(_X, _p[:, 0], title=f"P new- Step {t}")
            # #
            # # P grad
            # dP_grads = self.p_graph_PDE.get_eval_grads()
            # grad_dP_ = torch.cat([dP_grads[(1, 0)], dP_grads[(0, 1)]], dim=1)
            # # grad_dP_[self.P_pde_mask] = grad_dP
            # plot_interp_graph(_X, grad_dP_[:, 0], title=f"grad dP_x- Step {t}")
            # plot_interp_graph(_X, grad_dP_[:, 1], title=f"grad dP_y- Step {t}")

            # #Plotting V
            plot_graph.set_grid(v_i)
            _u, _ = plot_graph.get_all_us_Xs()
            plot_interp_graph(_X, _u[:, 0], title=f"Vx- Step {t:.3g}")
            plot_interp_graph(_X, _u[:, 1], title=f"Vy- Step {t:.3g}")

            # Viscous force
            plot_graph.set_grid(v_i)
            v_star_grad = plot_graph.get_grads()
            visc = self._viscosity(v_star_grad)#v_star_grad[(0, 2)][..., 0] + v_star_grad[(2, 0)][..., 0]
            visc[~self.V_star_mask] = torch.nan
            plot_interp_graph(_X, visc[:, 0], title=f"Viscosity X - Step {t}")

            # Convection
            plot_graph.set_grid(v_i)
            v_star_grad = plot_graph.get_grads()
            dvdxs = torch.stack([v_star_grad[(1, 0)], v_star_grad[(0, 1)]], dim=1)
            vs = plot_graph.get_all_us_Xs()[0]
            vs_expand = vs.unsqueeze(-1)  # shape = [N_us_grad, 2, 1]
            product = vs_expand * dvdxs  # shape = [N_us_grad, 2, 2]
            convective = - product.sum(dim=-1)  # shape = [N_us_grad, 2]
            convective[~self.V_star_mask] = torch.nan
            plot_interp_graph(_X, convective[:, 0], title=f"Convection X - Step {t}")

            # Final divergence
            plot_graph.set_grid(v_i)
            v_star_grad = plot_graph.get_grads()
            div_v_s = v_star_grad[(1, 0)][..., 0] + v_star_grad[(0, 1)][..., 1]
            div_v_s[~self.V_star_mask] = torch.nan
            plot_interp_graph(_X, div_v_s, title=f"Final divergence- Step {t}")
            #
            exit(4)
        return us_new

