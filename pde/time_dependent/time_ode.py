import torch
from cprint import c_print

from pde.graph_grid.graph_store import Point,  Deriv, T_Point
from pde.graph_grid.graph_store import P_TimeTypes as TT
from pde.config import Config
from pde.mesh_generation.generate_mesh import gen_mesh_time
from pde.graph_grid.graph_utils import plot_points, plot_interp_graph
from pde.time_dependent.time_cfg import ConfigTime
from pde.time_dependent.U_time_graph import UGraphTime, UTemp

def mesh_graph(cfg):
    N_comp = 1

    xmin, xmax = 0, 3
    ymin, ymax = 0.0, 1.5
    Xs, p_tags = gen_mesh_time(xmin, xmax, ymin, ymax, areas=[6e-3, 10e-3])
    Xs = torch.from_numpy(Xs).float()
    c_print(f'Number of mesh points: {len(Xs)}', "green")

    # Set up time-graph
    setup_T = []
    for i, (X, tag) in enumerate(zip(Xs, p_tags)):
        if tag == "Wall" or tag == "Left" or tag == "Right":
            value = [0 for _ in range(N_comp)]
            setup_T.append(T_Point([TT.FIXED], X, init_val=value))
        elif tag == "Normal":
            x, y = X
            a, b = x-1.5, y-0.75
            if a**2 + b**2 < 0.25:
                value = [1]
            else:
                value = [0]
            setup_T.append(T_Point([TT.NORMAL], X, init_val=value))
        else:
            raise ValueError(f"Unknown tag {tag}")

    setup_T = {i: point for i, point in enumerate(setup_T)}
    u_graph_time = UGraphTime(setup_T, N_component=N_comp, grad_acc=2, device=cfg.DEVICE)
    # plot_points(u_graph_time._Xs, u_graph_time.dirich_mask[:, 0], title="grad mask")
    with open("./save_u_graph_T2.pth", "wb") as f:
        torch.save(u_graph_time, f)
    return u_graph_time


def load_graph(cfg):
    u_graph_T = torch.load("save_u_graph_T2.pth", weights_only=False)
    return u_graph_T

class PDEFn:
    def __init__(self, u_graph_T: UGraphTime, cfg_in, cfg_T):
        self.u_graph_T = u_graph_T
        self.cfg_in = cfg_in
        self.cfg_T = cfg_T

        self.dt = cfg_T.dt

    def solve(self, t, step_no):
        us = self.u_graph_T.get_all_us_Xs()[0]

        grads = self.u_graph_T.get_grads()
        us_t = grads[(0, 0)]
        # dudt = - laplacian(u)
        laplacian = grads[(2, 0)] + grads[(0, 2)]

        #u_t+1 = u_t + dt * dudt
        u_t_1 = us_t + self.dt * laplacian


        self.u_graph_T.set_grid(u_t_1)



class TimePDEBase:
    """ Have a main PDE U_graph that is updated with every t. For update:
        1) Clone U_graph.
        1.1) Clone U_graph if we want state to be saved for later
        2) Solve PDE with U_graph
        3) Update time-PDE with new values.

        Assume graph doesn't change so deriv calc and intermediate sparse caches can be kept.
        """
    cfg_T: ConfigTime
    cfg_in: Config

    PDE_timefn: PDEFn

    u_graph_main: UGraphTime
    u_saves: dict[int, UTemp]

    def __init__(self, u_graph_T: UGraphTime, cfg_T: ConfigTime, cfg_in: Config):
        """ u_graph_T: Time graph.
            u_graph_PDE: Graph for internal PDE solver.
        """
        self.u_graph_T = u_graph_T
        self.cfg_T = cfg_T
        self.cfg_in = cfg_in
        self.u_saves = {}
        self.Xs = None
        self.PDE_timefn = PDEFn(u_graph_T, cfg_in, cfg_T) #ExplicitNS(u_graph_T, u_graph_PDE, cfg_in, cfg_T)

        self.device = "cuda"
        self.dtype = torch.float32

    def solve(self):
        cfg_T = self.cfg_T

        self.Xs = self.u_graph_T.get_all_us_Xs()[1]
        self.u_saves[0] = self.u_graph_T.get_all_us_Xs()[0].clone()
        timesteps = torch.linspace(cfg_T.time_domain[0], cfg_T.time_domain[1], cfg_T.timesteps * cfg_T.substeps, dtype=self.dtype)
        for step_num, t in enumerate(timesteps):
            print(f'\n{step_num = }, t = {t.item():.3g}')

            # if step_num == 5:
            #     dirich_mask = self.u_graph_T.dirich_mask
            #     dirich_values = torch.zeros_like(self.u_graph_T._us)[dirich_mask]
            #     print(f'{dirich_values.shape = }')
            #     self.u_graph_T.set_bc(dirich_bc=dirich_values)

            self.PDE_timefn.solve(t, step_num)

            if step_num % cfg_T.substeps == 0:
                self.u_saves[step_num+1] = self.u_graph_T.get_all_us_Xs()[0].clone()

            if step_num == 50:
                break

        for step, us in self.u_saves.items():
            plot_interp_graph(self.Xs, us[:, 0], title=f"Vx Step {step}")


    def update_boundary(self):
        pass


def main():
    from pde.utils import setup_logging

    setup_logging(debug=False)

    cfg = Config()
    time_cfg= ConfigTime()
    c_print(f'{time_cfg.dt = }', color="bright_magenta")

    #u_g_T = load_graph(cfg)
    u_g_T = mesh_graph(cfg)

    time_pde = TimePDEBase(u_g_T, time_cfg, cfg)
    time_pde.solve()

    # saved_graphs = time_pde.u_saves
    # for t, graph in saved_graphs.items():
    #     us, Xs = graph.us, graph.Xs
    #     plot_interp_graph(Xs, us[:, 0], title=f"t={t :.4g}")


if __name__ == "__main__":
    main()