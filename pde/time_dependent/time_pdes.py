import torch
from cprint import c_print

from pde.graph_grid.graph_store import Point,  Deriv, T_Point
from pde.graph_grid.graph_store import P_TimeTypes as TT, P_Types as T
from pde.config import Config
from pde.mesh_generation.generate_mesh import gen_mesh_time
from pde.graph_grid.U_graph import UGraph
from pde.graph_grid.graph_utils import plot_points, plot_interp_graph
from pde.time_dependent.ode_func import ExplicitNS, SemiExplNS
from pde.time_dependent.time_cfg import ConfigTime
from pde.time_dependent.U_time_graph import UGraphTime, UTemp

def mesh_graph(cfg):
    N_comp = 2

    xmin, xmax = 0, 3
    ymin, ymax = 0.0, 1.5
    Xs, p_tags = gen_mesh_time(xmin, xmax, ymin, ymax, areas=[2e-3, 5e-3])
    Xs = torch.from_numpy(Xs).float()
    c_print(f'Number of mesh points: {len(Xs)}', "green")

    # Set up time-graph
    setup_T = []
    for i, (X, tag) in enumerate(zip(Xs, p_tags)):
        if tag == "Wall":
            value = [0 for _ in range(N_comp)]
            setup_T.append(T_Point([TT.FIXED, TT.FIXED], X, init_val=value))
        elif tag == "Left": # or tag == "Left_extra":
            y = X[1]
            if y < 0.2 or y > 1.5:
                value = [0, 0]
            elif 0.4<y<0.6:
                value = [1.5, 0]
            else:
                value = [1, 0]
            value = [1, 0]

            setup_T.append(T_Point([TT.FIXED, TT.FIXED], X, init_val=value))
        elif tag == "Right":
            value = [1, 0]
            deriv = [Deriv(comp=[0], orders=[(1, 0)], value=0.), Deriv(comp=[1], orders=[(1, 0)], value=0.)] # dux/dx = 0 and duy/dx = 0
            setup_T.append(T_Point([TT.EXIT, TT.EXIT], X, init_val=value, derivatives=deriv))
        elif tag == "Normal"  or tag == "Left_extra" :
            vx_init = 1 - (X[0]-0.5)/2.5
            value = [vx_init, 0]

            setup_T.append(T_Point([TT.NORMAL, TT.NORMAL], X, init_val=value))
        else:
            raise ValueError(f"Unknown tag {tag}")

    setup_T = {i: point for i, point in enumerate(setup_T)}
    u_graph_time = UGraphTime(setup_T, N_component=N_comp, grad_acc=2, device=cfg.DEVICE)
    # plot_points(u_graph_time._Xs, u_graph_time.dirich_mask[:, 0], title="grad mask")
    with open("./save_u_graph_T.pth", "wb") as f:
        torch.save(u_graph_time, f)

    # Set up PDE graph for pressure
    setup_pde = {}
    for i, (X, tag) in enumerate(zip(Xs, p_tags)):
        # if np.any(np.all(np.isclose([[xmax, ymin], [xmax, ymax]], X), axis=1)):
        #     #setup_pde[i] = Point(T.DirichBC, X, value=[0])
        #     continue

        if tag == "Wall":
            deriv = [Deriv(comp=[0], orders=[(0, 1)], value=0.)]
            setup_pde[i] = Point(T.NeumOffsetBC, X, value=[0], derivatives=deriv)
        elif tag == "Left":
            deriv = [Deriv(comp=[0], orders=[(1, 0)], value=0.)]
            setup_pde[i] = Point(T.NeumOffsetBC, X, value=[0], derivatives=deriv)
            #setup_pde[i] = Point(T.Normal, X, value=[0])
        elif tag == "Right":
            setup_pde[i] = Point(T.DirichBC, X, value=[0])
        elif tag == "Normal"  or tag == "Left_extra":
            setup_pde[i] = Point(T.Normal, X, value=[0])
        # elif tag == "Left_extra":
        #     setup_pde[i] = Point(T.DirichBC, X, value=[0])
        else:
            raise ValueError(f"Unknown tag {tag}")


    u_graph_pde = UGraph(setup_pde, N_component=1, grad_acc=2, device=cfg.DEVICE)
    plot_points(u_graph_pde._Xs, u_graph_pde.updt_mask, title="grad mask")
    plot_points(u_graph_pde._Xs, u_graph_pde.pde_mask, title="pde mask")

    with open("./save_u_graph.pth", "wb") as f:
        torch.save(u_graph_pde, f)

    return u_graph_time, u_graph_pde


def load_graph(cfg):
    u_graph_T = torch.load("save_u_graph_T.pth", weights_only=False)
    u_graph_pde = torch.load("save_u_graph.pth", weights_only=False)
    return u_graph_T, u_graph_pde


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

    PDE_timefn: ExplicitNS | SemiExplNS

    u_graph_main: UGraphTime
    u_saves: dict[int, UTemp]

    def __init__(self, u_graph_T: UGraphTime, u_graph_PDE: UGraph, cfg_T: ConfigTime, cfg_in: Config):
        """ u_graph_T: Time graph.
            u_graph_PDE: Graph for internal PDE solver.
        """
        self.u_graph_T = u_graph_T
        self.u_graph_PDE = u_graph_PDE
        self.cfg_T = cfg_T
        self.cfg_in = cfg_in
        self.u_saves = {}
        self.Xs = None
        self.PDE_timefn = ExplicitNS(u_graph_T, u_graph_PDE, cfg_in, cfg_T) #ExplicitNS(u_graph_T, u_graph_PDE, cfg_in, cfg_T)

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

            update = self.PDE_timefn.solve(t, step_num)
            self.u_graph_T._us = update # set_grid_irreg(update)

            if step_num % 3 == 0:
                self.u_saves[step_num+1] = self.u_graph_T.get_all_us_Xs()[0].clone()

            if step_num == 31:
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

    u_g_T, u_g_PDE = load_graph(cfg)
    #u_g_T, u_g_PDE = mesh_graph(cfg)

    time_pde = TimePDEBase(u_g_T, u_g_PDE , time_cfg, cfg)
    time_pde.solve()

    # saved_graphs = time_pde.u_saves
    # for t, graph in saved_graphs.items():
    #     us, Xs = graph.us, graph.Xs
    #     plot_interp_graph(Xs, us[:, 0], title=f"t={t :.4g}")


if __name__ == "__main__":
    main()