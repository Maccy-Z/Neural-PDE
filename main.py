# import torch
#
# from pde.graph_grid.graph_store import Point, P_Types, Deriv
# from pde.graph_grid.U_graph import UGraph
# from pde.graph_grid.graph_utils import test_grid, gen_perim, plot_interp_graph
# from pde.config import Config
# from pde.NeuralPDE_Graph import NeuralPDEGraph
# from pdes.PDEs import Poisson, LearnedFunc
# from pde.utils import setup_logging
# from pde.loss import DummyLoss
# #from pde.mesh_generation.generate_mesh import gen_points_full
# from pde.mesh_generation.subproc_gen_mesh import run_subprocess
#
# def mesh_graph(cfg):
#     cfg = Config()
#     N_comp = 2
#
#     deriv = [Deriv(comp=[0], orders=[(1, 0)], value=0.), Deriv(comp=[1], orders=[(1, 0)], value=1.)]
#     value = [0. for _ in range(N_comp)]
#     #points, p_tags = gen_points_full()
#     points, p_tags = run_subprocess()
#
#     points = torch.from_numpy(points).float()
#     Xs_all = {}
#     for i, (point, tag) in enumerate(zip(points, p_tags)):
#         if P_Types.DERIV in tag:
#             Xs_all[i] = Point(tag, point, value=value, derivatives=deriv)
#         else:
#             Xs_all[i] = Point(tag, point, value=value)
#
#     u_graph = UGraph(Xs_all, N_component=2, grad_acc=4, device=cfg.DEVICE)
#
#     with open("save_u_graph.pth", "wb") as f:
#         torch.save(u_graph, f)
#
#     return u_graph
#
# def new_graph(cfg):
#     cfg = Config()
#     N_comp = 2
#
#     n_grid = 20
#     spacing = 1/(n_grid + 1)
#
#     deriv = [Deriv(comp=[0], orders=[(1, 0)], value=1.), Deriv(comp=[1], orders=[(1, 0)], value=0.)]
#
#     Xs_perim = gen_perim(1, 1, spacing)
#     perim_mask = (Xs_perim[:, 1] > 0) & (Xs_perim[:, 1] < 1) & (Xs_perim[:, 0] ==0)
#     Xs_neumann = Xs_perim[perim_mask]
#     #print(Xs_neumann)
#     Xs_dirich = Xs_perim[~perim_mask]
#
#     Xs_ghost = Xs_neumann.clone()
#     Xs_ghost[:, 0] = Xs_ghost[:, 0] - spacing
#     Xs_bulk = test_grid(spacing, (1- spacing), torch.tensor([n_grid, n_grid]), device="cpu")
#
#     Xs_fix = [Point(P_Types.DirichBC, X, value=[0. for _ in range(N_comp)]) for X in Xs_dirich]
#     Xs_deriv = [Point(P_Types.NeumCentralBC , X, value=[0. for _ in range(N_comp)], derivatives=deriv) for X in Xs_neumann]
#     Xs_ghost = [Point(P_Types.Ghost, X, value=[0. for _ in range(N_comp)]) for X in Xs_ghost]
#     Xs_bulk = [Point(P_Types.Normal, X, value= [0. for _ in range(N_comp)]) for X in Xs_bulk]
#
#
#     Xs_all = {i: X for i, X in enumerate(Xs_deriv + Xs_fix + Xs_bulk + Xs_ghost)}
#     u_graph = UGraph(Xs_all, N_component=N_comp, grad_acc=4, device=cfg.DEVICE)
#
#     with open("save_u_graph.pth", "wb") as f:
#         torch.save(u_graph, f)
#
#     return u_graph
#
# def load_graph(cfg):
#     u_graph = torch.load("save_u_graph.pth")
#     return u_graph
#
# def true_pde():
#     cfg = Config()
#     #u_graph = load_graph(cfg)
#     #u_graph = mesh_graph(cfg)
#     u_graph = new_graph(cfg)
#     pde_fn = Poisson(cfg, device=cfg.DEVICE)
#     pde_adj = NeuralPDEGraph(pde_fn, u_graph, cfg, DummyLoss())
#
#     pde_adj.forward_solve()
#
#     us, Xs = u_graph.get_all_us_Xs()
#     plot_interp_graph(Xs, us[:, 0])
#     plot_interp_graph(Xs, us[:, 1])
#
#
# if __name__ == "__main__":
#     setup_logging(debug=True)
#     torch.manual_seed(1)
#
#     true_pde()

