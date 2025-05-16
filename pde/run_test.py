import torch
from pde.config import Config
from pde.NeuralPDE_Graph import NeuralPDEGraph
from pdes.PDEs import Poisson, Fluid
from pde.utils import setup_logging
from pde.loss import DummyLoss



def load_graph(cfg):
    u_graph, triangles = torch.load("save_u_graph.pth", weights_only=False)
    return u_graph, triangles


def true_pde():
    cfg = Config()
    u_graph, triangles = load_graph(cfg)

    us_all, _ = u_graph.get_all_us_Xs()
    derivs = u_graph.deriv_calc_eval.derivative(us_all)

    pde_fn = Fluid(cfg, device=cfg.DEVICE)
    pde_adj = NeuralPDEGraph(pde_fn, u_graph, cfg, DummyLoss(), triangles)
    # pde_adj.forward_solve()

    (us_load, Xs_load) = torch.load("us_all.pth", weights_only=True)

    u_graph._us = us_load

    us_all, Xs_all = u_graph.get_all_us_Xs()
    # us_all[:, 2] = us_all[:, 2].clamp(max=1.)
    deriv_dict = u_graph.deriv_calc_eval.derivative(us_all)

    Us = deriv_dict[(0, 0)].T
    dUdx, dUdy = deriv_dict[(1, 0)].T, deriv_dict[(0, 1)].T
    d2Udx2, d2Udy2 = deriv_dict[(2, 0)].T, deriv_dict[(0, 2)].T

    temp = - dUdx[2] + d2Udx2[0] + d2Udy2[0]

    # pde_adj.plot_points(us_all[:, 2])
    pde_adj.plot_points(temp, title="pressure inlet condition")

if __name__ == "__main__":
    setup_logging(debug=True)
    torch.manual_seed(1)

    true_pde()


