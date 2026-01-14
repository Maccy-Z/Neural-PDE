import torch
from cprint import c_print

from pde.config import Config
from pde.NeuralPDE_Graph import NeuralPDEGraph
from pde.graph_grid.U_graph import UValues, UGraph
from pde.pdes.PDEs import Fluid, NNFunc
from pde.utils import setup_logging, ARTEFACT_DIR
from pde.loss import DummyLoss, MSELossNorm
from pde.run.generate_graph import mesh_graph
from pde.run.batching import GraphDataset, GraphSample


def true_pde():
    """ Generate true solution using known PDE. """
    from generate_graph import load_ds_graph
    cfg = Config()

    pde_fn = Fluid(cfg, device=cfg.device)
    loss_fn = DummyLoss()

    # U_graph, Us_values = mesh_graph(cfg)
    load_file = f'{ARTEFACT_DIR}/fvm2pde_dataset/01-14_21-08-43.pkl'
    U_graph, Us_values = load_ds_graph(load_file, cfg)
    U_graph.init_lin_solver(cfg)


    Us_values = U_graph.smooth_grid_like(Us_values)
    # Us_values = U_graph.zero_grid_like(Us_values)

    pde_adj = NeuralPDEGraph(pde_fn, cfg, loss_fn)

    Us_history, _ = pde_adj.forward_solve(U_graph, Us_values)

    for i, Us in enumerate(Us_history):
        U_graph.plot_interp(Us, title=f'Solution step {i}')


    return


if __name__ == "__main__":
    setup_logging(debug=1)
    torch.set_printoptions(linewidth=120, precision=7)
    torch.manual_seed(1)
    # torch.autograd.set_detect_anomaly(True)
    # torch.use_deterministic_algorithms(True)

    true_pde()