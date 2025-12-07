import torch

from pde.config import Config
from pde.NeuralPDE_Graph import NeuralPDEGraph
from pde.pdes.PDEs import Fluid, FluidLearned, NNFunc
from pde.utils import setup_logging, ARTEFACT_DIR
from pde.loss import DummyLoss, MSELoss2, MSELossNorm
from pde.run.generate_graph import mesh_graph, load_graph

def gen_graph():
    """ Generate true solution using known PDE. """
    cfg = Config()
    U_graph, Us_values = mesh_graph(cfg)
    pde_fn = Fluid(cfg, device=cfg.device)
    # pde_fn = HeatLearned(cfg, device=cfg.DEVICE)

    loss_fn = DummyLoss()

    pde_adj = NeuralPDEGraph(pde_fn, cfg, loss_fn)

    _, solve_info = pde_adj.forward_solve(U_graph, Us_values)

    print(f'{solve_info = }')
    # print(f'{solve_info['converged'] = }')
    U_graph.plot_interp(Us_values, title=f'{"Converged" if solve_info["converged"] else "Not Converged"} solution')

    return solve_info, Us_values, U_graph

def gen_dataset():
    i = 0
    while i < 10:
        solve_info, Us_values, U_graph = gen_graph()
        if not solve_info['converged']:
            print(f"Graph {i} did not converge, regenerating...")
            continue
        else:
            i += 1
            # Save the solution and graph
            save_dict = {"Us_values": Us_values, "U_graph": U_graph}
            with open(ARTEFACT_DIR / "dataset" / f"{i}.pth", "wb") as f:
                torch.save(save_dict, f)

if __name__ == "__main__":
    import numpy as np
    np.random.seed(1)
    setup_logging(debug=2)
    gen_dataset()

