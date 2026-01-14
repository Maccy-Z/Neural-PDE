import torch

from pde.config import Config
from pde.NeuralPDE_Graph import NeuralPDEGraph
from pde.pdes.PDEs import Fluid
from pde.utils import setup_logging, ARTEFACT_DIR
from pde.loss import DummyLoss
from pde.run.generate_graph import mesh_graph


def gen_graph():
    """ Generate true solution using known PDE. """
    cfg = Config()
    U_graph, Us_values = mesh_graph(cfg)
    U_graph.init_lin_solver(cfg)
    pde_fn = Fluid(cfg, device=cfg.device)
    loss_fn = DummyLoss()
    pde_adj = NeuralPDEGraph(pde_fn, cfg, loss_fn)

    _, solve_info = pde_adj.forward_solve(U_graph, Us_values)
    print(f'{solve_info = }')

    # Convergence requires: 1) Residuals small enough and 2) Smooth enough solution
    converged = solve_info["converged"]

    # Check gradient statistics
    grads, _  = U_graph.get_Us_dUs(Us_values)       # shape = [n_points, n_grads, n_comp]
    dUs_dXs = grads[:, [1, 2]]
    mean, std = dUs_dXs.mean(dim=(0, 1)), dUs_dXs.std(dim=(0, 1))

    dUs_dXs_norm = (dUs_dXs - mean) / (std + 1e-6)

    dUs_dXs_mag = torch.sqrt((dUs_dXs_norm ** 2).sum(dim=1))
    if dUs_dXs_mag.max() > 20:
        print("Gradient too large:", dUs_dXs_mag.max().item())
        converged = False

    # Plot
    U_graph.plot_interp(Us_values, title=f'{"Converged" if converged else "Not Converged"} solution')

    return converged, Us_values, U_graph


def gen_dataset():
    i = 0
    while i < 10:
        converged, Us_values, U_graph = gen_graph()
        if not converged:
            print(f"Graph {i} did not converge, regenerating...")
            continue
        else:
            i += 1
            # # Save the solution and graph
            # save_dict = {"Us_values": Us_values, "U_graph": U_graph}
            # with open(ARTEFACT_DIR / "dataset_pde" / f"{i}.pth", "wb") as f:
            #     torch.save(save_dict, f)


if __name__ == "__main__":
    import numpy as np
    np.random.seed(1)
    setup_logging(debug=2)
    gen_dataset()

