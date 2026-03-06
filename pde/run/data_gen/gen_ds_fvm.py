""" Convert a FVM dataset into pde dataset. """
import torch
import os

from pde.config import Config
from pde.NeuralPDE_Graph import NeuralPDEGraph
from pde.pdes.PDEs import Fluid
from pde.utils import setup_logging, ARTEFACT_DIR
from pde.loss import DummyLoss
from pde.run.generate_graph import load_ds_graph


def convert_graph(save_file):
    """ Generate true solution using known PDE. """
    cfg = Config()
    U_graph, Us_values = load_ds_graph(save_file, cfg)
    U_graph.plot_interp(Us_values, title=f'FVM solution')

    return Us_values, U_graph


def gen_dataset():
    i = 0
    fvm_ds_dir = f'{ARTEFACT_DIR}/fvm2pde_dataset'
    save_files = sorted(os.listdir(fvm_ds_dir))
    for save_file in save_files:
        print(f'{save_file = }')
        Us_values, U_graph = convert_graph(f'{fvm_ds_dir}/{save_file}')

        i += 1
        # Save the solution and graph
        save_dict = {"Us_values": Us_values, "U_graph": U_graph}
        with open(ARTEFACT_DIR / "dataset_fvm"
                                 "" / f"{i}.pth", "wb") as f:
            torch.save(save_dict, f)


if __name__ == "__main__":
    import numpy as np
    np.random.seed(1)
    setup_logging(debug=2)
    gen_dataset()



