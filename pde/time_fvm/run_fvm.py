from cprint import c_print
import pickle
import time
import torch

from pde.time_dependent.time_cfg import ConfigTime
from pde.graph_grid.fvm_store import EdgeBCTypes as E
from pde.graph_grid.fvm_store import Edge
from pde.config import Config
from pde.mesh_generation.generate_mesh import gen_mesh_fvm
from pde.time_fvm.time_fvm import FVMMesh, FVMEquation

def mesh_graph(cfg):
    N_comp = 3

    new_graph = True
    if new_graph:
        xmin, xmax = 0, 3
        ymin, ymax = 0.0, 1.5
        mesh_stuff = gen_mesh_fvm(xmin, xmax, ymin, ymax, areas=[1.5e-3, 1.5e-3])
        Xs, tri_idx, (int_edgs, bound_edgs), edge_tag = mesh_stuff
        pickle.dump(mesh_stuff, open("mesh_stuff.pkl", "wb"))
    else:
        mesh_stuff = pickle.load(open("mesh_stuff.pkl", "rb"))
        Xs, tri_idx, (int_edgs, bound_edgs), edge_tag = mesh_stuff

    Xs = torch.from_numpy(Xs).float()
    tri_idx = torch.from_numpy(tri_idx).int()
    int_edgs, bound_edgs = torch.from_numpy(int_edgs), torch.from_numpy(bound_edgs)
    all_edgs = torch.cat([int_edgs, bound_edgs], dim=0)
    bc_edge_mask = torch.cat([torch.zeros_like(int_edgs[:, 0], dtype=torch.bool), torch.ones_like(bound_edgs[:, 0], dtype=torch.bool)], dim=0)

    bc_tags = {}
    for bc_idx, (e_tag, e_vert) in enumerate(zip(edge_tag, bound_edgs, strict=True)):
        if e_tag == "Wall":
            bc_tags[bc_idx] = Edge([E.Dirich, E.Dirich, E.Neuman], [0.0, 0, None], [None, None, 0])   #(E.WALL, 0)
        elif e_tag == "Left":
            bc_tags[bc_idx] = Edge([E.Dirich, E.Dirich, E.Neuman], [0.0, 0, None], [None, None, 0]) #(E.INLET, 0)
        elif e_tag == "Right":
            bc_tags[bc_idx] = Edge([E.Dirich, E.Dirich, E.Neuman], [0.0, 0, None], [None, None, 0]) #Edge([E.Neuman, E.Neuman, E.Dirich], [None, None, 1], [0, 0, None])  #(E.EXIT, 0)
        else:
            raise ValueError(f'Unknown edge tag {e_tag}')


    c_print(f'Number of mesh points: {len(Xs)}', "green")

    return Xs, tri_idx, all_edgs, bc_edge_mask, bc_tags, N_comp

def init_conds(centroids):
    x, y = centroids[:, 0], centroids[:, 1]

    #us_init = torch.exp(-((cent_x - 1.5) ** 2) / 1)#
    us_init = torch.zeros_like(x).unsqueeze(1).repeat(1, 3)
    # us_init = (x-3) ** 2
    # # us_init = us_init.repeat(1, 3)
    us_init[:, 0] =  0 #((x>1) * (x < 2)) * 0.01 # torch.randn_like(us_init[:, 0]) * 0.00 #us_init[:, 0] * 1e-6 + 0.0
    us_init[:, 1] = 0
    us_init[:, 2] =  ((x>1) * (x < 2)) * 0.01

    # print(us_init)
    # exit(9)
    return us_init


def main():
    # from pde.utils import setup_logging
    torch.manual_seed(0)
    # setup_logging(debug=False)

    cfg = Config()
    time_cfg= ConfigTime()
    c_print(f'{time_cfg.dt = }', color="bright_magenta")

    #u_g_T = load_graph(cfg)
    prob_definition = mesh_graph(cfg)
    Xs, tri_idx, all_edgs, bc_edge_mask, bc_tags, N_comp = prob_definition

    mesh = FVMMesh(Xs, tri_idx, all_edgs, bc_edge_mask, device="cuda")
    centroids = mesh.centroids.clone()
    us_init = init_conds(centroids)
    solver = FVMEquation(mesh, N_comp, bc_tags, us_init=us_init, device="cuda")


if __name__ == "__main__":
    main()
