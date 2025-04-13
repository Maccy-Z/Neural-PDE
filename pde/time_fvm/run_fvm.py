from cprint import c_print
import pickle
import torch
import numpy as np

from pde.graph_grid.fvm_store import EdgeBCTypes as E
from pde.graph_grid.fvm_store import Edge
from pde.mesh_generation.generate_mesh import gen_mesh_fvm
from pde.time_fvm.time_fvm import FVMMesh, FVMEquation
from pde.graph_grid.graph_utils import plot_edges
from pde.time_fvm.config_fvm import ConfigFVM

def mesh_graph(cfg: ConfigFVM, new):
    N_comp = 4
    if new:
        c_print(f'Creating new mesh', "green")
        xmin, xmax = 0, 2
        ymin, ymax = 0, 1.4
        mesh_stuff = gen_mesh_fvm(xmin, xmax, ymin, ymax, areas=[cfg.min_A, cfg.max_A], cell_lnscale=cfg.lnscale)
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


    c_print(f'Number of mesh cells: {len(tri_idx)}', "green")
    c_print(f'Number of mesh edges: {len(all_edgs)}', "green")

    # all_tags = np.concatenate([np.zeros(len(int_edgs)), np.ones(len(edge_tag))], axis=0, dtype=np.float32)
    # all_tags = torch.from_numpy(all_tags)
    # plot_edges(Xs, all_edgs, all_tags)
    # exit(7)

    bc_tags = {}
    for bc_idx, (e_tag, e_vert) in enumerate(zip(edge_tag, bound_edgs, strict=True)):
        if e_tag == "NavierWall":
            bc_tags[bc_idx] = Edge([E.Dirich, E.Dirich, E.Neuman, E.Neuman], [0., 0, None, None], [None, None, 0, 0])   #(E.WALL, 0)

        elif e_tag == "Left":
            X0, X1 = Xs[e_vert]
            x0, y0 = X0
            x1, y1 = X1
            v_in = 0.5 if (0.1 < (y0+y1)/2 < 0.5) else 0
            T = 100 #if (y0+y1)/2 > 0.7 else 250
            bc_tags[bc_idx] = Edge([E.Neuman, E.Dirich, E.Dirich, E.Dirich], [None, 0, 1.01, T], [0, None, None, None]) #(E.INLET, 0)
        elif e_tag == "Right":
            bc_tags[bc_idx] = Edge([E.Neuman, E.Neuman, E.Farfield, E.Dirich], [None, None, 1.0, 100], [0, 0, None, None]) #Edge([E.Neuman, E.Neuman, E.Dirich], [None, None, 1], [0, 0, None])  #(E.EXIT, 0)
        else:
            raise ValueError(f'Unknown edge tag {e_tag}')


    return Xs, tri_idx, all_edgs, bc_edge_mask, bc_tags, N_comp


def init_conds(centroids, cfg: ConfigFVM, load_state):

    if load_state:
        with open("save_state.pt", "rb") as f:
            us_init = torch.load(f)
    else:
        x, y = centroids[:, 0], centroids[:, 1]

        us_init = torch.zeros_like(x).unsqueeze(1).repeat(1, 4)
        us_init[:, 0] = 0.1 #(0.4<y) * (y<0.8) * 0.1
        us_init[:, 1] = 0
        us_init[:, 2] = 1.  # + ((x>1) * (x < 2)) * 0.01
        us_init[:, 3] = 100

        # Energy: C_v * T + 0.5 * (u^2 + v^2)
        E = cfg.C_v * us_init[:, 3] + 0.5 * (us_init[:, 0] ** 2 + us_init[:, 1] ** 2)
        us_init[:, 3] = E * us_init[:, 2]  # Energy density
        # Convert to momentum
        us_init[:, 0] = us_init[:, 0] * us_init[:, 2]
        us_init[:, 1] = us_init[:, 1] * us_init[:, 2]


    return us_init


def main():
    import pickle
    torch.manual_seed(0)
    new = True
    load_state = False

    cfg = ConfigFVM()

    prob_definition = mesh_graph(cfg, new)
    Xs, tri_idx, all_edgs, bc_edge_mask, bc_tags, N_comp = prob_definition

    if new:
        c_print(f'Generating mesh...', "green")
        mesh = FVMMesh(Xs, tri_idx, all_edgs, bc_edge_mask, device="cuda")
        pickle.dump(mesh, open("mesh.pkl", "wb"))
    else:
        c_print(f'Loading mesh', "green")
        mesh = pickle.load(open("mesh.pkl", "rb"))

    print(f'{mesh.areas.min() = }')

    centroids = mesh.centroids.clone()
    us_init = init_conds(centroids, cfg, load_state)
    solver = FVMEquation(cfg, mesh, N_comp, bc_tags, us_init=us_init, device="cuda")
    solver.solve()

if __name__ == "__main__":
    print("Running fvm ")
    print()
    main()
