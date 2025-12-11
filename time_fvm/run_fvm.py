from cprint import c_print
import pickle
import torch
import numpy as np

from time_fvm.fvm_store import EdgeBCTypes as E
from time_fvm.fvm_store import Edge
from time_fvm.fvm_mesh import FVMMesh
from time_fvm.fvm_equation import FVMEquation
from mesh_gen.meshes_fvm import gen_mesh_fvm, gen_mesh_tunnel
from time_fvm.config_fvm import ConfigFVM
from time_fvm.sparse_utils import plot_edges

def mesh_graph(cfg: ConfigFVM, new):
    N_comp = 4
    if new:
        c_print(f'Creating new mesh', "green")

        mesh_stuff = gen_mesh_tunnel(areas=[cfg.min_A, cfg.max_A], cell_lnscale=cfg.lnscale)
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

    bc_tags = {}
    for bc_idx, (e_tag, e_vert) in enumerate(zip(edge_tag, bound_edgs, strict=True)):
        if e_tag == "NavierWall":
            bc_tags[bc_idx] = Edge([E.Dirich, E.Dirich, E.Neuman, E.Neuman], [0., 0, None, None], [None, None, 0, 0], tag=e_tag)
        elif e_tag == "Side":
            # bc_tags[bc_idx] = Edge([E.Neuman, E.Neuman, E.Dirich, E.Dirich], [None, None, 0.5, 100], [0, 0, None, None])
            bc_tags[bc_idx] = Edge([E.Farfield, E.Farfield, E.Farfield, E.Farfield], [None, None, None, None], [None, None, None, None], tag=e_tag)

        elif e_tag == "Left":
            X0, X1 = Xs[e_vert]
            x0, y0 = X0
            x1, y1 = X1
            v_in = 0.1 if (0.05 < (y0+y1)/2 < 1.45) else 0
            T = 100 #if (y0+y1)/2 > 0.7 else 250
            # bc_tags[bc_idx] = Edge([E.Neuman, E.Dirich, E.Dirich, E.Dirich], [None, 0, 1.01, T], [0, None, None, None])
            bc_tags[bc_idx] = Edge([E.Inlet, E.Inlet, E.Inlet, E.Inlet], [None, None, None, None], [None, None, None, None], tag=e_tag)

        elif e_tag == "Right":
            bc_tags[bc_idx] = Edge([E.Farfield, E.Farfield, E.Farfield, E.Farfield], [None, None, None, None], [None, None, None, None], tag=e_tag)
        else:
            raise ValueError(f'Unknown edge tag {e_tag}')


    return Xs, tri_idx, all_edgs, bc_edge_mask, bc_tags, N_comp


def init_conds(centroids, cfg: ConfigFVM, load_state, vx=5., vy=0., rho=1., T=100.):

    if load_state:
        with open("save_state.pt", "rb") as f:
            us_init = torch.load(f)
    else:
        x, y = centroids[:, 0], centroids[:, 1]

        us_init = torch.zeros_like(x).unsqueeze(1).repeat(1, 4)
        us_init[:, 0] = vx #50 * (x<.4) + 0 * (x>.4)
        us_init[:, 1] = vy
        us_init[:, 2] = rho # * (x<.4) + 1. #* (x>.4)
        us_init[:, 3] = T # * (x<.4) + 200 #* (x>.4)

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
    new = False
    load_state = False

    cfg = ConfigFVM()

    # Useful to set some parameters here
    T_nat = 100
    rho_nat = 1
    V_x_nat = 6.5
    cfg.exit_cfg.T_far = T_nat
    cfg.exit_cfg.rho_far = rho_nat
    cfg.exit_cfg.v_far = V_x_nat
    cfg.inlet_cfg.T_nat = T_nat
    cfg.inlet_cfg.rho_nat = rho_nat
    cfg.inlet_cfg.V_x_nat = V_x_nat


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
    us_init = init_conds(centroids, cfg, load_state, vx=V_x_nat, rho=rho_nat, T=T_nat)
    solver = FVMEquation(cfg, mesh, N_comp, bc_tags, us_init=us_init, device="cuda")
    solver.solve()

if __name__ == "__main__":
    print("Running fvm ")
    print()
    main()
