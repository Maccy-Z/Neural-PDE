import numpy as np
from collections import defaultdict
import torch
from cprint import c_print
import pickle

from pde.utils import ARTEFACT_DIR
from pde.graph_grid.graph_store import Point, Deriv
from pde.graph_grid.graph_store import P_Types as PT
from pde.graph_grid.U_graph import UGraph, setup_graph, UValues
from mesh_gen.meshes_pde import gen_points_full, gen_mesh_random
from pde.config import Config
from mesh_gen.mesh_gen_utils import plot_mesh


def boundary_normals(points, triangles, bc_edges):
    """
    Compute outward unit normals for every boundary vertex and
    return them in a dictionary {vertex_idx: normal (2‑vector)}.

    Parameters
    ----------
    points     : (nV, 2) torch.Tensor or ndarray[float]
    triangles  : (nT, 3) torch.Tensor or ndarray[int]
    bc_edges   : (nE, 2) torch.Tensor or ndarray[int]

    Returns
    -------
    normals_at_v : dict[int, np.ndarray(shape=(2,))]
        Mapping from boundary‑vertex index → outward unit normal.
    """
    points = points.numpy()
    triangles = triangles.numpy()

    # --- build edge → opposite‑vertex lookup --------------------------------
    edge_to_tri = {}
    for tri in triangles:
        for a, b, c in (
            (tri[0], tri[1], tri[2]),
            (tri[1], tri[2], tri[0]),
            (tri[2], tri[0], tri[1]),
        ):
            edge_to_tri[tuple(sorted((a, b)))] = c

    # --- compute outward normal for each boundary edge ----------------------
    nE = bc_edges.shape[0]
    edge_normals = np.zeros((nE, 2))
    edge_lengths = np.zeros(nE)

    for e_idx, (i, j) in enumerate(bc_edges):
        pi, pj = points[i], points[j]
        k = edge_to_tri[tuple(sorted((i, j)))]
        pk = points[k]

        t = pj - pi
        L = np.linalg.norm(t)
        if L == 0:
            raise ValueError(f"Zero‑length boundary edge ({i}, {j})")

        n_candidate = np.array([t[1], -t[0]]) / L
        midpoint = 0.5 * (pi + pj)
        v_int = pk - midpoint

        edge_normals[e_idx] = n_candidate if np.dot(n_candidate, v_int) < 0 else -n_candidate
        edge_lengths[e_idx] = L

    # --- accumulate inverse‑distance‑weighted normals at each boundary vertex
    v_to_edges = defaultdict(list)            # vertex → list[(edge_idx, weight)]
    for e_idx, (i, j) in enumerate(bc_edges):
        w = 1.0 / edge_lengths[e_idx]         # inverse‑distance weight
        v_to_edges[i].append((e_idx, w))
        v_to_edges[j].append((e_idx, w))

    normals_at_v = {}
    for v, edge_list in v_to_edges.items():
        weighted_sum = np.zeros(2)
        w_sum = 0.0
        for e_idx, w in edge_list:
            weighted_sum += w * edge_normals[e_idx]
            w_sum += w
        normal = weighted_sum / np.linalg.norm(weighted_sum)
        normals_at_v[int(v)] = normal         # ensure key is a Python int

    return normals_at_v


def plot_helper(Xs, p_tags):
    id_map = {s: i for i, s in enumerate(dict.fromkeys(p_tags))}
    ids = [id_map[s] for s in p_tags]
    plot_mesh(Xs, ids)

def mesh_heat(cfg, max_degree=2, grad_neigh=25):
    cfg = Config()
    N_comp = 1
    Xs, triangles, (int_edges, bc_edges), p_tags = gen_points_full()
    Xs = torch.from_numpy(Xs).float()
    triangles = torch.from_numpy(triangles).int()

    # all_tags = np.concatenate([np.zeros(len(int_edges)), np.ones(len(bc_edges))], axis=0, dtype=np.float32)
    # all_tags = torch.from_numpy(all_tags)
    # all_edgs = torch.from_numpy(np.concatenate([int_edges, bc_edges], axis=0, dtype=np.int32))
    # plot_edges(Xs, all_edgs, all_tags)
    # exit(7)

    normals = boundary_normals(Xs, triangles, bc_edges)

    # Process boundary points
    bc_edges = torch.from_numpy(bc_edges)
    bc_points = torch.unique(bc_edges.flatten(), dim=0)

    deriv = [Deriv(comp=[0], orders=[(1, 0)], value=0.)]#, Deriv(comp=[1], orders=[(1, 0)], value=1.)]
    Xs_all = {}
    for i, (X, tag) in enumerate(zip(Xs, p_tags)):
        x, y = X
        value = [x for _ in range(N_comp)]

        deriv = [Deriv(comp=[0], orders=[(0, 0)], value=0, weights=[1]),
                 # Deriv(comp=[1], orders=[(0, 0)], value=0, weights=[1]),
                 # Deriv(comp=[2], orders=[(0, 0)], value=0, weights=[1]),
                 ]

        if tag == "Normal":
            Xs_all[i] = Point(PT.Normal, X, value=value)
            assert i not in bc_points, "Normal point is also a boundary point"
        elif tag == "wall_bottom":
            Xs_all[i] = Point(PT.NeumOffsetBC, X, value=value, derivatives=deriv)
        elif tag == "wall_top":
            Xs_all[i] = Point(PT.NeumOffsetBC, X, value=value, derivatives=deriv)
        elif tag == "wall_left":
            Xs_all[i] = Point(PT.NeumOffsetBC, X, value=value, derivatives=deriv)
        elif tag == "wall_right":
            Xs_all[i] = Point(PT.NeumOffsetBC, X, value=value, derivatives=deriv)
        elif tag == "circle":
            # value = [1. for _ in range(N_comp)]
            # Xs_all[i] = Point(PT.DirichBC, X, value=value)
            n_hat = normals[i].tolist()
            deriv = [Deriv(comp=[0, 0], orders=[(1, 0), (0, 1)], value=-1., weights=[n_hat[0], n_hat[1]]),
                     # Deriv(comp=[1, 1], orders=[(1, 0), (0, 1)], value=-2., weights=[n_hat[0], n_hat[1]]),
                     # Deriv(comp=[2, 2], orders=[(1, 0), (0, 1)], value=-3., weights=[n_hat[0], n_hat[1]])
                     ]

            Xs_all[i] = Point(PT.NeumOffsetBC, X, value=value, derivatives=deriv)
        else:
            raise ValueError(f"Unknown point tag {tag}")

    c_print(f'n_points: {len(Xs_all)}, n_bc: {len(bc_edges)}', color="bright_green")
    U_graph = UGraph(Xs_all, N_comp=N_comp, grad_neigh=grad_neigh, max_degree=max_degree, tri=triangles, device=cfg.device)

    with open("../artefacts/save_u_graph.pth", "wb") as f:
        torch.save((U_graph, triangles), f)

    # exit("Done")
    return U_graph, triangles


def mesh_graph(cfg) -> tuple[UGraph, UValues]:
    N_comp = 3
    # Xs, triangles, (int_edges, bc_edges), p_tags = gen_points_full()
    Xs, triangles, (_, bc_edges), p_tags = gen_mesh_random()
    plot_helper(Xs, p_tags)

    Xs = torch.from_numpy(Xs).float()
    triangles = torch.from_numpy(triangles).int()

    # Process boundary points
    normals = boundary_normals(Xs, triangles, bc_edges)

    Xs_all = {}
    for i, (X, tag) in enumerate(zip(Xs, p_tags)):
        x, y = X

        value = [0*x for _ in range(N_comp)]

        if tag == "Normal":
            Xs_all[i] = Point(PT.Normal, X, value=value)
            continue

        # Boundary conditions
        n_hat = normals[i].tolist()
        # wall_deriv = Deriv(comp=[2, 2, 0, 0, 1, 1, 1, 0], orders=[(1, 0), (0, 1), (2, 0), (0, 2), (1, 1), (2, 0), (0, 2), (1, 1)], value=0,
        #       weights=[-n_hat[0], -n_hat[1], n_hat[0], n_hat[0], n_hat[0], n_hat[1], n_hat[1], n_hat[1]])
        wall_deriv = Deriv(comp=[2, 2, 0, 0, 1, 1], orders=[(1, 0), (0, 1), (2, 0), (0, 2), (2, 0), (0, 2)], value=0,
                           weights=[-n_hat[0], -n_hat[1], n_hat[0], n_hat[0], n_hat[1], n_hat[1]])

        if tag == "Navier_wall":
            deriv = [Deriv(comp=[0], orders=[(0, 0)], value=0.),
                     Deriv(comp=[1], orders=[(0, 0)], value=0.),
                     wall_deriv
                     ]

            Xs_all[i] = Point(PT.NeumOffsetBC, X, value=value, derivatives=deriv)
        elif tag == "wall_left":
            deriv = [
                        # Deriv(comp=[0], orders=[(1, 0)], value=0, weights=[1]),
                        # Deriv(comp=[0, 1], orders=[(1, 0), (0, 1)], value=0.),
                        wall_deriv,
                        # Deriv(comp=[2, 1, 1], orders=[(0, 1), (2, 0), (0, 2)], value=0, weights=[-1, 1, 1]),
                        Deriv(comp=[1], orders=[(1, 0)], value=0, weights=[1]),

                        # Pressure
                        Deriv(comp=[2], orders=[(0, 0)], value=2.),
                     ]
            Xs_all[i] = Point(PT.NeumOffsetBC, X, value=[0, 0, 1], derivatives=deriv)
        elif tag == "wall_right":
            deriv = [
                        # Deriv(comp=[0], orders=[(1, 0)], value=0., weights=[1]),
                        # Deriv(comp=[1, 0], orders=[(1, 0), (0, 1)], value=0, weights=[1, 1]),
                        wall_deriv,

                        # Deriv(comp=[2, 1, 1], orders=[(0, 1), (2, 0), (0, 2)], value=0, weights=[-1, 1, 1]),
                        Deriv(comp=[1], orders=[(1, 0)], value=0, weights=[1]),
                        # Pressure
                        Deriv(comp=[2], orders=[(0, 0)], value=0.),
                     ]
            Xs_all[i] = Point(PT.NeumOffsetBC, X, value=value, derivatives=deriv)
        else:
            raise ValueError(f"Unknown point tag {tag}")

    c_print(f'n_points: {len(Xs_all)}', color="bright_green")

    U_graph, Us_values = setup_graph(Xs_all, N_comp=N_comp, grad_neigh=cfg.grad_neigh, max_degree=cfg.max_degree, tri=triangles, device=cfg.device)
    return U_graph, Us_values


def load_ds_graph(file, cfg: Config) -> tuple[UGraph, UValues]:
    with open(file, 'rb') as f:
        ds: dict = pickle.load(f)

    Xs = ds['Xs']                       # shape = (N_us_tot, 2)
    triangles = ds['triangles']         # shape = (N_tri, 3)
    bc_edges = ds['bc_edges']           # shape = (N_bc_edges, 2)
    p_tags = ds['p_tags']               # shape = (N_bc_edges,)
    Us_true = ds['Us']                  # shape = (N_us_tot, N_comp)

    # Xs, triangles, (_, bc_edges), p_tags = gen_mesh_random()
    for i, (X, _) in enumerate(zip(Xs, p_tags)):
        x, y = X
        if x == 2 and y == 0:
            p_tags[i] = "NavierWall"
        elif x == 2 and y == 1.5:
            p_tags[i] = "NavierWall"
        elif x == 0 and y == 0:
            p_tags[i] = "NavierWall"
        elif x == 0 and y == 1.5:
            p_tags[i] = "NavierWall"

    # plot_helper(Xs, p_tags)

    Xs = torch.from_numpy(Xs).float()
    Us_true = torch.from_numpy(Us_true).float()
    triangles = torch.from_numpy(triangles).int()

    N_comp = 3

    # Process boundary points
    normals = boundary_normals(Xs, triangles, bc_edges)

    Xs_all = {}
    for i, (X, tag, U) in enumerate(zip(Xs, p_tags, Us_true)):
        x, y = X
        value = U.tolist() #[0 * x for _ in range(N_comp)]

        if tag == "Normal":
            Xs_all[i] = Point(PT.Normal, X, value=value)
            continue

        # Boundary conditions
        n_hat = normals[i].tolist()
        # wall_deriv = Deriv(comp=[2, 2, 0, 0, 1, 1, 1, 0], orders=[(1, 0), (0, 1), (2, 0), (0, 2), (1, 1), (2, 0), (0, 2), (1, 1)], value=0,
        #       weights=[-n_hat[0], -n_hat[1], n_hat[0], n_hat[0], n_hat[0], n_hat[1], n_hat[1], n_hat[1]])
        wall_deriv = Deriv(comp=[2, 2, 0, 0, 1, 1], orders=[(1, 0), (0, 1), (2, 0), (0, 2), (2, 0), (0, 2)], value=0,
                           weights=[-n_hat[0], -n_hat[1], n_hat[0], n_hat[0], n_hat[1], n_hat[1]])

        if tag == "Navier_wall" or tag == "NavierWall":
            deriv = [Deriv(comp=[0], orders=[(0, 0)], value=0.),
                     Deriv(comp=[1], orders=[(0, 0)], value=0.),
                     wall_deriv
                     ]
            Xs_all[i] = Point(PT.NeumOffsetBC, X, value=value, derivatives=deriv)

        elif tag == "wall_left" or tag == "Left":
            deriv = [
                # Deriv(comp=[0, 1], orders=[(1, 0), (0, 1)], value=0.),
                wall_deriv,
                # Deriv(comp=[2, 1, 1], orders=[(0, 1), (2, 0), (0, 2)], value=0, weights=[-1, 1, 1]),
                Deriv(comp=[1], orders=[(1, 0)], value=0, weights=[1]),

                # Pressure
                Deriv(comp=[2], orders=[(0, 0)], value=2.),
            ]
            Xs_all[i] = Point(PT.NeumOffsetBC, X, value=[0, 0, 1], derivatives=deriv)

        elif tag == "wall_right" or tag == "Right":
            deriv = [
                # Deriv(comp=[0], orders=[(1, 0)], value=0., weights=[1]),
                # Deriv(comp=[1, 0], orders=[(1, 0), (0, 1)], value=0, weights=[1, 1]),
                wall_deriv,

                # Deriv(comp=[2, 1, 1], orders=[(0, 1), (2, 0), (0, 2)], value=0, weights=[-1, 1, 1]),
                Deriv(comp=[1], orders=[(1, 0)], value=0, weights=[1]),
                # Pressure
                Deriv(comp=[2], orders=[(0, 0)], value=0.),
            ]
            Xs_all[i] = Point(PT.NeumOffsetBC, X, value=value, derivatives=deriv)

        else:
            raise ValueError(f"Unknown point tag {tag}")

    c_print(f'n_points: {len(Xs_all)}', color="bright_green")

    U_graph, Us_values = setup_graph(Xs_all, N_comp=N_comp, tri=triangles,
                                     grad_neigh=cfg.grad_neigh, max_degree=cfg.max_degree, device=cfg.device)

    return U_graph, Us_values



if __name__ == "__main__":
    torch.manual_seed(0)
    load_ds_graph(Config())



