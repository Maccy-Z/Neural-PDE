import torch
from cprint import c_print
import numpy as np
from collections import defaultdict

from pde.graph_grid.graph_store import Point, Deriv
from pde.graph_grid.graph_store import P_Types as PT
from pde.graph_grid.U_graph import UGraph
from pde.graph_grid.graph_utils import plot_edges
from pde.config import Config
from pde.NeuralPDE_Graph import NeuralPDEGraph
from pdes.PDEs import Poisson, Fluid
from pde.utils import setup_logging
from pde.loss import DummyLoss
from pde.mesh_generation.generate_mesh import gen_points_full


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

# def boundary_normals(points, triangles, bc_edges):
#     """
#     points: (nV, 2) array of vertex coordinates
#     triangles: (nT, 3) array of integer vertex indices
#     bc_edges: (nE, 2) array of integer vertex indices for boundary edges
#     returns: (nE, 2) array of outward unit normals
#     """
#     points = points.numpy()
#     triangles = triangles.numpy()
#
#     # 1) Build adjacency: map each undirected edge to its triangle and opposite vertex
#     edge_to_tri = {}
#     for tri in triangles:
#         for a, b, c in [(tri[0], tri[1], tri[2]),
#                         (tri[1], tri[2], tri[0]),
#                         (tri[2], tri[0], tri[1])]:
#             edge = tuple(sorted((a, b)))
#             edge_to_tri[edge] = c
#
#     normals = np.zeros((len(bc_edges), 2))
#     for idx, (i, j) in enumerate(bc_edges):
#         # 2) get coordinates
#         pi, pj = points[i], points[j]
#         # 3) find interior vertex
#         k = edge_to_tri[tuple(sorted((i, j)))]
#         pk = points[k]
#         # tangent
#         t = pj - pi
#         L = np.linalg.norm(t)
#         if L == 0:
#             raise ValueError(f"Zero length edge at index {idx}")
#         # two candidate normals
#         n1 = np.array([ t[1], -t[0] ]) / L
#         # midpoint and interior direction
#         m = 0.5 * (pi + pj)
#         v_int = pk - m
#         # pick outward: we want n·v_int < 0
#         if np.dot(n1, v_int) < 0:
#             normals[idx] = n1
#         else:
#             normals[idx] = -n1
#
#     return normals

def mesh_heat(cfg):
    cfg = Config()
    N_comp = 1
    Xs, triangles, (int_edges, bc_edges), p_tags  = gen_points_full()
    Xs = torch.from_numpy(Xs).float()
    triangles = torch.from_numpy(triangles).int()
    all_tags = np.concatenate([np.zeros(len(int_edges)), np.ones(len(bc_edges))], axis=0, dtype=np.float32)
    all_tags = torch.from_numpy(all_tags)
    all_edgs = torch.from_numpy(np.concatenate([int_edges, bc_edges], axis=0, dtype=np.int32))
    # plot_edges(Xs, all_edgs, all_tags)
    # exit(7)
    normals = boundary_normals(Xs, triangles, bc_edges)

    # Process boundary points
    bc_edges = torch.from_numpy(bc_edges)
    bc_points = torch.unique(bc_edges.flatten(), dim=0)

    deriv = [Deriv(comp=[0], orders=[(1, 0)], value=0.)]#, Deriv(comp=[1], orders=[(1, 0)], value=1.)]
    Xs_all = {}
    for i, (X, tag) in enumerate(zip(Xs, p_tags)):
        value = [0. for _ in range(N_comp)]
        if tag == "Normal":
            Xs_all[i] = Point(PT.Normal, X, value=value)
            assert i not in bc_points, "Normal point is also a boundary point"
        elif tag == "wall_bottom":
            Xs_all[i] = Point(PT.DirichBC, X, value=value, derivatives=deriv)
        elif tag == "wall_top":
            Xs_all[i] = Point(PT.DirichBC, X, value=value, derivatives=deriv)
        elif tag == "wall_left":
            Xs_all[i] = Point(PT.DirichBC, X, value=value, derivatives=deriv)
        elif tag == "wall_right":
            value = [1. for _ in range(N_comp)]
            Xs_all[i] = Point(PT.DirichBC, X, value=value, derivatives=deriv)
        elif tag == "circle":
            n_hat = normals[i].tolist()

            deriv = [Deriv(comp=[0, 0], orders=[(1, 0), (0, 1)], value=-2., weights=[n_hat[0], n_hat[1]])]
            Xs_all[i] = Point(PT.NeumOffsetBC, X, value=value, derivatives=deriv)
        else:
            raise ValueError(f"Unknown point tag {tag}")

    c_print(f'n_points: {len(Xs_all)}, n_bc: {len(bc_edges)}', color="bright_green")
    u_graph = UGraph(Xs_all, N_component=N_comp, grad_acc=4, tri=triangles, device=cfg.DEVICE)

    with open("save_u_graph.pth", "wb") as f:
        torch.save(u_graph, f)

    # exit("Done")
    return u_graph, triangles


def mesh_graph(cfg):
    cfg = Config()
    N_comp = 3
    Xs, triangles, (int_edges, bc_edges), p_tags  = gen_points_full()
    Xs = torch.from_numpy(Xs).float()
    triangles = torch.from_numpy(triangles).int()

    # all_tags = np.concatenate([np.zeros(len(int_edges)), np.ones(len(bc_edges))], axis=0, dtype=np.float32)
    # all_tags = torch.from_numpy(all_tags)
    # all_edgs = torch.from_numpy(np.concatenate([int_edges, bc_edges], axis=0, dtype=np.int32))
    # plot_edges(Xs, all_edgs, colors=all_tags)
    # exit(7)

    # Process boundary points
    normals = boundary_normals(Xs, triangles, bc_edges)
    bc_edges = torch.from_numpy(bc_edges)
    bc_points = torch.unique(bc_edges.flatten(), dim=0)

    Xs_all = {}
    for i, (X, tag) in enumerate(zip(Xs, p_tags)):
        value = [0. for _ in range(N_comp)]

        x, y = X
        # value[1] = 10#1 - 1 / 2 * x

        if tag == "Normal":
            Xs_all[i] = Point(PT.Normal, X, value=value)
            assert i not in bc_points, "Normal point is also a boundary point"
            continue

        # Boundary conditions
        n_hat = normals[i].tolist()
        if tag == "wall_bottom" or tag == "wall_top" or y <= 0 or y >= 1.5:
            deriv = [Deriv(comp=[0], orders=[(0, 0)], value=0.),
                     Deriv(comp=[1], orders=[(0, 0)], value=0.),
                     # Deriv(comp=[2, 2], orders=[(1, 0), (0, 1)], value=0., weights=[n_hat[0],n_hat[1]]),
                     # Deriv(comp=[0, 1], orders=[(1, 0), (0, 1)], value=0.),
                     Deriv(comp=[2, 1, 1], orders=[(0, 1), (2, 0), (0, 2)], value=0., weights=[-1, 1, 1]),
                     # Deriv(comp=[2, 1, 1, 0], orders=[(1, 0), (0, 2), (2, 0), (1, 1)], value=0, weights=[-1, 2, 1, 1]),
                     # Deriv(comp=[2, ], orders=[(0, 1)], value=0., weights=[1]),
                     ]

            Xs_all[i] = Point(PT.NeumOffsetBC, X, value=value, derivatives=deriv)
        elif tag == "wall_left":
            deriv = [
                        # Deriv(comp=[0], orders=[(1, 0)], value=0, weights=[1]),

                        # Deriv(comp=[0, 1], orders=[(1, 0), (0, 1)], value=0.),
                        # Deriv(comp=[1], orders=[(0, 1)], value=0., weights=[1]),

                # Deriv(comp=[2, 0, 0, 1], orders=[(1, 0), (2, 0), (0, 2), (1, 1)], value=0, weights=[-1, 2, 1, 1]),
                Deriv(comp=[2, 0, 0], orders=[(1, 0), (2, 0), (0, 2)], value=0, weights=[-1, 1, 1]),

                        # Deriv(comp=[2, 1, 1], orders=[(0, 1), (2, 0), (0, 2)], value=0, weights=[-1, 1, 1]),
                        Deriv(comp=[1], orders=[(1, 0)], value=0, weights=[1]),

                # Pressure
                        Deriv(comp=[2], orders=[(0, 0)], value=1.),
                     ]
            Xs_all[i] = Point(PT.NeumOffsetBC, X, value=[0, 0, 1], derivatives=deriv)

        elif tag == "wall_right":
            deriv = [
                        # Deriv(comp=[0], orders=[(1, 0)], value=0., weights=[1]),
                        # Deriv(comp=[1, 1], orders=[(1, 0), (0, 1)], value=0., weights=[n_hat[0]]),

                        # Deriv(comp=[0, 2], orders=[(1, 0), (0, 0)], value=0, weights=[1, -1]),
                        # Deriv(comp=[0, 1], orders=[(0, 1), (1, 0)], value=0, weights=[1, 1]),

                        # Deriv(comp=[2, 0, 0], orders=[(1, 0), (2, 0), (0, 2)], value=0, weights=[-1, 1, 1]),
                Deriv(comp=[2, 0, 0, 1], orders=[(1, 0), (2, 0), (0, 2), (1, 1)], value=0, weights=[-1, 2, 1, 1]),

                        # Deriv(comp=[2, 1, 1], orders=[(0, 1), (2, 0), (0, 2)], value=0, weights=[-1, 1, 1]),
                        Deriv(comp=[1], orders=[(1, 0)], value=0, weights=[1]),
                        # Pressure
                        Deriv(comp=[2], orders=[(0, 0)], value=0.),
                     ]
            Xs_all[i] = Point(PT.NeumOffsetBC, X, value=value, derivatives=deriv)

        elif tag == "circle":
            deriv = [   Deriv(comp=[0], orders=[(0, 0)], value=0.),
                        Deriv(comp=[1], orders=[(0, 0)], value=0.),

                        # Deriv(comp=[2, 2], orders=[(1, 0), (0, 1)], value=0., weights=[n_hat[0], n_hat[1]]),
                        Deriv(comp=[2, 2, 0, 0, 1, 1, 1, 0], orders=[(1, 0), (0, 1), (2, 0), (0, 2), (1, 1), (2, 0), (0, 2), (1, 1)], value=0,
                            weights=[-n_hat[0], -n_hat[1], n_hat[0], n_hat[0], n_hat[0], n_hat[1], n_hat[1], n_hat[1]]),
                     ]
            Xs_all[i] = Point(PT.NeumOffsetBC, X, value=value, derivatives=deriv)
        else:
            raise ValueError(f"Unknown point tag {tag}")

    c_print(f'n_points: {len(Xs_all)}, n_bc: {len(bc_edges)}', color="bright_green")
    u_graph = UGraph(Xs_all, N_component=N_comp, grad_acc=2, max_degree=2, tri=triangles, device=cfg.DEVICE)

    with open("save_u_graph.pth", "wb") as f:
        torch.save((u_graph, triangles), f)

    # exit("Done")
    return u_graph, triangles

def load_graph(cfg):
    u_graph, triangles = torch.load("save_u_graph.pth")
    return u_graph, triangles


def true_pde():
    cfg = Config()
    u_graph, triangles = load_graph(cfg)
    # u_graph, triangles = mesh_graph(cfg)
    # u_graph = new_graph(cfg)

    us_all, _ = u_graph.get_all_us_Xs()
    derivs = u_graph.deriv_calc_eval.derivative(us_all)

    pde_fn = Fluid(cfg, device=cfg.DEVICE)
    pde_adj = NeuralPDEGraph(pde_fn, u_graph, cfg, DummyLoss(), triangles)
    pde_adj.forward_solve()

    pde_adj.plot_interp()
    # pde_adj.plot_derivs((1, 0))
    # pde_adj.plot_derivs((0, 1))

    us_all, Xs_all = u_graph.get_all_us_Xs()
    # torch.save((us_all, Xs_all), "us_all.pth")


# def main():
#     torch.set_printoptions(linewidth=200, precision=3)
#     cfg = Config()
#
#     # Make computation graph
#     Xs_perim = gen_perim(1, 1, 0.1)
#     Xs_bulk = test_grid(0.02, 0.98, torch.tensor([12, 12]), device="cpu")
#     Xs_bc = [Point(P_Types.FIX, X, 0.) for X in Xs_perim]
#     Xs_bulk = [Point(P_Types.NORMAL, X, 0.) for X in Xs_bulk]
#     Xs_all = {i: X for i, X in enumerate(Xs_bc + Xs_bulk)}
#     u_graph = UGraph(Xs_all, device=cfg.DEVICE)
#
#
#     pde_fn = LearnedFunc(cfg, device=cfg.DEVICE)
#     optim = torch.optim.Adam(pde_fn.parameters(), lr=0.5)
#
#     pde_adj = NeuralPDEGraph(pde_fn, u_graph, cfg, DummyLoss())
#
#     for i in range(5):
#         pde_adj.forward_solve()
#
#         loss = pde_adj.adjoint_solve()
#         pde_adj.backward()
#
#         print(f'{loss = :.3g}')
#         for n, p in pde_fn.named_parameters():
#             print(f'p = {p.data.cpu()}')
#             print(f'grad = {p.grad.data.cpu()}')
#
#         optim.step()
#         optim.zero_grad()
#
#     us, Xs = u_graph.us, u_graph.Xs
#
#     plot_interp_graph(Xs, us)


if __name__ == "__main__":
    setup_logging(debug=True)
    # torch.manual_seed(1)

    true_pde()


