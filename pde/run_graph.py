import torch
from cprint import c_print
import numpy as np
from collections import defaultdict

from pde.graph_grid.graph_store import Point, Deriv
from pde.graph_grid.graph_store import P_Types as PT
from pde.graph_grid.U_graph import UGraph
from pde.config import Config
from pde.NeuralPDE_Graph import NeuralPDEGraph
from pdes.PDEs import HeatLearned, Fluid, FluidLearned, Heat, NNFunc
from pde.utils import setup_logging
from pde.loss import DummyLoss, MSELoss2, MaskLoss
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


def mesh_heat(cfg):
    cfg = Config()
    N_comp = 3
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
                 Deriv(comp=[1], orders=[(0, 0)], value=0, weights=[1]),
                 Deriv(comp=[2], orders=[(0, 0)], value=0, weights=[1]),
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
                     Deriv(comp=[1, 1], orders=[(1, 0), (0, 1)], value=-2., weights=[n_hat[0], n_hat[1]]),
                     Deriv(comp=[2, 2], orders=[(1, 0), (0, 1)], value=-3., weights=[n_hat[0], n_hat[1]])
                     ]

            Xs_all[i] = Point(PT.NeumOffsetBC, X, value=value, derivatives=deriv)
        else:
            raise ValueError(f"Unknown point tag {tag}")

    c_print(f'n_points: {len(Xs_all)}, n_bc: {len(bc_edges)}', color="bright_green")
    U_graph = UGraph(Xs_all, N_component=N_comp, grad_neigh=25, max_degree=2, tri=triangles, device=cfg.DEVICE)

    with open("save_u_graph.pth", "wb") as f:
        torch.save((U_graph, triangles), f)

    # exit("Done")
    return U_graph, triangles


def mesh_graph(cfg):
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
        x, y = X

        value = [0*x for _ in range(N_comp)]

        if tag == "Normal":
            Xs_all[i] = Point(PT.Normal, X, value=value)
            assert i not in bc_points, "Normal point is also a boundary point"
            continue

        # Boundary conditions
        n_hat = normals[i].tolist()
        # wall_deriv = Deriv(comp=[2, 2, 0, 0, 1, 1, 1, 0], orders=[(1, 0), (0, 1), (2, 0), (0, 2), (1, 1), (2, 0), (0, 2), (1, 1)], value=0,
        #       weights=[-n_hat[0], -n_hat[1], n_hat[0], n_hat[0], n_hat[0], n_hat[1], n_hat[1], n_hat[1]])
        wall_deriv = Deriv(comp=[2, 2, 0, 0, 1, 1], orders=[(1, 0), (0, 1), (2, 0), (0, 2), (2, 0), (0, 2)], value=0,
                           weights=[-n_hat[0], -n_hat[1], n_hat[0], n_hat[0], n_hat[1], n_hat[1]])

        if tag == "wall_bottom" or tag == "wall_top" or y <= 0 or y >= 1.5:
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
                        Deriv(comp=[2], orders=[(0, 0)], value=1.),
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

        elif tag == "circle":
            deriv = [   Deriv(comp=[0], orders=[(0, 0)], value=0.),
                        Deriv(comp=[1], orders=[(0, 0)], value=0.),

                        # Deriv(comp=[2, 2], orders=[(1, 0), (0, 1)], value=0., weights=[n_hat[0], n_hat[1]]),
                        wall_deriv,
                     ]
            Xs_all[i] = Point(PT.NeumOffsetBC, X, value=value, derivatives=deriv)
        else:
            raise ValueError(f"Unknown point tag {tag}")

    c_print(f'n_points: {len(Xs_all)}, n_bc: {len(bc_edges)}', color="bright_green")
    U_graph = UGraph(Xs_all, N_component=N_comp, grad_neigh=25, max_degree=2, tri=triangles, device=cfg.DEVICE)

    with open("save_u_graph.pth", "wb") as f:
        torch.save((U_graph, triangles), f)

    # exit("Done")
    return U_graph, triangles


def load_graph(cfg)-> tuple[UGraph, torch.Tensor]:
    u_graph, triangles = torch.load("save_u_graph.pth", weights_only=False)
    return u_graph, triangles


def true_pde():
    cfg = Config()
    # U_graph, triangles = load_graph(cfg)
    U_graph, triangles = mesh_graph(cfg)
    # U_graph, triangles = mesh_heat(cfg)

    Us_all, _ = U_graph.get_all_us_Xs()

    pde_fn = Fluid(cfg, device=cfg.DEVICE)
    # pde_fn = HeatLearned(cfg, device=cfg.DEVICE)

    # Us_target = U_graph.pde_mask.float()
    # Us_target = torch.repeat_interleave(Us_target, U_graph.N_comp, dim=0)
    loss_fn = DummyLoss()  # MaskLoss(Us_target)

    pde_adj = NeuralPDEGraph(pde_fn, U_graph, cfg, loss_fn, triangles)

    pde_adj.forward_solve()
    pde_adj.plot_interp(title="Initial solution")

    Us_all, updt_mask, _ = U_graph.get_us_mask()
    Us = Us_all[updt_mask]

    with open("./Us_solution.pth", "wb") as f:
        torch.save(Us, f)
    # print(f'{Us.shape = }, {Us_all.shape = }')

def optim_pde():
    cfg = Config()
    # U_graph, triangles = load_graph(cfg)
    U_graph, triangles = mesh_graph(cfg)
    # U_graph, triangles = mesh_heat(cfg)

    Us_true = torch.load("./Us_solution.pth")
    U_graph.set_grid(Us_true.clone())
    pde_fn = NNFunc(cfg, device=cfg.DEVICE)
    # pde_fn = HeatLearned(cfg, device=cfg.DEVICE)

    loss_fn = MSELoss2(Us_true)

    pde_adj = NeuralPDEGraph(pde_fn, U_graph, cfg, loss_fn, triangles)

    # pde_adj.forward_solve()
    pde_adj.forward_solve()

    pde_adj.plot_interp(title="Initial solution")
    # exit(3)
    # exit("Done with forward solve")

    optim = torch.optim.Adam(pde_fn.parameters(), lr=0.02, betas=(0.9, 0.99))

    for i in range(100):
        # U_graph.set_grid(Us_true.clone())

        converged = pde_adj.forward_solve()
        loss = pde_adj.adjoint_solve()
        print(f'{loss = :.5g}')

        for param_group in optim.param_groups:
            if converged['iter'] < 2:
                param_group['lr'] = min(param_group['lr'] * 1.1, 1000000.)
                print("lr = ", param_group['lr'])

            elif converged['iter'] >= 5:
                param_group['lr'] = max(param_group['lr'] * 0.5, 0.00001)
                print(param_group['lr'])

            # else:
            #     param_group['lr'] = param_group['lr'] * 0.9
            # print(converged['iter'], param_group['lr'])

        pde_adj.backward()
        # torch.nn.utils.clip_grad_value_(pde_fn.parameters(), clip_value=0.25)
        optim.step()

        # for n, p in pde_fn.named_parameters():
        #     c_print(f'{n=}: {p.cpu() = },', color="green")

        if i % 10 == 0:
            pde_adj.plot_interp()

        optim.zero_grad()


def plot_grads():
    from matplotlib import pyplot as plt

    cfg = Config()
    # U_graph, triangles = load_graph(cfg)
    U_graph, triangles = mesh_graph(cfg)
    # U_graph, triangles = mesh_heat(cfg)

    us_all, _ = U_graph.get_all_us_Xs()

    pde_fn = FluidLearned(cfg, device=cfg.DEVICE)
    # pde_fn = HeatLearned(cfg, device=cfg.DEVICE)

    Us_true = torch.load("./Us_solution.pth")
    loss_fn = MSELoss2(Us_true)

    # loss_fn = DummyLoss()

    pde_adj = NeuralPDEGraph(pde_fn, U_graph, cfg, loss_fn, triangles)

    pde_adj.forward_solve()
    pde_adj.plot_interp(title="Initial solution")

    # exit("Done with forward solve")

    losses, grads = [], []
    X_range = torch.linspace(0, 1000, 50)
    for i in X_range:
        # U_graph.reset()
        pde_fn.a.data[0] = i.to(device=cfg.DEVICE)

        pde_adj.forward_solve()
        loss = pde_adj.adjoint_solve().detach().cpu()
        pde_adj.backward()


        grad = pde_fn.a.grad[0].cpu()
        losses.append(loss)
        grads.append(grad)

        pde_fn.zero_grad()

        # break

    losses = torch.tensor(losses)
    grads = torch.tensor(grads)

    plt.plot(X_range, losses, marker='o')
    plt.xlabel('Loss')
    plt.show()

    plt.plot(X_range, grads, marker='o')
    plt.xlabel('Gradient')
    plt.show()
    print(losses)
    print(grads)


if __name__ == "__main__":
    setup_logging(debug=1)
    # torch.manual_seed(1)

    optim_pde()
    # plot_grads()
    # true_pde()

