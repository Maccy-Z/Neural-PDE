import torch
from cprint import c_print
import numpy as np

from pde.graph_grid.graph_store import Point, Deriv
from pde.graph_grid.graph_store import P_Types as PT
from pde.graph_grid.U_graph import UGraph
from pde.graph_grid.graph_utils import test_grid, gen_perim
from pde.config import Config
from pde.NeuralPDE_Graph import NeuralPDEGraph
from pdes.PDEs import Poisson, MagneticField
from pde.utils import setup_logging
from pde.loss import DummyLoss
from pde.mesh_generation.generate_mesh import gen_points_full



def boundary_normals(points, triangles, bc_edges):
    """
    points: (nV, 2) array of vertex coordinates
    triangles: (nT, 3) array of integer vertex indices
    bc_edges: (nE, 2) array of integer vertex indices for boundary edges
    returns: (nE, 2) array of outward unit normals
    """
    # 1) Build adjacency: map each undirected edge to its triangle and opposite vertex
    edge_to_tri = {}
    for tri in triangles:
        for a, b, c in [(tri[0], tri[1], tri[2]),
                        (tri[1], tri[2], tri[0]),
                        (tri[2], tri[0], tri[1])]:
            edge = tuple(sorted((a, b)))
            edge_to_tri[edge] = c

    normals = np.zeros((len(bc_edges), 2))
    for idx, (i, j) in enumerate(bc_edges):
        # 2) get coordinates
        pi, pj = points[i], points[j]
        # 3) find interior vertex
        k = edge_to_tri[tuple(sorted((i, j)))]
        pk = points[k]
        # tangent
        t = pj - pi
        L = np.linalg.norm(t)
        if L == 0:
            raise ValueError(f"Zero length edge at index {idx}")
        # two candidate normals
        n1 = np.array([ t[1], -t[0] ]) / L
        # midpoint and interior direction
        m = 0.5 * (pi + pj)
        v_int = pk - m
        # pick outward: we want n·v_int < 0
        if np.dot(n1, v_int) < 0:
            normals[idx] = n1
        else:
            normals[idx] = -n1

    return normals

def mesh_graph(cfg):
    cfg = Config()
    N_comp = 1
    points, triangles, p_tags, bc_edges = gen_points_full()
    normals = boundary_normals(points, triangles, bc_edges)
    points = torch.from_numpy(points).float()

    # Process boundary points
    bc_edges = torch.from_numpy(bc_edges)
    bc_points = torch.unique(bc_edges.flatten(), dim=0)

    deriv = [Deriv(comp=[0], orders=[(1, 0)], value=0.)]#, Deriv(comp=[1], orders=[(1, 0)], value=1.)]
    Xs_all = {}
    for i, (point, tag) in enumerate(zip(points, p_tags)):
        value = [0. for _ in range(N_comp)]
        if tag == "Normal":
            Xs_all[i] = Point(PT.Normal, point, value=value)
            assert i not in bc_points, "Normal point is also a boundary point"
        elif tag == "wall_bottom":
            Xs_all[i] = Point(PT.DirichBC, point, value=value, derivatives=deriv)
        elif tag == "wall_top":
            Xs_all[i] = Point(PT.DirichBC, point, value=value, derivatives=deriv)
        elif tag == "wall_left":
            Xs_all[i] = Point(PT.DirichBC, point, value=value, derivatives=deriv)
        elif tag == "wall_right":
            value = [1. for _ in range(N_comp)]
            Xs_all[i] = Point(PT.DirichBC, point, value=value, derivatives=deriv)
        elif tag == "circle":
            n_hat = normals[i].tolist()

            deriv = [Deriv(comp=[0, 0], orders=[(1, 0), (0, 1)], value=-2., weights=[n_hat[0], n_hat[1]])]
            Xs_all[i] = Point(PT.NeumOffsetBC, point, value=value, derivatives=deriv)
        else:
            raise ValueError(f"Unknown point tag {tag}")

    c_print(f'n_points: {len(Xs_all)}, n_bc: {len(bc_edges)}', color="bright_green")
    u_graph = UGraph(Xs_all, N_component=N_comp, grad_acc=4, tri=triangles, device=cfg.DEVICE)

    with open("save_u_graph.pth", "wb") as f:
        torch.save(u_graph, f)

    # exit("Done")
    return u_graph, triangles

def new_graph(cfg):
    cfg = Config()
    N_comp = 2

    n_grid = 20
    spacing = 1/(n_grid + 1)


    Xs_perim = gen_perim(1, 1, spacing)
    perim_mask = (Xs_perim[:, 1] > 0) & (Xs_perim[:, 1] < 1) & (Xs_perim[:, 0] ==0)
    Xs_neumann = Xs_perim[perim_mask]
    #print(Xs_neumann)
    Xs_dirich = Xs_perim[~perim_mask]

    Xs_ghost = Xs_neumann.clone()
    Xs_ghost[:, 0] = Xs_ghost[:, 0] - spacing
    Xs_bulk = test_grid(spacing, (1- spacing), torch.tensor([n_grid, n_grid]), device="cpu")

    deriv = [Deriv(comp=[0], orders=[(1, 0)], value=1.), Deriv(comp=[1], orders=[(1, 0)], value=0.)]
    deriv_test = [Deriv(comp=[0], orders=[(0, 0)], value=0.), Deriv(comp=[1], orders=[(0, 0)], value=0.)]

    # Xs_fix = [Point(PT.DirichBC, X, value=[0. for _ in range(N_comp)]) for X in Xs_dirich]
    Xs_fix = [Point(PT.NeumOffsetBC, X, value=[0. for _ in range(N_comp)], derivatives=deriv_test) for X in Xs_dirich]
    Xs_deriv = [Point(PT.NeumCentralBC , X, value=[0. for _ in range(N_comp)], derivatives=deriv) for X in Xs_neumann]
    Xs_ghost = [Point(PT.Ghost, X, value=[0. for _ in range(N_comp)]) for X in Xs_ghost]
    Xs_bulk = [Point(PT.Normal, X, value= [0. for _ in range(N_comp)]) for X in Xs_bulk]


    Xs_all = {i: X for i, X in enumerate(Xs_deriv + Xs_fix + Xs_bulk + Xs_ghost)}
    u_graph = UGraph(Xs_all, N_component=N_comp, grad_acc=4, device=cfg.DEVICE)

    with open("save_u_graph.pth", "wb") as f:
        torch.save(u_graph, f)

    return u_graph


def load_graph(cfg):
    u_graph = torch.load("save_u_graph.pth")
    return u_graph


def true_pde():
    cfg = Config()
    # u_graph = load_graph(cfg)
    u_graph, triangles = mesh_graph(cfg)
    # u_graph = new_graph(cfg)
    pde_fn = Poisson(cfg, device=cfg.DEVICE)
    pde_adj = NeuralPDEGraph(pde_fn, u_graph, cfg, DummyLoss(), triangles)

    pde_adj.forward_solve()

    us, Xs = u_graph.get_all_us_Xs()

    pde_adj.plot_interp()


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
    torch.manual_seed(1)

    true_pde()


