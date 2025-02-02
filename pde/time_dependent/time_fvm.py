import torch
from cprint import c_print

from pde.graph_grid.graph_store import Point,  Deriv, T_Point
from pde.graph_grid.graph_store import P_TimeTypes as TT
from pde.config import Config
from pde.mesh_generation.generate_mesh import gen_mesh_fvm
from pde.graph_grid.graph_utils import plot_points, plot_interp_graph, plot_edges
from pde.time_dependent.time_cfg import ConfigTime
from pde.time_dependent.U_time_graph import UGraphTime, UTemp

from matplotlib import pyplot as plt

class FVMNodes:
    centroids: torch.Tensor  # shape = (n_triangles, 2)
    values: torch.Tensor  # shape = (n_triangles, N_component)

    def __init__(self, centroids, values):
        self.centroids = centroids
        self.values = values

class FVMEdges:
    normals: torch.Tensor  # shape = (n_edges, 2)

    tri_to_edge: torch.Tensor  # shape = (n_triangles, 3)
    edge_to_tri: dict[int, torch.Tensor]  # shape = {n_edges}[anit_idx, para_idx], ordered so triangle parallel to edge normal comes last, antiparallel first.
    edge_to_tri_w: dict[int, torch.Tensor]  # shape = {n_edges}[w_anti, w_para]

    bc_edge_mask: torch.Tensor  # shape = (n_edges)

    fluxes: torch.Tensor  # shape = (n_edges, N_component)

    def __init__(self, n_edges, n_triangles, n_comp, normals, edge_to_tri, edge_to_tri_w, bc_edge_mask):
        self.n_edges = n_edges
        self.n_triangles = n_triangles
        self.n_comp = n_comp

        self.normals = normals
        self.edge_to_tri = edge_to_tri
        self.edge_to_tri_w = edge_to_tri_w
        self.bc_edge_mask = bc_edge_mask

        self.fluxes = torch.empty(n_edges, n_comp)

    def edge_fluxes(self, Us, bc_flux):
        """ Compute fluxes for each edge.
            Us.shape = (n_triangles, N_component)
        """
        flux_vals = []
        for edge, tri in self.edge_to_tri.items():
            if tri.shape[0] == 1:
                continue
            w = self.edge_to_tri_w[edge]
            Us_edge = Us[tri]       # shape = [2, N_component]
            fluxes = w[0] * Us_edge[0] + w[1] * Us_edge[1]
            flux_val = torch.sum(fluxes * self.normals[edge], dim=-1)

            flux_vals.append(flux_val)

        flux_vals = torch.stack(flux_vals)
        self.fluxes[~self.bc_edge_mask] = flux_vals
        self.fluxes[self.bc_edge_mask] = bc_flux




class FVMMesh:
    normals: torch.Tensor  # shape = (n_edges, 2)
    lengths: torch.Tensor  # shape = (n_edges)
    centroids: torch.Tensor  # shape = (n_triangles, 2)


    def __init__(self, vertices, triangles, edges, bc_edge_mask, device="cuda"):
        self.vertices = vertices
        self.triangles = triangles
        self.edges = edges
        self.bc_edge_mask = bc_edge_mask
        self.device = device

        self._compute_edge_props(vertices, triangles, edges)

    def _compute_edge_props(self, vertices, triangles, edges):
        # Compute edge normals and lengths
        edge_vertex = vertices[edges]
        edge_vectors = edge_vertex[:, 1] - edge_vertex[:, 0]
        #lengths = torch.norm(edge_vectors, dim=1, keepdim=True)
        normals = torch.stack([edge_vectors[:, 1], -edge_vectors[:, 0]], dim=1)
        self.normals = normals                          # shape = [n_edges, 2]
        midpoints = torch.mean(edge_vertex, dim=1)      # shape = [n_edges, 2]

        # Triangle area and centroid
        tri_points = vertices[triangles]
        self.centroids = torch.mean(tri_points, dim=1)  # shape = [n_triangles, 2]

        tri_to_edge = self._get_tri_edges(triangles, edges) # shape = [n_triangles, 3]
        unique_edges, inverse_indices = torch.unique(tri_to_edge, sorted=True, return_inverse=True)
        edge_to_tri, tri_edge_idxs = {}, {}
        for edge in unique_edges:
            pos = (edge == tri_to_edge).nonzero()

            edge_to_tri[edge.item()] = pos[:, 0]
            tri_edge_idxs[edge.item()] = pos[:, 1]



        # Sort triangle in order of edge signed direction
        tri_edge_signs = self._tri_edge_sign(self.centroids, edge_vectors, midpoints, tri_to_edge, self.normals)
        self.tri_edge_sign = tri_edge_signs
        # ORDER: [-, +], so edge normal parallel to center comes last.
        edge_to_tri_ordered = {}
        p_m, m_p = torch.tensor([1, -1]), torch.tensor([-1, 1])
        for edge in edge_to_tri.keys():
            tri_idx = edge_to_tri[edge]
            tri_edge = tri_edge_idxs[edge]

            order = tri_edge_signs[tri_idx, tri_edge]

            # Boundary edges only have 1 triangle
            if order.shape[0] == 1:
                assert self.bc_edge_mask[edge] == True, "Inconsistent boundary bug"
                if order.item() == 1:
                    edge_to_tri_ordered[edge] = tri_idx
                elif order.item() == -1:
                    edge_to_tri_ordered[edge] = tri_idx
            else:
                if torch.all(order == p_m):
                    edge_to_tri_ordered[edge] = torch.flip(tri_idx, dims=[0])
                elif torch.all(order == m_p):
                    edge_to_tri_ordered[edge] = tri_idx

        # Compute distance from triangle centroid to edge midpoint
        # weight = [d_far / (d_far + d_near)]
        edge_to_tri_w = {}
        for edge, tri in edge_to_tri_ordered.items():
            if tri.shape[0] == 1:
                continue
            v = midpoints[edge] - self.centroids[tri]
            n = self.normals[edge]
            n_hat = n / torch.norm(n, dim=-1, keepdim=True)
            d = torch.abs(torch.sum(v * n_hat, dim=-1))

            w_anti = d[1] / (d[0] + d[1])
            w_para = d[0] / (d[0] + d[1])
            edge_to_tri_w[edge] = torch.stack([w_anti, w_para], dim=0)

            print()
            print(f'{d = }')
            print(f'{edge_to_tri_w[edge]}')
        self.edge_to_tri_w = edge_to_tri_w
        exit(4)

        # Flux interpolation factor for converting cell value to flux.
        flux_interp = []
        for edge, tri in edge_to_tri.items():
            if tri.shape[0] == 1:
                continue

            print(edge, tri)


        plot_edges(vertices, edges, title="Edges")
        plot_points(self.centroids, torch.zeros_like(self.centroids[:, 0]), title="Centroids")
        exit(7)


        return normals

    def _tri_edge_sign(self, centroids, edge_vectors, midpoints, tri_to_edge, normals):
        signs = []
        for edge, center in zip(tri_to_edge, centroids):
            edge_vect = edge_vectors[edge]      # shape = [3, 2]
            midpoint = midpoints[edge]      # shape = [3, 2]
            normal = normals[edge]          # shape = [3, 2]

            p_diff = midpoint - center      # shape = [3, 2]
            p_diff = p_diff / torch.norm(p_diff, dim=-1, keepdim=True)
            edge_vect = edge_vect / torch.norm(edge_vect, dim=-1, keepdim=True)

            # (midpt-center) X edge_vect
            cross = edge_vect[:, 0] * p_diff[:, 1] - edge_vect[:, 1] * p_diff[:, 0]
            sign_X = -torch.sign(cross)
            # Or normals dot (center - midpoint)
            dot = torch.sum(normal * p_diff, dim=-1)
            sign_dot = torch.sign(dot)

            assert torch.all(sign_X == sign_dot), f'{sign_X = }, {sign_dot = }'

            signs.append(sign_dot)
        signs = torch.stack(signs).long()

        return signs



    def _get_tri_edges(self, triangles, edges):
        """
            Compute which edges belong to each triangle
            triangles.shape = (n_triangles, 3)
            edges.shape = (n_edges, 2)
        """
        # 1) Normalize each edge (sort nodes in ascending order).
        # -------------------------------------------------------
        # edges_sorted will be shape [m, 2] with each row sorted.
        edges_sorted, _ = edges.sort(dim=1)

        # 2) Build a lookup: (nodeA, nodeB) -> edge_index
        # -----------------------------------------------
        edge_dict = {}
        for idx, e in enumerate(edges_sorted):
            # Make a tuple key (nodeA, nodeB)
            key = (e[0].item(), e[1].item())
            edge_dict[key] = idx

        # 3) For each triangle, find the 3 edges
        # --------------------------------------
        # We'll create a result tensor of shape [num_triangles, 3],
        # each row will store the indices of the 3 edges of that triangle.

        tri_to_edge = []
        for tri in triangles:
            # Extract triangle nodes (v0, v1, v2)
            v0 = tri[0].item()
            v1 = tri[1].item()
            v2 = tri[2].item()

            # Sort each pair so we can look it up in the edge_dict
            e1 = tuple(sorted((v0, v1)))
            e2 = tuple(sorted((v1, v2)))
            e3 = tuple(sorted((v2, v0)))

            # Get the edge indices
            edge_indices = [
                edge_dict[e1],
                edge_dict[e2],
                edge_dict[e3]
            ]
            tri_to_edge.append(edge_indices)

        tri_to_edge = torch.tensor(tri_to_edge)
        return tri_to_edge



def mesh_graph(cfg):
    N_comp = 2

    xmin, xmax = 0, 3
    ymin, ymax = 0.0, 1.5
    Xs, tri_idx, (int_edgs, bound_edgs) = gen_mesh_fvm(xmin, xmax, ymin, ymax, areas=[6e-3, 10e-3])
    Xs = torch.from_numpy(Xs).float()
    tri_idx = torch.from_numpy(tri_idx).int()
    int_edgs, bound_edgs = torch.from_numpy(int_edgs), torch.from_numpy(bound_edgs)
    all_edgs = torch.cat([int_edgs, bound_edgs], dim=0)
    bc_edge_mask = torch.cat([torch.zeros_like(int_edgs[:, 0], dtype=torch.bool), torch.ones_like(bound_edgs[:, 0], dtype=torch.bool)], dim=0)
    mesh = FVMMesh(Xs, tri_idx, all_edgs, bc_edge_mask)


    exit(4)
    c_print(f'Number of mesh points: {len(Xs)}', "green")

    # Set up time-graph
    setup_T = []
    for i, (X, tag) in enumerate(zip(Xs, p_tags)):
        if tag == "Wall" or tag == "Left" or tag == "Right":
            value = [0 for _ in range(N_comp)]
            setup_T.append(T_Point([TT.FIXED], X, init_val=value))
        elif tag == "Normal":
            x, y = X
            a, b = x-1.5, y-0.75
            if a**2 + b**2 < 0.25:
                value = [1]
            else:
                value = [0]
            setup_T.append(T_Point([TT.NORMAL], X, init_val=value))
        else:
            raise ValueError(f"Unknown tag {tag}")

    setup_T = {i: point for i, point in enumerate(setup_T)}
    u_graph_time = UGraphTime(setup_T, N_component=N_comp, grad_acc=2, device=cfg.DEVICE)
    # plot_points(u_graph_time._Xs, u_graph_time.dirich_mask[:, 0], title="grad mask")
    with open("./save_u_graph_T2.pth", "wb") as f:
        torch.save(u_graph_time, f)
    return u_graph_time


def load_graph(cfg):
    u_graph_T = torch.load("save_u_graph_T2.pth", weights_only=False)
    return u_graph_T

class PDEFn:
    def __init__(self, u_graph_T: UGraphTime, cfg_in, cfg_T):
        self.u_graph_T = u_graph_T
        self.cfg_in = cfg_in
        self.cfg_T = cfg_T

        self.dt = cfg_T.dt

    def solve(self, t, step_no):
        us = self.u_graph_T.get_all_us_Xs()[0]

        grads = self.u_graph_T.get_grads()
        us_t = grads[(0, 0)]
        # dudt = - laplacian(u)
        laplacian = grads[(2, 0)] + grads[(0, 2)]

        #u_t+1 = u_t + dt * dudt
        u_t_1 = us_t + self.dt * laplacian


        self.u_graph_T.set_grid(u_t_1)



class TimePDEBase:
    """ Have a main PDE U_graph that is updated with every t. For update:
        1) Clone U_graph.
        1.1) Clone U_graph if we want state to be saved for later
        2) Solve PDE with U_graph
        3) Update time-PDE with new values.

        Assume graph doesn't change so deriv calc and intermediate sparse caches can be kept.
        """
    cfg_T: ConfigTime
    cfg_in: Config

    PDE_timefn: PDEFn

    u_graph_main: UGraphTime
    u_saves: dict[int, UTemp]

    def __init__(self, u_graph_T: UGraphTime, cfg_T: ConfigTime, cfg_in: Config):
        """ u_graph_T: Time graph.
            u_graph_PDE: Graph for internal PDE solver.
        """
        self.u_graph_T = u_graph_T
        self.cfg_T = cfg_T
        self.cfg_in = cfg_in
        self.u_saves = {}
        self.Xs = None
        self.PDE_timefn = PDEFn(u_graph_T, cfg_in, cfg_T) #ExplicitNS(u_graph_T, u_graph_PDE, cfg_in, cfg_T)

        self.device = "cuda"
        self.dtype = torch.float32

    def solve(self):
        cfg_T = self.cfg_T

        self.Xs = self.u_graph_T.get_all_us_Xs()[1]
        self.u_saves[0] = self.u_graph_T.get_all_us_Xs()[0].clone()
        timesteps = torch.linspace(cfg_T.time_domain[0], cfg_T.time_domain[1], cfg_T.timesteps * cfg_T.substeps, dtype=self.dtype)
        for step_num, t in enumerate(timesteps):
            print(f'\n{step_num = }, t = {t.item():.3g}')

            # if step_num == 5:
            #     dirich_mask = self.u_graph_T.dirich_mask
            #     dirich_values = torch.zeros_like(self.u_graph_T._us)[dirich_mask]
            #     print(f'{dirich_values.shape = }')
            #     self.u_graph_T.set_bc(dirich_bc=dirich_values)

            self.PDE_timefn.solve(t, step_num)

            if step_num % cfg_T.substeps == 0:
                self.u_saves[step_num+1] = self.u_graph_T.get_all_us_Xs()[0].clone()

            if step_num == 50:
                break

        for step, us in self.u_saves.items():
            plot_interp_graph(self.Xs, us[:, 0], title=f"Vx Step {step}")


    def update_boundary(self):
        pass


def main():
    from pde.utils import setup_logging

    setup_logging(debug=False)

    cfg = Config()
    time_cfg= ConfigTime()
    c_print(f'{time_cfg.dt = }', color="bright_magenta")

    #u_g_T = load_graph(cfg)
    u_g_T = mesh_graph(cfg)

    time_pde = TimePDEBase(u_g_T, time_cfg, cfg)
    time_pde.solve()

    # saved_graphs = time_pde.u_saves
    # for t, graph in saved_graphs.items():
    #     us, Xs = graph.us, graph.Xs
    #     plot_interp_graph(Xs, us[:, 0], title=f"t={t :.4g}")


if __name__ == "__main__":
    main()