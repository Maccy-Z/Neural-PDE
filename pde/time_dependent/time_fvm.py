import torch
from cprint import c_print
from matplotlib import pyplot as plt

from pde.config import Config
from pde.mesh_generation.generate_mesh import gen_mesh_fvm
from pde.graph_grid.graph_utils import plot_points, plot_interp_graph, plot_edges
from pde.time_dependent.time_cfg import ConfigTime

class FVMMesh:
    n_cells: int
    n_edges: int
    n_bc_edge: int

    areas: torch.Tensor  # shape = (n_cells)
    normals: torch.Tensor  # shape = (n_edges, 2)
    lengths: torch.Tensor  # shape = (n_edges)
    centroids: torch.Tensor  # shape = (n_cells, 2)
    tri_to_edge: torch.Tensor  # shape = (n_cells, 3)
    tri_edge_sign: torch.Tensor  # shape = (n_cells, 3)
    edge_to_tri: dict[int, torch.Tensor]  # shape = {n_edges}[2]        # Mapping edge to triangle indices. Ordered [antiparallel, parallel] to edge normal.
    edge_to_tri_w: dict[int, torch.Tensor]  # shape = {n_updt_edge}[2]

    def __init__(self, vertices, triangles, edges, bc_edge_mask, device="cuda"):
        self.vertices = vertices
        self.triangles = triangles
        self.edges = edges
        self.bc_edge_mask = bc_edge_mask
        self.device = device

        self.n_cells = triangles.shape[0]
        self.n_edges = edges.shape[0]
        self.n_bc_edge = bc_edge_mask.sum().item()
        assert edges.shape[0] == bc_edge_mask.shape[0], f'Different number of edges from bc edge mask {edges.shape = }, {bc_edge_mask.shape = }'

        self._compute_edge_props(vertices, triangles, edges)

    def _compute_edge_props(self, vertices, triangles, edges):
        # Compute edge normals and lengths
        edge_vertex = vertices[edges]
        edge_vectors = edge_vertex[:, 1] - edge_vertex[:, 0]        # Ordering is used as edge index from here.
        normals = torch.stack([edge_vectors[:, 1], -edge_vectors[:, 0]], dim=1)
        self.normals = normals                          # shape = [n_edges, 2]
        midpoints = torch.mean(edge_vertex, dim=1)      # shape = [n_edges, 2]

        # Triangle area and centroid
        tri_points = vertices[triangles]
        self.areas = self._tri_area(tri_points)
        self.centroids = torch.mean(tri_points, dim=1)  # shape = [n_cells, 2]


        # Compute mapping of edges to triangles
        tri_to_edge = self._get_tri_edges(triangles, edges) # shape = [n_cells, 3]
        self.tri_to_edge = tri_to_edge
        unique_edges, _ = torch.unique(tri_to_edge, sorted=True, return_inverse=True)
        edge_to_tri, tri_edge_idxs = {}, {}
        for edge in unique_edges:
            pos = (edge == tri_to_edge).nonzero()

            edge_to_tri[edge.item()] = pos[:, 0]
            tri_edge_idxs[edge.item()] = pos[:, 1]

        # edges = tri_to_edge[481]
        # print(edges)
        # print(tri_points[481])
        # print(edge_vectors[edges])
        # print(self.centroids[481])
        # exit(7)

        # Sort triangle in order of edge signed direction
        self.tri_edge_signs = self._tri_edge_sign(self.centroids, edge_vectors, midpoints, tri_to_edge, self.normals)
        # ORDER: [-, +], so edge normal parallel to center comes last.
        edge_to_tri_ordered = {}
        p_m, m_p = torch.tensor([1, -1]), torch.tensor([-1, 1])
        for edge in edge_to_tri.keys():
            tri_idx = edge_to_tri[edge]
            tri_edge = tri_edge_idxs[edge]

            order = self.tri_edge_signs[tri_idx, tri_edge]

            # Boundary edges only have 1 triangle
            if order.shape[0] == 1:
                assert self.bc_edge_mask[edge] == True, "Inconsistent boundary bug"
                edge_to_tri_ordered[edge] = tri_idx
            else:
                if torch.all(order == p_m):
                    edge_to_tri_ordered[edge] = torch.flip(tri_idx, dims=[0])
                elif torch.all(order == m_p):
                    edge_to_tri_ordered[edge] = tri_idx
        self.edge_to_tri = edge_to_tri_ordered

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

        self.edge_to_tri_w = edge_to_tri_w

    def _tri_area(self, vertices):
        """ vertices.shape = (n_cells, 3, 2) """
        a = vertices[:, 0]
        b = vertices[:, 1]
        c = vertices[:, 2]

        # Compute the vectors for each triangle
        ab = b - a  # shape [n, 2]
        ac = c - a  # shape [n, 2]

        # Compute the 2D cross product (determinant) for each triangle
        cross = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]  # shape [n]

        # Triangle area is half the absolute value of the cross product
        area = 0.5 * torch.abs(cross)

        return area


    def _tri_edge_sign(self, centroids, edge_vectors, midpoints, tri_to_edge, normals):
        signs = []

        j = 0
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

            # if j == 15:
            #     print(edge, center)
            #     print(edge_vect)
            #     print(sign_dot)
            #     exit("Found it ")
            # j += 1



        signs = torch.stack(signs).long()
        return signs

    def _get_tri_edges(self, triangles, edges):
        """
            Compute which edges belong to each triangle
            triangles.shape = (n_cells, 3)
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


class FVMCells:
    values: torch.Tensor  # shape = (n_cells, N_component)
    areas: torch.Tensor  # shape = (n_cells)
    tri_to_edge: torch.Tensor  # shape = (n_cells, 3)
    tri_edge_sign: torch.Tensor  # shape = (n_cells, 3)

    def __init__(self, mesh: FVMMesh, n_component, init_val=None):
        self.tri_to_edge = mesh.tri_to_edge
        self.tri_edge_sign = mesh.tri_edge_signs.unsqueeze(-1)
        self.areas = mesh.areas
        n_cells = mesh.n_cells
        if init_val is None:
            self.values = torch.zeros(n_cells, n_component)
        else:
            assert init_val.shape == (n_cells, n_component), f'Incorrect us init shape {init_val.shape = }'
            self.values = init_val.clone()

    def update_cells(self, fluxes, dt):
        """ Update cell values using fluxes.
            fluxes.shape = (n_edges, N_component)

            du/dt = -div(flux) = -sum_i (sign_i * flux_i)
        """
        divs = []
        for i, tri in enumerate(self.tri_to_edge):

            divergence = torch.sum(self.tri_edge_sign[i] * fluxes[tri], dim=0) / self.areas[i]
            divs.append(divergence * dt)
            if i == 481:
                print(f'{i = }, div = {divergence}, signs = {self.tri_edge_sign[i].squeeze().tolist() }, fluxes = {fluxes[tri].squeeze()}, area = {self.areas[i]:.3g}')
            self.values[i] -= dt * divergence

        # divs = torch.stack(divs)
        # biggest = divs.abs().max()
        # if biggest > 10:
        #     print(torch.argmax(divs.abs()))
        #exit("Done step")

class FVMEdges:
    n_edges: int

    normals: torch.Tensor  # shape = (n_edges, 2)
    tri_to_edge: torch.Tensor  # shape = (n_cells, 3)
    edge_to_tri: dict[int, torch.Tensor]  # shape = {n_edges}[anit_idx, para_idx], ordered so triangle parallel to edge normal comes last, antiparallel first.
    edge_to_tri_w: dict[int, torch.Tensor]  # shape = {n_edges}[w_anti, w_para]
    bc_edge_mask: torch.Tensor  # shape = (n_edges)

    fluxes: torch.Tensor  # shape = (n_edges, N_component)

    def __init__(self, mesh: FVMMesh, n_comp):
        self.n_edges = mesh.n_edges

        self.normals = mesh.normals
        self.edge_to_tri = mesh.edge_to_tri
        self.edge_to_tri_w = mesh.edge_to_tri_w
        self.bc_edge_mask = mesh.bc_edge_mask

        self.fluxes = torch.empty(self.n_edges, n_comp)

    def edge_fluxes(self, Us, u_face_bc):
        """ Compute fluxes for each edge.
            Us.shape = (n_cells, N_component)
            Linear flux interpolation: flux = w_anti * U_anti + w_para * U_para

        """
        u_face = []
        for edge, tri in self.edge_to_tri.items():
            if tri.shape[0] == 1:
                continue
            w = self.edge_to_tri_w[edge]
            Us_edge = Us[tri]       # shape = [2, N_component]
            # Linear interpolate face values
            face_scal = w[0] * Us_edge[0] + w[1] * Us_edge[1]
            face_vect = face_scal * torch.tensor([1, 0])
            # Flux = face_val dot normal
            flux_val = torch.dot(face_vect, self.normals[edge])

            u_face.append(flux_val)

        u_face = torch.stack(u_face)    # shape = [n_updt_edges, N_component]
        self.fluxes[~self.bc_edge_mask] = u_face.unsqueeze(-1)
        self.fluxes[self.bc_edge_mask] = u_face_bc

        return self.fluxes


class FVMSolver:
    mesh: FVMMesh
    edges: FVMEdges
    cells: FVMCells

    def __init__(self, mesh: FVMMesh, n_comp, us_init=None):
        self.mesh = mesh
        self.edges = FVMEdges(mesh, n_comp)
        self.cells = FVMCells(mesh, n_comp, us_init)

        self.plot_flux(torch.zeros(mesh.n_edges, n_comp))
        self.plot_cells(self.cells.values[:, 0], title="Inital Cell Values")

        dt = 0.01
        for i in range(21):
            fluxes = self.edges.edge_fluxes(self.cells.values, torch.zeros(mesh.n_bc_edge, n_comp))
            self.cells.update_cells(fluxes, dt)

            if i % 1 == 0:
                #self.plot_flux(fluxes, title=f"Fluxes at t={i * dt :.2g}")
                self.plot_cells(self.cells.values[:, 0], title=f"Values at t={i * dt :.2g}")

    def plot_flux(self, fluxes, title="Fluxes"):
        plot_edges(self.mesh.vertices.cpu(), self.mesh.edges.cpu(), title=title, color=fluxes[:, 0].abs())

    def plot_cells(self, value, title="Cell Values"):
        plot_points(self.mesh.centroids.cpu(), value, title=title)



def mesh_graph(cfg):
    N_comp = 2

    xmin, xmax = 0, 3
    ymin, ymax = 0.0, 1.5
    Xs, tri_idx, (int_edgs, bound_edgs) = gen_mesh_fvm(xmin, xmax, ymin, ymax, areas=[10e-3, 15e-3])
    Xs = torch.from_numpy(Xs).float()
    tri_idx = torch.from_numpy(tri_idx).int()
    int_edgs, bound_edgs = torch.from_numpy(int_edgs), torch.from_numpy(bound_edgs)
    all_edgs = torch.cat([int_edgs, bound_edgs], dim=0)
    bc_edge_mask = torch.cat([torch.zeros_like(int_edgs[:, 0], dtype=torch.bool), torch.ones_like(bound_edgs[:, 0], dtype=torch.bool)], dim=0)


    mesh = FVMMesh(Xs, tri_idx, all_edgs, bc_edge_mask)

    cent_x = mesh.centroids[:, 0].unsqueeze(-1).clone()
    us_init = 3.5 - cent_x

    solver = FVMSolver(mesh, 1, us_init=us_init)

    exit(4)
    c_print(f'Number of mesh points: {len(Xs)}', "green")



def load_graph(cfg):
    u_graph_T = torch.load("save_u_graph_T2.pth", weights_only=False)
    return u_graph_T


def main():
    from pde.utils import setup_logging

    setup_logging(debug=False)

    cfg = Config()
    time_cfg= ConfigTime()
    c_print(f'{time_cfg.dt = }', color="bright_magenta")

    #u_g_T = load_graph(cfg)
    u_g_T = mesh_graph(cfg)

    # time_pde = TimePDEBase(u_g_T, time_cfg, cfg)
    # time_pde.solve()

    # saved_graphs = time_pde.u_saves
    # for t, graph in saved_graphs.items():
    #     us, Xs = graph.us, graph.Xs
    #     plot_interp_graph(Xs, us[:, 0], title=f"t={t :.4g}")


if __name__ == "__main__":
    main()