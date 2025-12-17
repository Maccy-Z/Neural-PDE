import torch
import numpy as np
import os

from time_fvm.ds_generation.downsampling import adaptive_remesh
from time_fvm.sparse_utils import plot_interp_cell, plot_interp_vertex


def load_step(file_path):
    """
    Loads a single time step file and returns the un-normalized primative values.
    """
    data = np.load(file_path)
    prim_mean = data['prim_mean']
    prim_std = data['prim_std']
    cell_primatives_scaled = data['cell_primatives'].astype(np.float32)
    bc_primatives_scaled = data['bc_primatives'].astype(np.float32)

    cell_primatives = cell_primatives_scaled * prim_std + prim_mean
    bc_primatives = bc_primatives_scaled * prim_std + prim_mean

    return data['t'], torch.from_numpy(cell_primatives).float(), torch.from_numpy(bc_primatives).float()


class AveragedGraphs:
    sum_cells: torch.Tensor
    sum_bc: torch.Tensor
    def __init__(self):
        self.count = 0

        self.sum_cells = 0
        self.sum_bc = 0


    def add_step(self, cell_values: torch.Tensor, bc_values: torch.Tensor):
        """ Add a new graph to the average.
            shape = [N_cells, N_comp=4]
        """
        self.sum_cells += cell_values.double()
        self.sum_bc += bc_values.double()
        self.count += 1

    def get_average(self):
        mean_cells = self.sum_cells / self.count
        mean_bc = self.sum_bc / self.count
        return mean_cells.float(), mean_bc.float()


def main(save_dir='/home/maccyz/Documents/Neural_PDE/time_fvm/artefacts/saves/12-17_19-47-25'):
    """
    Plot out the saved mesh and time step data.
    """
    print(f"\nLoading from '{save_dir}'...")

    # Load mesh properties
    mesh_props_path = os.path.join(save_dir, 'mesh_props.npz')
    mesh_props = np.load(mesh_props_path)
    mesh_props = dict(mesh_props)

    bc_edge_tags = mesh_props.pop('bc_type_str')
    print(f'{mesh_props.keys() = }')

    # Find and load time-step files
    time_files = sorted([f for f in os.listdir(save_dir) if f.startswith('t_') and f.endswith('.npz')])
    print(f"Found {len(time_files)} time-step file(s)")
    print()
    # Average graphs over time
    averaged_graphs = AveragedGraphs()
    time_files.sort(key=lambda x: float(x.split('_')[1].replace('.npz', '')))
    for save_i in time_files:
        file_path = os.path.join(save_dir, save_i)
        t, cell_primitives, bc_primitives = load_step(file_path)

        # print(f'{t = }')
        if t > 1: # Skip initial transients
            averaged_graphs.add_step(cell_primitives, bc_primitives)
    mean_cells, mean_bc = averaged_graphs.get_average()
    mean_cells_norm = (mean_cells - mean_cells.mean(dim=0, keepdim=True)) / (mean_cells.std(dim=0, keepdim=True) + 1e-12)

    # Create new adaptively remeshed graph
    bc_edges = mesh_props['edges'][mesh_props['bc_edge_masK']]
    new_points, new_triangles, u_nodes_new, u_cells_new, bc_vertices, bc_vertex_tags = adaptive_remesh(
        points=mesh_props['vertices'],
        triangles=mesh_props['triangles'],
        u_cells=mean_cells_norm[:, :2],   # Use x-velocity for adaptivity
        bc_tags=bc_edge_tags,
        bc_edges=bc_edges,
        n_vertices_new=mesh_props['vertices'].shape[0] // 5,  # Reduce to 1/4 vertices0
        p_power=1.0, floor=0.2, g_quant=0.95,
        r0=0.025,
        boundary_keep_ratio=0.3,
    )

    # plot_interp(mesh_props['vertices'], mean_cells.T[:2], mesh_props['triangles'], title=f'Average over {averaged_graphs.count} steps')

    new_points, u_cells_new, new_triangles = torch.from_numpy(new_points).float(), torch.from_numpy(u_cells_new).float(), torch.from_numpy(new_triangles)
    u_nodes_new = torch.from_numpy(u_nodes_new).float()

    print(f'{new_points.shape = }, {u_nodes_new.shape = }, {new_triangles.shape = }')
    plot_interp_vertex(new_points, u_nodes_new.T[:2], new_triangles, title='Adaptively remeshed x-velocity', edgecolors="k")

    from matplotlib import pyplot as plt # For debugging
    bc_vertex_tags = [str(t) for t in bc_vertex_tags]
    bc_points = new_points[bc_vertices]
    cmap = {"Left": "red", "Right": "blue", "NavierWall": "green"}
    print(f'{bc_points.shape = }')
    for p, t in zip(bc_points, bc_vertex_tags):
        c = cmap[t]
        plt.scatter(p[:1], p[1:], c=c)
    plt.show()

if __name__ == '__main__':
    main()

