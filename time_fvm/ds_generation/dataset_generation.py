import torch
import numpy as np
import os
import pickle
from cprint import c_print

from base_cfg import BASE_DIR
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


def main(save_name=f'01-14_21-08-43'):
    """
    Plot out the saved mesh and time step data.
    """
    save_dir = f'{BASE_DIR}/artefacts/fvm_saves/{save_name}'
    c_print(f"\nLoading from '{save_dir}'...", color="green")

    # Load mesh properties
    mesh_props_path = os.path.join(save_dir, 'mesh_props.npz')
    mesh_props = np.load(mesh_props_path)
    mesh_props = dict(mesh_props)

    bc_edge_tags = mesh_props.pop('bc_type_str')

    # Find and load time-step files
    time_files = sorted([f for f in os.listdir(save_dir) if f.startswith('t_') and f.endswith('.npz')])
    c_print(f"Found {len(time_files)} time-step file(s)", color="green")
    print()
    # Average graphs over time
    averaged_graphs = AveragedGraphs()
    time_files.sort(key=lambda x: float(x.split('_')[1].replace('.npz', '')))
    for save_i in time_files:
        file_path = os.path.join(save_dir, save_i)
        t, cell_primitives, bc_primitives = load_step(file_path)

        # print(f'{t = }')
        if t > 1.: # Skip initial transients
            averaged_graphs.add_step(cell_primitives, bc_primitives)
    mean_cells, mean_bc = averaged_graphs.get_average()
    mean_cells_norm = (mean_cells - mean_cells.mean(dim=0, keepdim=True)) / (mean_cells.std(dim=0, keepdim=True) + 1e-12)

    # Create new adaptively remeshed graph
    bc_edges = mesh_props['edges'][mesh_props['bc_edge_masK']]
    new_points, new_triangles, Us_new, bc_p_tags, bc_edges = adaptive_remesh(
        points=mesh_props['vertices'],
        triangles=mesh_props['triangles'],
        u_cells=mean_cells_norm,   # Use x-velocity for adaptivity
        bc_tags=bc_edge_tags,
        bc_edges=bc_edges,
        p_power=1.0, floor=0.5, g_quant=0.95,
        r_min=0.015, r_max=0.0275,
        boundary_keep_ratio=0.66,
    )

    # Get point tags for every point
    p_tags = []
    for t in bc_p_tags:
        if t is None:
            p_tags.append('Normal')
        else:
            p_tags.append(str(t))

    # Save new mesh
    save_dict = {
        'Xs': new_points,
        'triangles': new_triangles,
        'bc_edges': bc_edges,
        'p_tags': p_tags,
        'Us': Us_new,
    }
    save_path = f'{BASE_DIR}/artefacts/fvm2pde_dataset/{save_name}.pkl'
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(save_dict, f)
    c_print(f'Saved new remeshed graph to {save_path}', color="green")

    new_points, new_triangles = torch.from_numpy(new_points).float(), torch.from_numpy(new_triangles)
    Us_new = torch.from_numpy(Us_new).float()
    c_print(f'N points: {new_points.shape[0]}, N bcs: {bc_edges.shape[0]}', color="green")
    plot_interp_vertex(new_points, Us_new.T[:3], new_triangles, title='Adaptively remeshed')#, edgecolors="k")


if __name__ == '__main__':
    main()

