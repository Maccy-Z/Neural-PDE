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
    time_files.sort(key=lambda x: float(x.split('_')[1].replace('.npz', '')))
    for save_i in time_files:
        file_path = os.path.join(save_dir, save_i)
        t, cell_primitives, bc_primitives = load_step(file_path)

        vertices = mesh_props['vertices']
        triangles = mesh_props['triangles']
        vertices = torch.from_numpy(vertices).float()
        triangles = torch.from_numpy(triangles).long()

        print(f'{vertices.shape = }, {triangles.shape = }, {cell_primitives.shape = }')
        plot_interp_cell(vertices, cell_primitives.T, triangles, title='Adaptively remeshed')  # , edgecolors="k")


if __name__ == '__main__':
    main()
