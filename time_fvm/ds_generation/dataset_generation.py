import torch
import numpy as np
import os

from time_fvm.ds_generation.saving import plot_interp


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


def main(save_dir='/home/maccyz/Documents/Neural_PDE/time_fvm/saves/12-11_21-18-54'):
    """
    Plot out the saved mesh and time step data.
    """
    print(f"\nLoading from '{save_dir}'...")

    # Load mesh properties
    mesh_props_path = os.path.join(save_dir, 'mesh_props.npz')
    mesh_props = np.load(mesh_props_path)
    mesh_props = dict(mesh_props)

    print(f'{mesh_props.keys() = }')
    bc_tags = mesh_props.pop('bc_str_tags')
    mesh_props = {k: torch.from_numpy(v) for k, v in mesh_props.items()}

    # Find and load time-step files
    time_files = sorted([f for f in os.listdir(save_dir) if f.startswith('t_') and f.endswith('.npz')])
    if not time_files:
        print("No time-step files found.")
        return

    print(f"Found {len(time_files)} time-step file(s)")

    # Load and process the first time step file
    # Sort time files by time value
    time_files.sort(key=lambda x: float(x.split('_')[1].replace('.npz', '')))
    for save_i in time_files:
        t = save_i.split('_')[1].replace('.npz', '')
        file_path = os.path.join(save_dir, save_i)
        t, cell_primatives, bc_primatives = load_step(file_path)

        print(f"Time: {t:.4g}")
        plot_interp(mesh_props['vertices'], cell_primatives.T[:2], mesh_props['triangles'], title=f't={t:.3g}')


if __name__ == '__main__':
    main()

