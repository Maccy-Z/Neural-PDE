import numpy as np
from datetime import datetime
import os
import torch

from fvm_mesh import FVMMesh
from sparse_utils import plot_interp

class Saver:
    def __init__(self, mesh: FVMMesh, save_dir=None):
        if save_dir is None:
            timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
            self.save_dir = f'saves/{timestamp}'
        else:
            self.save_dir = save_dir

        os.makedirs(self.save_dir, exist_ok=True)

        # Mesh property: main
        centroids = mesh.centroids  # shape = [n_cells, comp=2]
        triangles = mesh.triangles  # shape = [n_cells, comp=3]
        vertices = mesh.vertices
        # Mesh property: BC edges
        bc_edge_mask = mesh.bc_edge_mask
        bc_midpoints = mesh.midpoints[bc_edge_mask]  # shape = [n_bc_edge, comp=2]
        bc_normals = mesh.normals[bc_edge_mask]  # shape = [n_bc_edge, comp=2]
        mesh_props = {"triangles": triangles.cpu().numpy(), "vertices": vertices.cpu().numpy(), "centroids": centroids.cpu().numpy(),
                      "bc_midpoints": bc_midpoints.numpy(), "bc_normals": bc_normals.numpy()}
        np.savez_compressed(f'{self.save_dir}/mesh_props.npz', **mesh_props)


    def save(self, t, E_props, primatives):
        bc_edge_mask = E_props.mesh.bc_edge_mask
        # Save centroid values
        primatives = primatives  # shape = [n_cells, comp=4]
        # Save boundary edge values
        Vs_bc = E_props.Vs_faces[bc_edge_mask].mean(dim=1)  # shape = [n_bc_edge, comp=2]
        rho_bc = E_props.rho_faces[bc_edge_mask].mean(dim=1)  # shape = [n_bc_edge, comp=1]
        T_bc = E_props.T_faces[bc_edge_mask].mean(dim=1)  # shape = [n_bc_edge, comp=1]
        bc_vals = torch.cat([Vs_bc, rho_bc, T_bc], dim=1)  # shape = [n_bc_edge, comp=4]

        # Save to file.
        # Normalise the primatives and bc_vals and save as fp16
        prim_mean, prim_std = primatives.mean(dim=0, keepdim=True), primatives.std(dim=0, keepdim=True)
        prim_scale = (primatives - prim_mean) / prim_std
        bc_scale = (bc_vals - prim_mean) / prim_std
        values = {"t": t.cpu().item(),
                    "prim_mean": prim_mean.cpu().numpy(), "prim_std": prim_std.cpu().numpy(),
                    "cell_primatives": prim_scale.cpu().half().numpy(), "bc_primatives": bc_scale.cpu().half().numpy()}

        t = t.cpu().item()
        t_str = f'{t:.4g}'  # Ensure step number is zero-padded to 5 digits
        np.savez_compressed(f'{self.save_dir}/t_{t_str}.npz', **values)


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


def main(save_dir='/home/maccyz/Documents/Neural_PDE/time_fvm/saves/12-11_19-10-46'):
    """
    Main function to demonstrate loading a save folder.
    It will first create a dummy save folder if it doesn't exist.
    """
    print(f"\nLoading from '{save_dir}'...")

    # Load mesh properties
    mesh_props_path = os.path.join(save_dir, 'mesh_props.npz')
    mesh_props = np.load(mesh_props_path)
    print("Mesh properties keys:", list(mesh_props.keys()))
    mesh_props = dict(mesh_props)
    mesh_props = {k: torch.from_numpy(v) for k, v in mesh_props.items()}

    # print("Centroids shape:", mesh_props['centroids'].shape)

    # Find and load time-step files
    time_files = sorted([f for f in os.listdir(save_dir) if f.startswith('t_') and f.endswith('.npz')])
    if not time_files:
        print("No time-step files found.")
        return

    print(f"Found {len(time_files)} time-step file(s). Loading the first one: {time_files[0]}")

    # Load and process the first time step file
    # Sort time files by time value
    time_files.sort(key=lambda x: float(x.split('_')[1].replace('.npz', '')))
    for save_i in time_files:
        t = save_i.split('_')[1].replace('.npz', '')
        file_path = os.path.join(save_dir, save_i)
        t, cell_primatives, bc_primatives = load_step(file_path)

        print(f"Time: {save_i}")
        plot_interp(mesh_props['vertices'], cell_primatives.T[:2], mesh_props['triangles'], title=f't={t:.3g}')

if __name__ == '__main__':
    main()
