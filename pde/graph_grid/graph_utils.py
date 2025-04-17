import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from cprint import c_print
import matplotlib.tri as tri

from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse import csr_matrix
from scipy.interpolate import griddata
import numpy as np

def show_graph(edge_idx, Xs, us):
    """
    Plots a graph where the color of each node represents its value from the specified attribute.
    """
    edge_idx = edge_idx.cpu().clone().numpy().T
    Xs = Xs.cpu().clone().numpy()
    us = us.cpu().clone().numpy()

    # Remove self loops
    edge_idx = edge_idx[edge_idx[:, 0] != edge_idx[:, 1]]

    # Create a graph
    G = nx.Graph()
    G.add_edges_from(edge_idx)

    # Convert positions array to a dictionary for networkx
    pos_dict = {i: Xs[i] for i in range(Xs.shape[0])}

    # Draw the graph
    plt.figure(figsize=(8, 6))

    # Draw nodes with color reflecting their values
    nx.draw_networkx_nodes(
        G,
        pos=pos_dict,
        node_size=300,
        nodelist=sorted(G.nodes()),
        node_color=[us[node] for node in sorted(G.nodes())],
        cmap=plt.cm.viridis,
    )

    # Draw edges
    nx.draw_networkx_edges(
        G,
        pos=pos_dict,
        arrowstyle='-|>',  # Arrow style
        arrowsize=10,  # Arrow size
        edge_color='grey',
        width=1,
        connectionstyle='arc3,rad=0.1',  # Adds curvature to the edges
        arrows=True  # Display arrows on the edges
    )

    # Create labels for nodes with their values
    labels = {i: f'{i}' for i in range(len(us))}
    nx.draw_networkx_labels(G, pos=pos_dict, labels=labels)

    # Display the graph
    #plt.title("Graph Visualization with NumPy Inputs")
    # plt.axis('off')
    plt.tight_layout()
    plt.show()


def gen_perim(width, height, spacing):
    """
    Generates a tensor of 2D points representing the perimeter of a rectangle.
    Parameters:
    - width: Width of the rectangle.
    - height: Height of the rectangle.
    - spacing: Spacing between consecutive points.
    Returns:
    - A tensor of shape (N, 2), where N is the number of points, and each row is a 2D point (x, y).
    """
    # Create points for the bottom and top sides of the rectangle
    bottom = torch.stack([torch.arange(0, width, spacing), torch.zeros_like(torch.arange(0, width, spacing))], dim=-1)
    top = torch.stack([torch.arange(0, width, spacing), torch.full_like(torch.arange(0, width, spacing), height)], dim=-1)

    # Create points for the left and right sides of the rectangle
    left = torch.stack([torch.zeros_like(torch.arange(0, height, spacing)), torch.arange(0, height, spacing)], dim=-1)
    right = torch.stack([torch.full_like(torch.arange(0, height+1e-7, spacing), width), torch.arange(0, height+1e-7, spacing)], dim=-1)

    # Remove duplicated corner points
    left = left[1:] if len(left) > 1 else left  # Remove the bottom-left corner
    #right = right[1:] if len(right) > 1 else right  # Remove the top-right corner

    # Concatenate all points
    points = torch.cat([bottom, right, top.flip(0), left.flip(0)], dim=0)

    return points


def diag_permute(A_sparse: torch.Tensor):
    """ Reorder the rows and columns of a matrix to be as diagonal as possible using the reverse Cuthill-McKee algorithm.
        Args:
        - A (torch.Tensor): Input matrix.
        Returns:
        - A_permuted (torch.Tensor): Permuted matrix.
    """
    A_sparse = A_sparse.detach().cpu()
    A_sparse = A_sparse.coalesce()

    A = csr_matrix((A_sparse.values().numpy(), A_sparse.indices().numpy()), shape=A_sparse.size())

    perm = reverse_cuthill_mckee(A)

    return perm


def plot_sparsity(A, title=""):
    # Example: Create a sample sparse matrix in PyTorch

    matrix = A.to_dense().detach().cpu()
    binary_matrix = (matrix != 0).int().numpy()  # Convert to binary and then to numpy array

    # Create the plot using imshow
    plt.figure(figsize=(6, 6))
    plt.imshow(binary_matrix, cmap='Greys', interpolation='nearest')  # 'Greys' colormap for binary data

    # Set x and y ticks to show integer labels
    num_rows, num_cols = binary_matrix.shape
    plt.xticks(np.arange(0, num_cols, step=1))  # Show column indices as integers
    plt.yticks(np.arange(0, num_rows, step=1))  # Show row indices as integers

    # Set labels and title
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    plt.title(title)

    # Display grid lines for clarity
    plt.grid(True, which='both', color='lightgrey', linewidth=0.5)

    # Show the plot
    plt.show()


def test_grid(xmin: float, xmax: float, N: Tensor, device='cpu'):
    # Establish range of y values from N and dx
    xmin, xmax = torch.tensor(xmin), torch.tensor(xmax)
    Lx = xmax - xmin
    dx = Lx / (N[0] - 1)
    Ly = dx * (N[1] - 1)

    Xmin = torch.stack([xmin, xmin]).to(device)
    Xmax = torch.stack([xmax, xmin + Ly]).to(device)

    x_values = torch.linspace(Xmin[0], Xmax[0], N[0], device=device)
    y_values = torch.linspace(Xmin[1], Xmax[1], N[1], device=device)

    # Create the grid of points
    n_grid, m_grid = torch.meshgrid(y_values, x_values, indexing='xy')

    # Combine the grids to form the final N x M grid of points
    Xs = torch.stack([m_grid, n_grid], dim=-1).view(-1, 2)  # shape = [N, 2]
    Xs = Xs + (torch.rand_like(Xs)-0.5) * dx / 1.3


    c_print(f'Grid Range: {Xmin.tolist()} to {Xmax.tolist()}.', 'green')
    c_print(f'Grid Spacing: {dx:.5g}, grid Shape: {N.tolist()}.', 'green')

    return Xs


def plot_interp_graph(points, values, resolution=1000, title='Nearest Neighbor Interpolation', lim=None):
    # Create a grid over the coordinate space
    # Convert points and values to numpy arrays if they aren't already
    points, values = points.detach().cpu().numpy(), values.detach().squeeze().cpu().numpy()
    if lim is not None:
        values = np.clip(values, lim[0], lim[1])
    # Automatically determine the range for x and y
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()

    # Create a grid based on the ranges of x and y with the specified resolution
    grid_x, grid_y = np.mgrid[x_min:x_max:complex(resolution), y_min:y_max:complex(resolution)]

    # Interpolate the scalar values using nearest neighbors
    grid_z = griddata(points, values, (grid_x, grid_y), method='nearest')

    # Plot the interpolated data
    plt.figure(figsize=(16, 16))
    if lim is not None:
        plt.imshow(grid_z.T, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap='viridis', vmin=lim[0], vmax=lim[1])
    else:
        plt.imshow(grid_z.T, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap='viridis')
    plt.colorbar()

    # Scatter plot of the original points
    plt.scatter(points[:, 0], points[:, 1], marker='.', s=40)

    # Title and axis labels
    plt.title(title, fontsize=30)
    plt.xlabel('X')
    plt.ylabel('Y')

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_points(Xs, values, lims=None, title="", show_index=False, Xlims=None):
    Xs = Xs.cpu()
    values = values.cpu()

    if len(values.shape) == 1:
        values = values.unsqueeze(0)
        fig, axes = plt.subplots(1, 1, figsize=(12, 9))
        axes = [axes]
    else:
        n_plots = values.shape[0]
        fig, axes = plt.subplots(n_plots, 1, figsize=(8, n_plots*4))

    # Loop over each batch
    if Xlims is None:
        Xlims = (Xs[:, 0].min(), Xs[:, 0].max()), (Xs[:, 1].min(), Xs[:, 1].max())

    for i, ax in enumerate(axes):
        ax.set_title(f"{title} - Batch {i}")
        if lims is None:
            sc = ax.scatter(Xs[:, 0], Xs[:, 1], c=values[i], cmap='viridis')
        else:
            sc = ax.scatter(Xs[:, 0], Xs[:, 1], c=values[i], cmap='viridis',
                            vmin=lims[0], vmax=lims[1])

        if show_index:
            for i, X in enumerate(Xs):
                x, y = X
                if (Xlims[0][0] <= x <= Xlims[0][1]) and (Xlims[1][0] <= y <= Xlims[1][1]):
                    ax.text(x, y, f"{i}", fontsize=8)
        fig.colorbar(sc, ax=ax)
        ax.set_aspect('equal', adjustable='box')

        ax.set_xlim(Xlims[0])
        ax.set_ylim(Xlims[1])

    #plt.tight_layout()
    plt.show()


def plot_interp(Xs, values, triangles, Xlims=None, title="", resolution=1000):
    """
    Xs: Tensor of vertex coordinates (N x 2)
    values: Tensor of face-based values.
            If values is 1D, it's assumed to be defined on the triangulation faces.
            If 2D, each row is treated as a separate batch.
    lims: Optional tuple ((xmin, xmax), (ymin, ymax)) to set the plot limits.
    title: Plot title.
    resolution: (Unused here; kept for interface consistency)
    """
    # Convert to numpy arrays.
    Xs = Xs.cpu().numpy()
    values = values.cpu().numpy()
    triangles = triangles.cpu().numpy()

    # If values is 1D, expand to a batch of one.
    if len(values.shape) == 1:
        values = values[None, :]
        fig, axes = plt.subplots(1, 1, figsize=(12, 9))
        axes = [axes]
    else:
        n_plots = values.shape[0]
        fig, axes = plt.subplots(n_plots, 1, figsize=(8, n_plots * 4))

    # Create a triangulation from the vertex locations.
    triang = tri.Triangulation(Xs[:, 0], Xs[:, 1], triangles)

    if isinstance(title, str):
        title = [title] * len(axes)

    # Determine plot limits.
    if Xlims is not None:
        xlim, ylim = Xlims
    else:
        xlim = (Xs[:, 0].min(), Xs[:, 0].max())
        ylim = (Xs[:, 1].min(), Xs[:, 1].max())

    in_region = []
    for tri_indices in triang.triangles:
        verts = np.column_stack((Xs[tri_indices, 0], Xs[tri_indices, 1]))
        if np.all((verts[:, 0] >= xlim[0]) & (verts[:, 0] <= xlim[1]) &
                  (verts[:, 1] >= ylim[0]) & (verts[:, 1] <= ylim[1])):
            in_region.append(True)
        else:
            in_region.append(False)
    in_region = np.array(in_region)

    # Create a new triangulation using only the triangles inside the region.
    new_triangles = triang.triangles[in_region]
    new_triang = tri.Triangulation(Xs[:, 0], Xs[:, 1], triangles=new_triangles)

    # Loop over each batch and plot only the triangles inside the region.
    for i, ax in enumerate(axes):
        ax.set_title(f"{title[i]}")

        # Filter the face-based values for the triangles inside the region.
        new_facecolors = values[i][in_region]

        # Plot using the new triangulation and corresponding facecolors.
        tc = ax.tripcolor(new_triang, facecolors=new_facecolors, edgecolors='none',
                          cmap='viridis', shading='flat')
        fig.colorbar(tc, ax=ax)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()


def plot_edges(coords, edge_idx, colors=None, title="", show_index=False, lims=None, Xlims=None):
    """ Plot the edges of the mesh.
        coords.shape = (n, 2)
        edge_idx.shape = (m, 2)
        If 'color' is provided, it should be a torch tensor. In the case that
        'color' is 1D, it is assumed to be (m,) and converted to (m,1) for plotting.
    """
    # Convert inputs from torch tensors to numpy arrays.
    coords = coords.cpu().detach().numpy()
    edge_idx = edge_idx.cpu().detach().numpy()

    if colors is None:
        colors = torch.zeros(len(edge_idx))
        
    # If color is a 1D tensor, unsqueeze to (m, 1) and create a single subplot.
    if len(colors.shape) == 1:
        colors = colors.unsqueeze(-1)
        fig, axes = plt.subplots(1, 1, figsize=(16, 12))
        axes = [axes]
    else:
        n_plots = colors.shape[1]
        fig, axes = plt.subplots(n_plots, 1, figsize=(8, n_plots * 4))

    colormap = plt.get_cmap("viridis")
    edge_colors = colors.cpu().detach().numpy().T  # shape = (n_plots, m)

    # Extract the coordinates of each edge. Each row in 'points' is an edge defined
    # by its two endpoints (shape: (m, 2, 2)).
    points = coords[edge_idx]

    # Plot each batch (subplot).
    for i, ax in enumerate(axes):
        ax.set_aspect('equal', adjustable='box')

        # Plot each edge using the corresponding color.
        if Xlims is None:
            xmin, xmax = coords[:, 0].min(), coords[:, 0].max()
            ymin, ymax = coords[:, 1].min(), coords[:, 1].max()
        else:
            (xmin, xmax), (ymin, ymax) = Xlims

        edge_nums, edge_scalars = [], []
        for j, (edge, s) in enumerate(zip(points, edge_colors[i], strict=True)):
            midpoint = edge.mean(axis=0)
            if not ((xmin <= midpoint[0] <= xmax) and (ymin <= midpoint[1] <= ymax)):
                continue
            edge_scalars.append(s)
            edge_nums.append(j)

        edge_scalars = np.array(edge_scalars)
        min_c, max_c = edge_scalars.min(), edge_scalars.max()
        edge_scalars = (edge_scalars - min_c) / (max_c - min_c + 1e-9)

        for j, s in zip(edge_nums, edge_scalars, strict=True):
            c = colormap(s)
            edge = points[j]
            ax.plot(edge[:, 0], edge[:, 1], color=c)
            if show_index:
                midpoint = edge.mean(axis=0)
                ax.text(midpoint[0], midpoint[1], f"{j}", fontsize=8)
                ax.annotate(
                    '',  # No text
                    xy=(edge[1]),  # Arrow tip (end of the line)
                    xytext=midpoint,  # Arrow tail (start of the line)
                    arrowprops=dict(arrowstyle='->', lw=.5)
                )

            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])


        # If colors are provided, create a ScalarMappable for the colorbar.
        ax.set_title(f"{title} - Batch {i}, [min={min_c.item():.3g}, max={max_c.item():.3g}]")

        # Use the original scalar range for this batch.
        norm = plt.Normalize(vmin=min_c.item(), vmax=max_c.item())
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        # Optional: attach the actual scalar array (could also use an empty array)
        cbar = fig.colorbar(sm, ax=ax)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # Sample data: list of (x, y) coordinates and their scalar values
    points = np.array([
        (0, 0), (1, 0), (1, 1), (0, 1),
        (0.5, 0.5), (0.75, 0.75), (0.25, 0.25)
    ])
    values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

    # Plot the interpolation graph
    plot_interp_graph(points, values)

