import torch
from scipy.sparse.csgraph import dijkstra
from scipy import sparse as sp
import numpy as np
from rbf.pde import fd

from pde.utils_sparse import plot_sparsity, csr_torch_to_scipy, csr_scipy_to_torch


def gen_multi_idx_tuple(m):
    """
    Indicies in tuple form for dict indexing
    """
    indices = []
    for total_degree in range(m + 1):
        for alpha_x in range(total_degree + 1):
            alpha_y = total_degree - alpha_x
            indices.append((alpha_x, alpha_y))

    indices = sorted(indices, key=lambda x: (x[0] + x[1], -x[0]))
    return indices

def nearest_neighbors(tris, Xs, n_neigh):
    n_points = Xs.shape[0]

    # 1) collect all undirected edges from the triangles
    #    each triangle (i,j,k) → edges (i,j), (j,k), (k,i)
    e01 = tris[:, [0, 1]]
    e12 = tris[:, [1, 2]]
    e20 = tris[:, [2, 0]]
    edges = torch.cat([e01, e12, e20], dim=0)

    # 2) make undirected: add the reversed edges
    rev = edges[:, [1, 0]]
    edges = torch.cat([edges, rev], dim=0)

    # 3) remove duplicates
    edges = torch.unique(edges, dim=0)  # shape [E, 2]

    row, col = edges.t()  # each is shape [E]

    # 4) compute Euclidean distances for each edge
    diff = Xs[row] - Xs[col]  # [E, 2]
    dists = diff.norm(dim=1)  # [E]

    # 5a) build a sparse adjacency (recommended if graph is large & sparse)
    adj_sparse = torch.sparse_coo_tensor(
        indices=torch.stack([row, col], dim=0),  # [2, E]
        values=dists,  # [E]
        size=(n_points, n_points)
    ).coalesce().to_sparse_csr()

    adj_sparse = csr_torch_to_scipy(adj_sparse)

    D = dijkstra(adj_sparse, directed=False, return_predecessors=False)
    idx = np.argpartition(D, n_neigh, axis=1)[:, :n_neigh]
    return idx


def calc_coeff(Xs: torch.Tensor, stencils: np.array, n_neigh: int, diff_orders):
    n_points = Xs.shape[0]

    Xs = Xs.numpy()
    # diff_order = [[2, 0], [0, 2]]
    diff_orders = np.array(diff_orders)
    data = fd.weights(
        Xs, Xs[stencils],
        diffs=diff_orders,
        phi="phs3",
        order=1,
        eps=0.01,
        sum_terms=False
    )
    rows = np.repeat(range(n_points), n_neigh)
    cols = stencils.ravel()
    data = data.ravel()
    w = sp.coo_matrix((data, (rows, cols)), (n_points, n_points))

    # w = fd.weight_matrix(Xs, Xs, n=49, diffs=np.array(diff_order), order=1, phi="phs3", eps=0.01)
    w = csr_scipy_to_torch(w)

    indices = w.indices().to(torch.float32)[[1, 0], :]
    weights = w.values().to(torch.float32)

    return indices, weights

def main():
    from matplotlib import pyplot as plt

    # 2D grid
    plot_num = 0
    Xs_all = torch.load("../Xs_all.pt", weights_only=True)
    values = 1 - 0.5*Xs_all[:, 0]

    w = fd.weight_matrix(Xs_all, Xs_all, n=8, diffs=np.array((1, 0)), order=1, phi="phs3", eps=0.1)
    w = csr_scipy_to_torch(w)

    print(f'{w[plot_num] = }')
    indices = w.coalesce().indices()
    weights = w.coalesce().values()

    idx_mask = indices[0] == plot_num
    indices = indices[:, idx_mask]
    weights = weights[idx_mask]

    plt.scatter(Xs_all[:, 0], Xs_all[:, 1], s=10, c='blue')
    plt.scatter(Xs_all[plot_num, 0], Xs_all[plot_num, 1], s=100, c='red', label='Center Point')
    plt.scatter(Xs_all[indices, 0], Xs_all[indices, 1], s=50, c='green', label='Neighbors')
    plt.show()

    # plt.scatter(Xs_all[:, 0], Xs_all[:, 1], s=10, c=values)
    # plt.show()

    derivative = w @ values

    # plt.scatter(Xs_all[:, 0], Xs_all[:, 1], s=10, c=derivative)
    # plt.show()

    # print(derivative)


    indices, weights = torch.load("../coeffs.pt")
    w_old = torch.sparse_coo_tensor(indices, weights, w.shape).T.coalesce()

    print(f'{w_old[plot_num] = }')
    derivative_old = w_old @ values

    idx_mask = indices[0] == plot_num
    indices = indices[:, idx_mask]
    weights = weights[idx_mask]

    plt.scatter(Xs_all[:, 0], Xs_all[:, 1], s=10, c='blue')
    plt.scatter(Xs_all[plot_num, 0], Xs_all[plot_num, 1], s=100, c='red', label='Center Point')
    plt.scatter(Xs_all[indices, 0], Xs_all[indices, 1], s=50, c='green', label='Neighbors')
    plt.show()
    #
    # for p, w in zip(points, w[0]):
    #     print(f"Point: {p[0].item():.3g}, Weight: {w:.3g}")


# Example usage
if __name__ == "__main__":
    main()