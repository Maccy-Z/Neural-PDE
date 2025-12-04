import torch
from scipy.sparse.csgraph import dijkstra
from scipy import sparse as sp
import numpy as np
import pickle
import hashlib
import os
from functools import wraps
from rbf.pde import fd
import logging

from pde.utils_sparse import plot_sparsity, csr_torch_to_scipy, csr_scipy_to_torch
from pde.utils import ARTEFACT_DIR

CACHE_DIR = ARTEFACT_DIR / "cache_calc_coeff"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_key(func_name, *args, **kwargs):
    """Generates a cache key based on function name and arguments."""
    # Serialize arguments to a byte string
    # For complex types like torch.Tensor and np.array,
    # we need a consistent way to represent them for hashing.
    # Converting to bytes via pickle is one way, but ensure tensors are on CPU.

    serialized_args = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            # Move to CPU and convert to numpy for more stable hashing/pickling
            # Adding .tobytes() for numpy array ensures a more stable hash
            serialized_args.append(pickle.dumps(arg.cpu().numpy().tobytes()))
        elif isinstance(arg, np.ndarray):
            serialized_args.append(pickle.dumps(arg.tobytes()))
        else:
            serialized_args.append(pickle.dumps(arg)) # For basic types like int, tuple

    for key, value in sorted(kwargs.items()): # Sort kwargs for consistency
        if isinstance(value, torch.Tensor):
            serialized_args.append(pickle.dumps((key, value.cpu().numpy().tobytes())))
        elif isinstance(value, np.ndarray):
            serialized_args.append(pickle.dumps((key, value.tobytes())))
        else:
            serialized_args.append(pickle.dumps((key, value)))

    # Create a hash of the serialized arguments
    hasher = hashlib.md5() # Or sha256 for lower collision probability
    hasher.update(func_name.encode())
    for arg_bytes in serialized_args:
        hasher.update(arg_bytes)

    return hasher.hexdigest()

def disk_cache(func):
    """Decorator to cache function results to disk."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Handle keyword arguments for calc_coeff if they are explicitly passed
        # For simplicity, this example assumes positional arguments match the signature
        # or that you'll adjust key generation accordingly.

        # A more robust way to map args/kwargs to the function signature
        # for key generation might be needed for very generic decorators.
        # For this specific function, we can be explicit.

        # Create a representation of arguments for caching
        # Ensure tensors and arrays are handled correctly
        key_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                # Detach and move to CPU to ensure hashability and prevent autograd issues in cache key
                key_args.append(arg.detach().cpu())
            else:
                key_args.append(arg)

        key_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                key_kwargs[k] = v.detach().cpu()
            else:
                key_kwargs[k] = v

        cache_key = get_cache_key(func.__name__, *key_args, **key_kwargs)
        cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")

        if os.path.exists(cache_file):
            logging.debug(f"Loading from cache: {func.__name__} (key: {cache_key})")
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e: # Handle potential unpickling errors
                logging.warning(f"Error loading graph from cache: {e}. Recalculating.")
                os.remove(cache_file) # Remove corrupted cache file

        logging.debug(f"Calculating and caching: {func.__name__} (key: {cache_key})")
        result = func(*args, **kwargs)
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e: # Handle potential pickling errors
            logging.warning(f"Error saving graph to cache: {e}")
            if os.path.exists(cache_file):
                os.remove(cache_file) # Clean up if saving failed

        return result
    return wrapper

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

@disk_cache
def calc_coeff(Xs: torch.Tensor, stencils: np.ndarray, n_neigh: int, diff_orders: tuple[int, int]):

    n_points = Xs.shape[0]

    Xs = Xs.numpy()
    diff_orders = np.array(diff_orders)
    data = fd.weights(
        Xs, Xs[stencils],
        diffs=diff_orders,
        phi="phs5",
        order=2,
        eps=0.1,
        sum_terms=False
    )
    rows = np.repeat(range(n_points), n_neigh)
    cols = stencils.ravel()
    data = data.ravel()
    w = sp.coo_matrix((data, (rows, cols)), (n_points, n_points))

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