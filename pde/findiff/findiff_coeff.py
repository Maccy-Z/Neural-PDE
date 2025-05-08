import torch
from typing import Literal
from functools import lru_cache, wraps
from scipy.spatial import KDTree
from cprint import c_print
from torch.multiprocessing import Pool
import numpy as np

from pde.findiff.min_norm import min_sq_norm, min_abs_norm
from pde.graph_grid.graph_store import Point, P_Types

diff_options = Literal["pinv", "sq_weight_norm", "abs_weight_norm"]


class ConvergenceError(Exception):
    pass


def lru_cache_tensor(maxsize=128):
    """
    Decorator to apply LRU caching to functions that take a tensor and another parameter.
    """

    def tensor_to_key(tensor: torch.Tensor):
        """
        Converts a tensor to a hashable key by flattening its data and combining with its shape.
        """
        # Ensure tensor is on CPU and detached from any computation graph
        tensor_cpu = tensor.detach().cpu()
        # Convert tensor data to a tuple of floats (or ints)
        tensor_data = tuple(tensor_cpu.numpy().flatten())
        # Get tensor shape
        tensor_shape = tensor_cpu.size()
        return (tensor_data, tensor_shape)

    def decorator(func):
        @lru_cache(maxsize=maxsize)
        def cached_func(tensor_key, param):
            # Reconstruct the tensor from the key
            tensor_data, tensor_shape = tensor_key
            tensor = torch.tensor(tensor_data).reshape(tensor_shape)
            return func(tensor, param)

        @wraps(func)
        def wrapper(tensor, param):
            # Convert tensor to a hashable key
            tensor_key = tensor_to_key(tensor)
            return cached_func(tensor_key, param)

        return wrapper

    return decorator


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


@lru_cache(maxsize=5)
def generate_multi_indices(m):
    """
    Generate all multi-indices (alpha_x, alpha_y) for monomials up to degree m.
    m (int): The maximum total degree of the monomials.
    Returns: List shape (M, 2) containing all multi-indices, where M is the number of monomials.
      Each row represents a multi-index [alpha_x, alpha_y].
    """

    indices = []
    for total_degree in range(m + 1):
        for alpha_x in range(total_degree + 1):
            alpha_y = total_degree - alpha_x
            indices.append([alpha_x, alpha_y])
    return indices


@lru_cache_tensor(maxsize=5)
def compute_D_vector(alpha, k):
    """
    Compute the vector D containing the derivatives of the monomials evaluated at the center point. One-hot coefficient.
    alpha: List of all derivative indices
    k: Derivative order to compute
    """
    k = torch.tensor(k)  # Shape (2,)

    # Identify monomials where alpha == k
    condition = (alpha[:, 0] == k[0]) & (alpha[:, 1] == k[1])

    D = torch.zeros(alpha.size(0), dtype=torch.float32)

    idx = condition.nonzero(as_tuple=True)[0]

    assert idx.numel() == 1, f'Wanted derivaitive order must be less than M'
    # Compute D_j = k_x! * k_y!
    D_value = torch.exp(torch.lgamma(k[0] + 1) + torch.lgamma(k[1] + 1))
    D[idx] = D_value

    return D


def construct_A_matrix(delta_x, delta_y, multi_indices):
    """
    Construct the transpose of matrix A by evaluating the monomials at the coordinate differences.

    Parameters:
    - delta_x (torch.Tensor): Tensor of shape (N,) containing the x-coordinate differences of points.
    - delta_y (torch.Tensor): Tensor of shape (N,) containing the y-coordinate differences of points.
    - multi_indices (torch.Tensor): Tensor of shape (M, 2) containing the multi-indices of the monomials.

    Returns:
    - torch.Tensor: The transpose of matrix A, of shape (M, N), where each element A.T[j, i] =
                    (delta_x[i])^(alpha_x[j]) * (delta_y[i])^(alpha_y[j]).
    """
    # Extract alpha_x and alpha_y from multi-indices and reshape for broadcasting
    alpha_x = multi_indices[:, 0].unsqueeze(1).float()  # Shape (M, 1)
    alpha_y = multi_indices[:, 1].unsqueeze(1).float()  # Shape (M, 1)

    # Reshape delta_x and delta_y for broadcasting
    delta_x = delta_x.unsqueeze(0)  # Shape (1, N)
    delta_y = delta_y.unsqueeze(0)  # Shape (1, N)

    # Evaluate monomials using broadcasting to compute A transpose matrix
    A_T = (delta_x ** alpha_x) * (delta_y ** alpha_y)  # Shape (M, N)

    return A_T


def fin_diff_weights(center, points, derivative_order, m, method: diff_options, atol=1e-4, eps=6e-8):
    """
    Compute the finite difference weights for approximating the specified derivative at the center point.
    Parameters:
    - center (tensor): The coordinates (x0, y0) of the center point.
    - points (tensor): List of (x, y) coordinates of  points.
    - derivative_order (tuple): A tuple (k_x, k_y) specifying the orders of the derivative with respect to x and y.
    - m (int): The maximum total degree of the monomials (order of accuracy).
    - method (str): Method to compute the weights. Options: 'pinv', 'sq_weight_norm', 'abs_weight_norm'.
    - atol (float): Acceptable absolute solve max error.
    - eps (float): Floating point error scale.
    Returns:
    - torch.Tensor: A tensor of shape (N,) containing the weights w_i for the finite difference approximation.
    """
    # Step 1: Generate multi-indices for monomials up to degree m
    multi_indices = generate_multi_indices(m)  # Shape (M, 2)
    multi_indices = torch.tensor(multi_indices, dtype=torch.long)

    # Step 2: Compute the D vector containing derivatives of monomials at the center
    D = compute_D_vector(multi_indices, derivative_order)  # Shape (M,)

    # Step 3: Compute coordinate differences between  points and the center point
    deltas = points - center
    delta_x, delta_y = deltas[:, 0], deltas[:, 1]  # Shape (N,)

    # Step 4: Construct the matrix A by evaluating monomials at the coordinate differences
    A_T = construct_A_matrix(delta_x, delta_y, multi_indices)  # Shape (N, M)

    # Step 5: Solve the linear system A_T w = D.
    # Step 5.1: Weight magnitude of w, for underdetermined system / extra points. (Error formula)
    weights = torch.norm(deltas, p=2, dim=1) ** (m + 2)
    weights = weights + eps

    if method == "abs_weight_norm":
        try:
            w, status = min_abs_norm(A_T, D, weights)
        except ValueError as e:
            status = e.args[0]
            raise ConvergenceError(status, f'') from None
    elif method=="pinv":
        A_T_pinv = torch.linalg.pinv(A_T)  # Compute the pseudoinverse of A_T
        w = A_T_pinv @ D  # Compute the weights w (Shape: (N,))
        status = None
    elif method=="sq_weight_norm":
        w, status = min_sq_norm(A_T, weights, D)
    else:
        exit("Method not implemented")

    err = A_T @ w - D
    max_err = torch.abs(err).max()
    if max_err > atol:
        raise ConvergenceError(status, f'Error too large: {max_err.item() = :.3g}')


    return w, {'status': status, 'max_err': max_err,}
               #'mean_err': err.abs().mean(), 'abs_res': lambda: w.abs() @ weights, 'sq_res': lambda: w.unsqueeze(0) @ torch.diag(weights) @ w}


def _calc_coeff_single(j, X, param_dict):
    """ Inner loop for multiprocessing"""
    global global_Xs_all, global_adj_mat
    Xs_all, adj_mat = global_Xs_all, global_adj_mat


    diff_order = param_dict["order"]
    diff_acc = param_dict["acc"]
    N_us_tot = param_dict["N_us_tot"]
    min_points = param_dict["min_points"]
    max_points = param_dict["max_points"]

    # Find the nearest neighbors and calculate coefficients.
    # If the calculation fails (not enough points for given accuracy), increase the number of neighbors until it succeeds.
    for i in adj_mat.keys(): #range(min_points, max_points, 25):
        try:
            # _, neigh_idx = kdtree.query(X, k=i)
            # where_neigh = np.where(adj_idx[0] == j)[0]
            # neigh_idx = adj_idx[1, where_neigh]
            neigh_idx = adj_mat[i][j]
            neigh_Xs = Xs_all[neigh_idx]

            w, _ = fin_diff_weights(X, neigh_Xs, diff_order, diff_acc, "abs_weight_norm", atol=1e-3, eps=6e-8)
        except ConvergenceError:
            print(f"{j} Adding more points")
        else:
            break
    else:
        c_print(f"Using looser tolerance for point {j}, {X=}", color="bright_magenta")
        # Using Try again with looser tolerance, probably from fp64 -> fp32 rounding.
        try:
            # _, neigh_idx = kdtree.query(X, k=min(max_points, N_us_tot))
            # neigh_Xs = Xs_all[neigh_idx]
            neigh_idx = adj_mat[max(adj_mat.keys())][j]
            neigh_Xs = Xs_all[neigh_idx]
            w, _ = fin_diff_weights(X, neigh_Xs, diff_order, diff_acc, "abs_weight_norm", atol=2e-3, eps=18e-8)
        except ConvergenceError as e:
            # Unable to find suitable weights.
            status, err_msg = e.args
            c_print(f'{i = }, {err_msg = }, {status = }', color='bright_magenta')
            raise ConvergenceError(f'Could not find weights for {X.tolist()}') from None

    # # Only create edge if weight is not 0
    mask = torch.abs(w) > 1e-5
    w_want = w[mask]

    neigh_idx_want = torch.tensor(neigh_idx[mask])
    source_nodes = torch.full((len(neigh_idx_want),), j, dtype=torch.long)
    edge_idx = torch.stack([neigh_idx_want, source_nodes], dim=0)

    # Pytorch multiprocessing bug
    edge_idx, w_want = edge_idx.numpy(), w_want.numpy()
    return edge_idx, w_want


global_kdtree, global_Xs_all, global_adj_mat = None, None, None
def _init_pool(Xs_all, adj_mat):
    global global_kdtree, global_Xs_all, global_adj_mat
    global_Xs_all, global_adj_mat = Xs_all, adj_mat


def calc_coeff(point_dict: dict[int, Point], diff_acc: int, diff_order: tuple[int, int], adj_mat: dict):
    """ Calculate finite difference coefficients.
    Xs_all: torch.Tensor [N_nodes, 2]. All nodes in the graph.
    point_dict: dict[int, Point]. Dictionary of points where gradients are calculated
    N_nodes: int
    diff_acc: int
    diff_order: Tuple[int, int]
    """

    Xs_all = torch.stack([point.X for point in point_dict.values()])
    N_us_tot = len(point_dict)
    min_points = min(50, N_us_tot)
    max_points = min(251, N_us_tot + 1)

    param_dict = {"order": diff_order, "acc": diff_acc, "N_us_tot": N_us_tot, "min_points": min_points, "max_points": max_points}

    mp_args = [(j, point.X, param_dict) for j, point in point_dict.items()]

    with Pool(processes=16, initializer=_init_pool, initargs=(Xs_all, adj_mat)) as pool:
        results = pool.starmap(_calc_coeff_single, mp_args)


    edge_idxs, weights = zip(*results)
    # Pytorch multiprocessing bug
    edge_idxs = np.concatenate(edge_idxs, axis=1)
    weights = np.concatenate(weights)
    edge_idxs, weights = torch.from_numpy(edge_idxs), torch.from_numpy(weights)

    return edge_idxs, weights


def main():
    import pickle

    torch.set_printoptions(precision=3, sci_mode=False)

    with open("../save.pkl", "rb") as f:
        error_point = pickle.load(f)

    c_print(f"Error point: {error_point['X']}", color="bright_cyan")
    c_print(f"Error: {error_point['err']}", color="bright_cyan")

    # Define the center point coordinates
    x0, y0 = error_point['X']  # Center point

    points = error_point['neigh_Xs']  # List of (x, y) coordinates of neighboring points
    print(points.shape)

    # Specify the derivative order to approximate (e.g., first derivative with respect to x)
    derivative_order = (0,2)  # (k_x, k_y)

    # Set the maximum total degree of the monomials (order of accuracy)
    m = 4   # Should be at least the sum of the derivative orders

    # Compute the finite difference weights
    from matplotlib import pyplot as plt
    plt.scatter(*zip(*points))
    xmin, xmax = points[:, 0].min(), points[:, 0].max()
    ymin, ymax = points[:, 1].min(), points[:, 1].max()
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.show()

    center = torch.tensor([x0, y0], dtype=torch.float32)

    # Compute the finite difference weights
    weights, status = fin_diff_weights(center, points, derivative_order, m, method="abs_weight_norm", atol=100e-4, eps=6e-8)

    # # Display the computed weights
    # print()
    print(f'err = {status["mean_err"]:.3g}, {status["max_err"] = :.3g}, {status["abs_res"] = :.3g}')
    for w, p in zip(weights, points):
        if w.abs() > 1e-5:
            print(f"{p}: {w:.3f}")
            plt.scatter(*p, c='r')


    plt.scatter(x0, y0, c='g')
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.show()


# Example usage
if __name__ == "__main__":
    main()