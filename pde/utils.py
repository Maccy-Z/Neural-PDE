from matplotlib import pyplot as plt
import torch
import scipy.sparse as sp
# from scipy.sparse.csgraph import dijkstra


class AdjMatSelector:
    def __init__(self, triangles: torch.Tensor):
        adj_mat = self.triangle_to_adjacency(triangles)
        # self._D = dijkstra(adj_mat, directed=False, return_predecessors=False)
        # self._n = self._D.shape[0]


    def tri_to_n_hop(self, tris, hops=6):
        """
        Build an (undirected) n-hop adjacency matrix from a [n_tri,3] triangle index tensor.

        Args:
            tris: LongTensor of shape [n_tri, 3], each row is (a,b,c) indices of a triangle.

        Returns:
            adj: ByteTensor of shape [n_verts, n_verts], where adj[i,j] = 1 if (i,j) is an edge.
        """
        n_verts = int(tris.max().item()) + 1

        # same edge stacking
        e0 = tris[:, [0, 1]]
        e1 = tris[:, [1, 2]]
        e2 = tris[:, [2, 0]]
        edges = torch.cat([e0, e1, e2], dim=0)

        # mirror for undirected
        rev_edges = edges[:, [1, 0]]
        all_edges = torch.cat([edges, rev_edges], dim=0).t()  # shape [2, 6*n_tri]

        # values are all 1
        vals = torch.ones(all_edges.shape[1], dtype=torch.float32, device=tris.device)

        adj_mat = torch.sparse_coo_tensor(all_edges, vals, (n_verts, n_verts)).to_sparse_csr()

        M = adj_mat
        n_hop_idx = {}
        for n in range(2, hops + 1):
            M = M @ adj_mat

            n_hop_idx[n] = ((M.crow_indices(), M.col_indices()))

        # Get n-hop neighbours
        n_hop_reach_cols = {}
        for hop in [4, 5, 6]:
            crow_idxs, col_idxs = n_hop_idx[hop]
            reached_cols = {}
            for i in range(n_verts):
                start = crow_idxs[i]
                stop = crow_idxs[i + 1]
                reached_cols[i] = col_idxs[start:stop].to(torch.int32).numpy()

            n_hop_reach_cols[hop] = reached_cols

        return n_hop_reach_cols





def show_grid(u: torch.Tensor, title=None, origin="lower"):
    """
    Visualize a 2D grid of values using matplotlib
    """
    # Check if the input is a 2D grid
    if u.ndim == 1:
        u = u.reshape(1, -1)
    elif u.ndim > 2:
        raise ValueError("Input must be a 1D or 2D array")

    plt.figure()
    u = u.T.cpu().detach().numpy()

    plt.imshow(u, cmap='viridis', origin=origin)
    plt.colorbar()
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def get_split_indices(tensor_size, m):
    """
    Get pairs of indices to split a 1D tensor into m chunks of possibly uneven size.

    Parameters:
    tensor_size (int): The size of the input 1D tensor.
    m (int): The number of chunks to split the tensor into.

    Returns:
    list of tuples: A list of (start, end) index pairs for each chunk.
    """
    split_sizes = [(tensor_size // m) + (1 if x < (tensor_size % m) else 0) for x in range(m)]

    indices = [0] + [sum(split_sizes[:i + 1]) for i in range(m)]

    index_pairs = [(indices[i], indices[i + 1]) for i in range(len(indices) - 1)]

    return index_pairs


def clamp(n, min_val, max_val):
    return max(min(n, max_val), min_val)


def adjust_slice(slice_obj, start_adjust=0, stop_adjust=0):
    """Adjust the given slice object by modifying its start and stop values."""
    new_start = slice_obj.start + start_adjust
    new_stop = slice_obj.stop + stop_adjust
    return slice(new_start, new_stop)

def dict_key_by_value(d, value):
    for k, v in d.items():
        if v == value:
            return k

    raise ValueError(f"Value {value} not found in dictionary")



def setup_logging(debug=True):
    import logging
    import sys

    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)

    if debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.WARNING
    logging.basicConfig(level=log_level, stream=sys.stdout, format='\033[31m%(levelname)s: \033[33m%(message)s \033[0m')
    logging.info('Logging setup complete')

    torch.set_printoptions(precision=3, sci_mode=False)