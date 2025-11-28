import torch
from dataclasses import dataclass
from enum import Flag, auto


class P_Types(Flag):
    """ Types of points.
        Note: Checks should be done as PRIMITIAVE in POINT_VALUE.
    """
    # Primitive types
    PDE = auto() # Standard PDE: u = f(us, x) enforced on this point.
    NONE = auto() # No function on point. Ghost point. u function inherited from central DERIV point.

    FIX = auto()   # Dirichlet: u = Dirichlet(x) enforced on this point.
    DERIV = auto()  # deriv(u, us) = Neumann(x) enforced on this point. Can be used as central Neumann derivative or edge, depending on other nodes.

    UPDATE = auto()   # u requires fitting on point.

    # User BC types
    DirichBC = FIX  # Dirichlet BC enforced on point.
    NeumCentralBC = DERIV | PDE | UPDATE # Neumann BC + PDE enforced on point.
    NeumOffsetBC = DERIV | UPDATE # Only Neumann BC on point.
    BothBC = FIX | DERIV  # Both Dirichlet and Neumann BC enforced on point.
    Ghost = NONE | UPDATE # Ghost point

    # Normal point
    Normal = PDE | UPDATE # Standard PDE enforced on point.

class P_TimeTypes(Flag):
    """ Point types for time dependent problems. """
    FIXED = auto()  # Fixed value point. No fitting required.
    MANUAL = auto()  # Manually set value. No fitting required.

    NORMAL = auto()  # Normal point. Fitting required.
    EXIT = auto()  # Exit point. Fit Neumann boundary = 0.

@dataclass
class Deriv:
    """ Stores all derivative boundary conditions on a point.
        sum{ w_m * d^n u_k / dx_i dx_j } = value
    """
    comp: list[int]     # Which components of us to apply derivative to.
    orders: list[tuple[int, int]]       # Derivative orders. (i, j) = d^i/dx_i d^j/dx_j
    value: float            # Sum values
    weights: list[float] = None # Weights for each derivative order. If None, all weights are 1.

    def __post_init__(self):
        assert len(self.comp) == len(self.orders), f"Number of components must equal number of derivative orders, {len(self.comp) = }, {len(self.orders) = }."
        if self.weights is None:
            self.weights = [1.0] * len(self.comp)

@dataclass
class Point:
    """ value:  If NORMAL, value = initial value.
                If BOUNDARY, value = boundary value.
        derivatives: Sum of derivatives = value. """
    point_type: P_Types
    X: torch.Tensor
    value: float|list[float] = None
    derivatives: list[Deriv] = None
    n_deriv: int = None
    is_dirichlet: list[bool] = 1  # If DERIV, which derivatives are Dirichlet (i.e. orders == [(0, 0)]).

    def __post_init__(self):
        if P_Types.DERIV in self.point_type:
            assert self.derivatives is not None, "Derivatives must be provided for DERIV points."
        else:
            assert self.derivatives is None, "Derivatives must be None for non-DERIV points."

        if P_Types.DERIV in self.point_type:
            self.n_deriv = len(self.derivatives)

            # Mark out Dirichlet derivatives specified using Neuman bcs.
            self.is_dirichlet = []
            if self.derivatives is not None:
                for d_idx, d in enumerate(self.derivatives):
                    if d.orders == [(0, 0)]:
                        self.is_dirichlet.append(True)
                    else:
                        self.is_dirichlet.append(False)



class DerivGraph:
    """
    A sparse representation of a finite-difference operator (derivative stencil).
      - edge_idx: LongTensor of shape (2, nnz), giving the (row, col) indices
      - weights:  FloatTensor of shape (nnz,), giving the nonzero values
      - shape:    A tuple (rows, cols) for this operator
      - device:   Which device ('cpu' or 'cuda') the tensors are on
    """

    def __init__(self, edge_idx, weights, shape, device="cpu"):
        self.edge_idx = edge_idx
        self.weights = weights
        self.shape = shape
        self.device = device

        if device == "cuda":
            self.edge_idx = self.edge_idx.cuda(non_blocking=True)
            self.weights = self.weights.cuda(non_blocking=True)


    def coo(self):
        coo = torch.sparse_coo_tensor(self.edge_idx, self.weights, size=self.shape, device=self.device)
        return coo

    @staticmethod
    def add(D1, D2):
        """
        Return a new DerivGraph whose operator is D1 + D2.

        Requirements:
          - D1.shape == D2.shape
          - Both are on the same device or you handle cross-device copying
        """
        assert D1.shape == D2.shape, "Shapes must match for addition."
        assert D1.device == D2.device, "Devices must match for addition."
        # Create PyTorch sparse COO tensors from each DerivGraph
        A1 = D1.coo()
        A2 = D2.coo()

        # Sparse addition
        A_sum = A1 + A2

        # Coalesce so that indices/values are unique and consolidated
        A_sum = A_sum.coalesce()

        # Extract new edges/weights
        new_edge_idx = A_sum.indices()
        new_weights = A_sum.values()

        return DerivGraph(
            edge_idx=new_edge_idx,
            weights=new_weights,
            shape=D1.shape,
            device=D1.device
        )

    @staticmethod
    def compose(D1, D2, mask=None):
        """
        Return a new DerivGraph whose operator is the composition (matrix multiplication) D1 * D2.
        For sequential operators M1 and M2, the new one is M1 @ (eye[Mask]) @ M2.

        Requirements:
          - The inner dimensions must match: D1.shape[1] == D2.shape[0]
        """
        # Check dimension compatibility
        assert D1.shape[1] == D2.shape[0], (
            "Inner dimensions must match for composition: "
            f"D1 is {D1.shape}, D2 is {D2.shape}."
        )
        assert D1.device == D2.device, "Devices must match for composition."
        if mask is not None:
            assert mask.shape[0] == D1.shape[0], "Mask must have same number of rows as D1."
            mask = torch.diag(mask).to_sparse_coo().to(D1.device)

        # Build PyTorch sparse COO for each
        A1 = D1.coo()
        A2 = D2.coo()

        # Sparse matrix multiplication: result has shape (D1.shape[0], D2.shape[1])
        if mask is not None:
            A1 = torch.sparse.mm(A1, mask)
        A12 = torch.sparse.mm(A1, A2)
        A12 = A12.coalesce()

        # Build new edge_idx, weights, and shape
        new_edge_idx = A12.indices()
        new_weights = A12.values()
        new_shape = (D1.shape[0], D2.shape[1])

        return DerivGraph(
            edge_idx=new_edge_idx,
            weights=new_weights,
            shape=new_shape,
            device=D1.device
        )


def main():
    torch.set_printoptions(precision=3, sci_mode=False)



if __name__ == "__main__":
    main()