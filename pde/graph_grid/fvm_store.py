from enum import Flag, auto
from dataclasses import dataclass
import torch

class EdgeBCTypes(Flag):
    """ Point types for time dependent problems. """

    Dirich = auto()  # Fixed value point
    Neuman = auto()  # Fixed gradient
    Both = Dirich | Neuman  # Both Dirichlet and Neumann BC enforced on point.
    Farfield = auto()  # Farfield boundary condition


@dataclass
class Edge:
    """"""
    edge_type: list[EdgeBCTypes]
    U: list[float] = None
    dUdn: list[float] = None
    rho_far: float = None

    def __post_init__(self):
        for e, u, dudn in zip(self.edge_type, self.U, self.dUdn, strict=True):
            if EdgeBCTypes.Dirich in e:
                assert u is not None, "Dirichlet BC requires a value."
                assert dudn is None, "Dirichlet BC does not require a gradient."

            if EdgeBCTypes.Neuman in e:
                assert dudn is not None, "Neumann BC requires a gradient."
                assert u is None, "Neumann BC does not require a value."

            if EdgeBCTypes.Farfield in e:
                assert self.rho_far is not None, "Farfield BC requires a rho_far value."

        # Replace Nones in U and dUdn with const
        self.U = [float('NaN') if u is None else u for u in self.U]
        self.dUdn = [float('NaN') if d is None else d for d in self.dUdn]

    def dirichlet(self):
        return [EdgeBCTypes.Dirich in e for e in self.edge_type]

    def neumann(self):
        return [EdgeBCTypes.Neuman in e for e in self.edge_type]

    def farfield(self):
        return [EdgeBCTypes.Farfield in e for e in self.edge_type]
