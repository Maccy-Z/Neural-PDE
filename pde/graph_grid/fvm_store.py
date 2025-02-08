from enum import Flag, auto
from dataclasses import dataclass
import torch

class EdgeBCTypes(Flag):
    """ Point types for time dependent problems. """

    Dirich = auto()  # Fixed value point
    Neuman = auto()  # Fixed gradient
    Both = Dirich | Neuman  # Both Dirichlet and Neumann BC enforced on point.


@dataclass
class Edge:
    """"""
    edge_type: list[EdgeBCTypes]
    U: list[float] = None
    dUdn: list[float] = None

    def __post_init__(self):
        # Replace Nones in U and dUdn with const
        self.U = [1e10 if u is None else u for u in self.U]
        self.dUdn = [1e10 if d is None else d for d in self.dUdn]

    def dirichlet(self):
        return [EdgeBCTypes.Dirich in e for e in self.edge_type]

    def neumann(self):
        return [EdgeBCTypes.Neuman in e for e in self.edge_type]