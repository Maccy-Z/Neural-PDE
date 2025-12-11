from enum import Flag, auto
from dataclasses import dataclass


class EdgeBCTypes(Flag):
    """ Point types for time dependent problems. """

    Dirich = auto()  # Fixed value point
    Neuman = auto()  # Fixed gradient
    Both = Dirich | Neuman  # Both Dirichlet and Neumann BC enforced on point.
    Farfield = auto()  # Farfield boundary condition
    Inlet = auto()


@dataclass
class Edge:
    """"""
    edge_type: list[EdgeBCTypes]
    U: list[float| None] = None
    dUdn: list[float | None] = None
    euler_wall: bool = False
    tag: str = None

    def __post_init__(self):
        for e, u, dudn in zip(self.edge_type, self.U, self.dUdn, strict=True):
            if EdgeBCTypes.Dirich in e:
                assert u is not None, "Dirichlet BC requires a value."
                assert dudn is None, "Dirichlet BC does not require a gradient."

            if EdgeBCTypes.Neuman in e:
                assert dudn is not None, "Neumann BC requires a gradient."
                assert u is None, "Neumann BC does not require a value."

        # Replace Nones in U and dUdn with const
        self.U = [float('NaN') if u is None else u for u in self.U]
        self.dUdn = [float('NaN') if d is None else d for d in self.dUdn]

    def dirichlet(self):
        return [EdgeBCTypes.Dirich in e for e in self.edge_type]

    def neumann(self):
        return [EdgeBCTypes.Neuman in e for e in self.edge_type]

    def farfield(self) -> bool:
        farfield = [EdgeBCTypes.Farfield in e for e in self.edge_type]
        is_farfield = all(farfield)
        assert is_farfield or not any(farfield), f"All boundary must be farfield if any are: {farfield}"
        return is_farfield

    def inlet(self) -> bool:
        inlet = [EdgeBCTypes.Inlet in e for e in self.edge_type]
        is_inlet = all(inlet)
        assert is_inlet or not any(inlet), f"All boundary must be inlet if any are: {inlet}"
        return is_inlet
