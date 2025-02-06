from enum import Flag, auto


class EdgeBCTypes(Flag):
    """ Point types for time dependent problems. """

    WALL = auto()  # Wall boundary condition. 0 flow.
    INLET = auto()  # Inlet boundary condition.
    EXIT = auto()  # Exit point.
