from dataclasses import dataclass

@dataclass
class ConfigFVM:

    # solver parameters
    dt: float = 0.008
    n_iter: int = 20001

    # mesh parameters
    min_A: float = 1e-4
    max_A: float = 5e-3
    lnscale: float = 4

    # Physical parameters
    viscosity: float = 0.5e-3
    visc_bulk: float = 5e-3
    c: float = 1.
    v_far: float = 0.1
    p_far: float = 1
    v_factor = 0.1     # Modification for velocity KT scheme
    exit_mode: str = "decay"    # {decay, farfield} BC
    decay_rate: float = 20
