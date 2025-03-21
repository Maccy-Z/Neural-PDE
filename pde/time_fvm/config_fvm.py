from dataclasses import dataclass

@dataclass
class ConfigFVM:

    # solver parameters
    dt: float = 0.007
    n_iter: int = 20001

    # mesh parameters
    min_A: float = 1e-4
    max_A: float = 5e-3
    lnscale: float = 4

    # Physical parameters
    viscosity: float = 1e-6
    visc_bulk: float = 0e-5
    c: float = 1.
    # Stability parameters
    v_factor: float = 0.1     # Modification for velocity KT scheme
    bulk_visc_lim: float = 1.

    # Exit parameters
    v_far: float = 0.1
    p_far: float = 1
    exit_mode: str = "decay"    # {decay, farfield} BC
    decay_rate: float = 20
