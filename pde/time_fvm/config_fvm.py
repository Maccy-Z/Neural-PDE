from dataclasses import dataclass

@dataclass
class ConfigFarfield:
    mode: str = "decay"    # {decay, farfield} BC

    decay_tau: float = 5.
    beta_tau: float = 0.33

    S_upper: float = 0.1
    f_S0: float = 0.1
    f_S_offset: float = 0.1     # When to turn off U_charachteristic

@dataclass
class ConfigFVM:

    # solver parameters
    dt: float = 0.004
    n_iter: int = 5001

    # mesh parameters
    min_A: float = 0.5e-4
    max_A: float = 5e-3
    lnscale: float = 4

    # Physical parameters
    viscosity: float = 1e-5
    visc_bulk: float = 5e-4
    c: float = 1.
    # Stability parameters
    v_factor: float = 0.5     # Modification for velocity KT scheme
    bulk_visc_lim: float = 0.25

    # Exit parameters
    v_far: float = 0.1
    p_far: float = 1
    exit_cfg: ConfigFarfield = None

    def __post_init__(self):
        self.exit_cfg = ConfigFarfield()