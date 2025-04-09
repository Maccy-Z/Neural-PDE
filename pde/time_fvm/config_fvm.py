from dataclasses import dataclass
import torch

@dataclass
class ConfigFarfield:
    mode: str = "decay"    # {decay, farfield, interior} BC

    # Farfield physical parameters
    v_far: float = 0.1
    rho_far: float = 1

    # Farfield limit / simulation parameters
    decay_tau: float = 2.5
    beta_tau: float = 0.33

    decay_beta: float = 0.002

@dataclass
class ConfigFVM:

    # solver parameters
    dt: float = 0.01
    n_iter: int = 10000

    # mesh parameters
    min_A: float = 2e-4
    max_A: float = 5e-3
    lnscale: float = 4

    # Physical parameters
    viscosity: float = 5e-4
    visc_bulk: float = 0e-4
    thermal_cond: float = 1e-5

    c: float = 1.
    gamma: float = 1.4
    C_v: float = 0.01

    # Stability parameters
    v_factor: float = 1     # Modification for velocity KT scheme
    # bulk_visc_lim: float = 0.25
    lim_p: int = 4          # Order of limiter (1 for BJ)
    lim_K: int = 2

    # Exit parameters
    exit_cfg: ConfigFarfield = None

    def __post_init__(self):
        self.exit_cfg = ConfigFarfield()
