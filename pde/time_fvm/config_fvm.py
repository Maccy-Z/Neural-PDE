from dataclasses import dataclass
import torch

@dataclass
class ConfigFarfield:
    mode: str = "interior"    # {decay, farfield, interior} BC

    # Farfield physical parameters
    v_far: float = 0.1
    rho_far: float = 1

    # Farfield limit / simulation parameters
    decay_tau: float = 5.
    beta_tau: float = 0.33

    decay_beta: float = 0.002

@dataclass
class ConfigFVM:

    # solver parameters
    dt: float = 0.005
    n_iter: int = 10001

    # mesh parameters
    min_A: float = 5e-4
    max_A: float = 10e-3
    lnscale: float = 4

    # Physical parameters
    viscosity: float = 1e-5
    visc_bulk: float = 1e-5
    c: float = 1.
    C_v: float = 1.4
    # Stability parameters
    v_factor: float = 1     # Modification for velocity KT scheme
    bulk_visc_lim: float = 0.25

    # Exit parameters
    exit_cfg: ConfigFarfield = None

    def __post_init__(self):
        self.exit_cfg = ConfigFarfield()
