from dataclasses import dataclass
import torch

@dataclass
class ConfigFarfield:
    mode: str = "decay"    # {decay, farfield, interior} BC

    # Farfield physical parameters
    v_far: float = 0
    rho_far: float = 0.3

    # Farfield limit / simulation parameters
    decay_tau: float = 0.001
    beta_tau: float = 0.33

    decay_beta: float = 0.002

@dataclass
class ConfigFVM:

    # solver parameters
    dt: float = 5e-6
    n_iter: int = 50000

    # mesh parameters
    min_A: float = 2e-4
    max_A: float = 7e-3
    lnscale: float = 4

    # Physical parameters
    viscosity: float = 6e-5
    visc_bulk: float = 2e-5
    thermal_cond: float = 0.01

    gamma: float = 1.4  # Ratio of specific heats
    C_v: float = 718     # Specific heat at constant volume
    #M: float = 0.029    # Molar mass of air

    # Stability parameters
    v_factor: float = 1     # Modification for velocity KT scheme
    # bulk_visc_lim: float = 0.25
    lim_p: int = 4          # Order of limiter (1 for BJ)
    lim_K: int = 0.25

    # Exit parameters
    exit_cfg: ConfigFarfield = None

    def __post_init__(self):
        self.exit_cfg = ConfigFarfield()
