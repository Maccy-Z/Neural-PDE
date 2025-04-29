from dataclasses import dataclass
import torch

@dataclass
class ConfigFarfield:
    mode: str = "farfield_blended"    # {decay, farfield, interior} BC

    # Farfield physical parameters
    v_far: float = 0.0
    rho_far: float = 0.4
    T_far: float = 75

    # Farfield limit / simulation parameters
    decay_tau: float = 0.05
    beta_tau: float = 0.33

    decay_beta: float = 0.1

@dataclass
class ConfigFVM:

    # solver parameters
    dt: float = 1e-6
    n_iter: int = 50000

    # mesh parameters
    min_A: float = 1e-4
    max_A: float = 3e-3
    lnscale: float = 3

    # Physical parameters
    viscosity: float = 3e-5
    visc_bulk: float = 1e-5
    thermal_cond: float = 1e-6

    gamma: float = 1.4  # Ratio of specific heats
    C_v: float = 700     # Specific heat at constant volume

    # Stability parameters
    v_factor: float = 1     # Modification for velocity KT scheme
    # bulk_visc_lim: float = 0.25
    lim_p: int = 4          # Order of limiter (1 for BJ)
    lim_K: int = 1

    # Exit parameters
    exit_cfg: ConfigFarfield = None

    def __post_init__(self):
        self.exit_cfg = ConfigFarfield()

        self.R = (self.gamma - 1) * self.C_v
