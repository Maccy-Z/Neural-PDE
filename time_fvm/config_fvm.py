from dataclasses import dataclass
import torch

@dataclass
class ConfigFarfield:
    mode: str = "farfield_blended"    # {decay, farfield, farfield_blended, adaptive, interior} BC

    # Farfield physical parameters
    v_far: float = 2.
    rho_far: float = 1
    T_far: float = 100

    # Farfield limit / simulation parameters
    decay_tau: float = 0.05
    beta_tau: float = 0.33

    decay_beta: float = 0.1

@dataclass
class ConfigFVM:

    # solver parameters
    dt: float = 1e-4
    n_iter: int = 50000

    # mesh parameters
    min_A: float = 0.5e-3
    max_A: float = 2e-3
    lnscale: float = 2

    # Physical parameters
    viscosity: float = 1000e-5     # At room temp
    visc_bulk: float = 1e-5
    thermal_cond: float = 1e-6
    S_const: float = 110.4       # Sutherland's constant

    gamma: float = 1.2  # Ratio of specific heats
    C_v: float = 2     # Specific heat at constant volume

    # Stability parameters
    v_factor: float = 0.1     # Modification for velocity KT scheme
    lim_p: int = 4          # Order of limiter (1 for BJ)
    lim_K: int = 0.1

    # Exit parameters
    exit_cfg: ConfigFarfield = None

    def __post_init__(self):
        self.exit_cfg = ConfigFarfield()

        self.R = (self.gamma - 1) * self.C_v        # specific gas constant
