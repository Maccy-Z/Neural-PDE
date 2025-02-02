from dataclasses import dataclass

@dataclass()
class ConfigTime:
    time_domain: tuple[int] = (0, 0.08)
    timesteps: int = 80
    substeps: int = 2

    dt: float = None

    def __post_init__(self):
        self.dt = (self.time_domain[1] - self.time_domain[0]) / self.timesteps



