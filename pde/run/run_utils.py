import torch

from pde.config import Config

class MetricTracker:
    tracking_dict: dict[str, list[torch.Tensor]]
    def __init__(self, cfg: Config):
        self.tracking_dict = {}

    def add_metric(self, new_vals: dict[str, torch.Tensor]):
        for key in new_vals.keys():
            if key not in self.tracking_dict:
                self.tracking_dict[key] = []

        for key, val in new_vals.items():
                self.tracking_dict[key].append(val)

    def get_mean_metrics(self, keys: list[str]) -> dict[str, torch.Tensor]:
        """ Return average metrics, and reset. """
        return_dict = {}
        for key in keys:
            if key not in self.tracking_dict:
                raise ValueError(f"Key {key} not found in tracking_dict")
            all_metrics = torch.stack(self.tracking_dict[key])
            mean_metric = all_metrics.mean()
            return_dict[key] = mean_metric
            # Reset
            self.tracking_dict[key] = []

        return return_dict

    def get_metrics(self, key: str) -> torch.Tensor:
        """ Return metrics, and reset. """
        if key not in self.tracking_dict:
            raise ValueError(f"Key {key} not found in tracking_dict")
        all_metrics = torch.stack(self.tracking_dict[key])
        # Reset
        self.tracking_dict[key] = []

        return all_metrics
