"""Custom learning rate schedulers with warmup."""
import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmupScheduler(_LRScheduler):
    """
    Cosine annealing scheduler with linear warmup.

    Args:
        optimizer: Wrapped optimizer.
        warmup_steps: Number of steps for linear warmup.
        max_steps: Total number of training steps.
        min_lr_ratio: Minimum learning rate as a fraction of base learning rate.
        last_epoch: The index of last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr_ratio: float = 0.1,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Calculate learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = (self.last_epoch + 1) / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing after warmup
            progress = (self.last_epoch - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            # Scale from min_lr_ratio to 1.0
            lr_factor = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_factor
            return [base_lr * lr_factor for base_lr in self.base_lrs]

