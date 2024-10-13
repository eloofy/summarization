from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.optim import AdamW, SGD, RMSprop, Adagrad, Adam


class WarmupScheduler(LRScheduler):
    """
    Scheduler class with warmup
    """

    def __init__(
        self,
        optimizer: AdamW | SGD | RMSprop | Adagrad | Adam,
        warmup_steps: int,
        scheduler: CosineAnnealingLR | LinearLR,
        lr_start: float,
        last_epoch: int = -1,
    ):
        """
        Init class

        :param optimizer: optimizer obj
        :param warmup_steps: nu, warmup steps
        :param scheduler: scheduler class (CosLR, LinLR)
        :param lr_start: start lr
        :param last_epoch: last epoch
        """
        self.warmup_steps = warmup_steps
        self.scheduler = scheduler
        self.lr_start = lr_start
        self.current_step = 0
        super().__init__(optimizer, last_epoch)

    def step(self, epoch=None):
        """Make step"""

        self.current_step += 1
        if self.current_step < self.warmup_steps:
            lr = self.lr_start * (self.current_step / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
        else:
            self.scheduler.step(self.current_step - self.warmup_steps)

    def get_lr(self):
        """Get lr"""
        return self.scheduler.get_lr()


def get_warmup_scheduler(
    optimizer: AdamW | SGD | RMSprop | Adagrad | Adam,
    warmup_steps: int,
    scheduler: CosineAnnealingLR | LinearLR,
    lr_start: float,
):
    """
    Get warmup scheduler
    """
    return WarmupScheduler(optimizer, warmup_steps, scheduler, lr_start)