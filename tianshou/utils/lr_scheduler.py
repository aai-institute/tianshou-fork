from typing import Dict, List

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR


class MultipleLRSchedulers:
    """A wrapper for multiple learning rate schedulers.

    Every time :meth:`~tianshou.utils.MultipleLRSchedulers.step` is called,
    it calls the step() method of each of the schedulers that it contains.
    Example usage:
    ::

        scheduler1 = ConstantLR(opt1, factor=0.1, total_iters=2)
        scheduler2 = ExponentialLR(opt2, gamma=0.9)
        scheduler = MultipleLRSchedulers(scheduler1, scheduler2)
        policy = PPOPolicy(..., lr_scheduler=scheduler)
    """

    def __init__(self, *args: torch.optim.lr_scheduler.LambdaLR):
        self.schedulers = args

    def step(self) -> None:
        """Take a step in each of the learning rate schedulers."""
        for scheduler in self.schedulers:
            scheduler.step()

    def state_dict(self) -> List[Dict]:
        """Get state_dict for each of the learning rate schedulers.

        :return: A list of state_dict of learning rate schedulers.
        """
        return [s.state_dict() for s in self.schedulers]

    def load_state_dict(self, state_dict: List[Dict]) -> None:
        """Load states from state_dict.

        :param List[Dict] state_dict: A list of learning rate scheduler
            state_dict, in the same order as the schedulers.
        """
        for s, sd in zip(self.schedulers, state_dict):
            s.__dict__.update(sd)


def get_linear_lr_schedular(
    optim: torch.optim.Optimizer,
    step_per_epoch: int,
    step_per_collect: int,
    epochs: int,
):
    """Decay learning rate to 0 linearly."""
    max_update_num = np.ceil(step_per_epoch / step_per_collect) * epochs
    lr_scheduler = LambdaLR(optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num)
    return lr_scheduler
