from collections.abc import Callable
from typing import Any
import numpy as np
from matplotlib.figure import Figure
from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter

from tianshou.utils.logger.base import VALID_LOG_VALS_TYPE, BaseLogger
from tianshou.utils.warning import deprecation


class TensorboardLogger(BaseLogger):
    """A logger that relies on tensorboard SummaryWriter by default to visualize and log statistics.

    :param SummaryWriter writer: the writer to log data.
    :param train_interval: the log interval in log_train_data(). Default to 1000.
    :param test_interval: the log interval in log_test_data(). Default to 1.
    :param update_interval: the log interval in log_update_data(). Default to 1000.
    :param info_interval: the log interval in log_info_data(). Default to 1.
    :param save_interval: the save interval in save_data(). Default to 1 (save at
        the end of each epoch).
    :param write_flush: whether to flush tensorboard result after each
        add_scalar operation. Default to True.
    """

    def __init__(
        self,
        writer: SummaryWriter,
        train_interval: int = 1000,
        test_interval: int = 1,
        update_interval: int = 1000,
        info_interval: int = 1,
        save_interval: int = 1,
        write_flush: bool = True,
    ) -> None:
        super().__init__(train_interval, test_interval, update_interval, info_interval)
        self.save_interval = save_interval
        self.write_flush = write_flush
        self.last_save_step = -1
        self.writer = writer

    def write(self, step_type: str, step: int, data: dict[str, VALID_LOG_VALS_TYPE]) -> None:
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                self.writer.add_histogram(k, v, global_step=step, bins="auto")
            elif isinstance(v, Figure):
                self.writer.add_figure(k, v, global_step=step)
            else:
                self.writer.add_scalar(k, v, global_step=step)
        if self.write_flush:  # issue 580
            self.writer.flush()  # issue #482

    def save_data(
        self,
        epoch: int,
        env_step: int,
        gradient_step: int,
        save_checkpoint_fn: Callable[[int, int, int], str] | None = None,
    ) -> None:
        if save_checkpoint_fn and epoch - self.last_save_step >= self.save_interval:
            self.last_save_step = epoch
            save_checkpoint_fn(epoch, env_step, gradient_step)
            self.write("save/epoch", epoch, {"save/epoch": epoch})
            self.write("save/env_step", env_step, {"save/env_step": env_step})
            self.write(
                "save/gradient_step",
                gradient_step,
                {"save/gradient_step": gradient_step},
            )

    def restore_data(self) -> tuple[int, int, int]:
        ea = event_accumulator.EventAccumulator(self.writer.log_dir)
        ea.Reload()

        try:  # epoch / gradient_step
            epoch = ea.scalars.Items("save/epoch")[-1].step
            self.last_save_step = self.last_log_test_step = epoch
            gradient_step = ea.scalars.Items("save/gradient_step")[-1].step
            self.last_log_update_step = gradient_step
        except KeyError:
            epoch, gradient_step = 0, 0
        try:  # offline trainer doesn't have env_step
            env_step = ea.scalars.Items("save/env_step")[-1].step
            self.last_log_train_step = env_step
        except KeyError:
            env_step = 0

        return epoch, env_step, gradient_step


class BasicLogger(TensorboardLogger):
    """BasicLogger has changed its name to TensorboardLogger in #427.

    This class is for compatibility.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        deprecation(
            "Class BasicLogger is marked as deprecated and will be removed soon. "
            "Please use TensorboardLogger instead.",
        )
        super().__init__(*args, **kwargs)
