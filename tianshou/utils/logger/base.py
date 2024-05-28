import typing
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from numbers import Number
from typing import Any

import numpy as np

VALID_LOG_VALS_TYPE = int | Number | np.number | np.ndarray | float
# It's unfortunate, but we can't use Union type in isinstance, hence we resort to this
VALID_LOG_VALS = typing.get_args(VALID_LOG_VALS_TYPE)
_VALID_LOG_DICT = dict[str, VALID_LOG_VALS_TYPE]
VALID_LOG_DICT = dict[str, VALID_LOG_VALS_TYPE | _VALID_LOG_DICT]

TRestoredData = dict[str, np.ndarray | dict[str, "TRestoredData"]]


class DataScope(Enum):
    TRAIN = "train"
    TEST = "test"
    UPDATE = "update"
    INFO = "info"


class BaseLogger(ABC):
    """The base class for any logger which is compatible with trainer."""

    def __init__(
        self,
        train_interval: int = 1000,
        test_interval: int = 1,
        update_interval: int = 1000,
        info_interval: int = 1,
        exclude_arrays: bool = True,
    ) -> None:
        """:param train_interval: the log interval in log_train_data(). Default to 1000.
        :param test_interval: the log interval in log_test_data(). Default to 1.
        :param update_interval: the log interval in log_update_data(). Default to 1000.
        :param info_interval: the log interval in log_info_data(). Default to 1.
        :param exclude_arrays: whether to exclude numpy arrays from the logger's output
        """
        super().__init__()
        self.train_interval = train_interval
        self.test_interval = test_interval
        self.update_interval = update_interval
        self.info_interval = info_interval
        self.exclude_arrays = exclude_arrays
        self.last_log_train_step = -1
        self.last_log_test_step = -1
        self.last_log_update_step = -1
        self.last_log_info_step = -1

    @abstractmethod
    def write(self, step_type: str, step: int, data: VALID_LOG_DICT) -> None:
        """Specify how the writer is used to log data.

        :param str step_type: namespace which the data dict belongs to.
        :param step: stands for the ordinate of the data dict.
        :param data: the data to write with format ``{key: value}``.
        """

    def finalize(self):
        pass

    def log_train_data(self, log_data: dict, env_step: int) -> None:
        """Use writer to log statistics generated during training.

        :param log_data: a dict containing the information returned by the collector during the train step.
        :param env_step: stands for the timestep the collector result is logged.
        """
        # TODO: move interval check to calling method
        if env_step - self.last_log_train_step >= self.train_interval:
            self.write(f"{DataScope.TRAIN.value}/env_step", env_step, log_data)
            self.last_log_train_step = env_step

    def log_test_data(self, log_data: dict, step: int) -> None:
        """Use writer to log statistics generated during evaluating.

        :param log_data:a dict containing the information returned by the collector during the evaluation step.
        :param step: stands for the timestep the collector result is logged.
        """
        # TODO: move interval check to calling method (stupid because log_test_data is only called from function in utils.py, not from BaseTrainer)
        if step - self.last_log_test_step >= self.test_interval:
            self.write(f"{DataScope.TEST.value}/env_step", step, log_data)
            self.last_log_test_step = step

    def log_update_data(self, log_data: dict, step: int) -> None:
        """Use writer to log statistics generated during updating.

        :param log_data:a dict containing the information returned during the policy update step.
        :param step: stands for the timestep the policy training data is logged.
        """
        # TODO: move interval check to calling method
        if step - self.last_log_update_step >= self.update_interval:
            self.write(f"{DataScope.UPDATE.value}/gradient_step", step, log_data)
            self.last_log_update_step = step

    def log_info_data(self, log_data: dict, step: int) -> None:
        """Use writer to log global statistics.

        :param log_data: a dict containing information of data collected at the end of an epoch.
        :param step: stands for the timestep the training info is logged.
        """
        if (
            step - self.last_log_info_step >= self.info_interval
        ):  # TODO: move interval check to calling method
            self.write(f"{DataScope.INFO.value}/epoch", step, log_data)
            self.last_log_info_step = step

    @abstractmethod
    def save_data(
        self,
        epoch: int,
        env_step: int,
        gradient_step: int,
        save_checkpoint_fn: Callable[[int, int, int], str] | None = None,
    ) -> None:
        """Use writer to log metadata when calling ``save_checkpoint_fn`` in trainer.

        :param epoch: the epoch in trainer.
        :param env_step: the env_step in trainer.
        :param gradient_step: the gradient_step in trainer.
        :param function save_checkpoint_fn: a hook defined by user, see trainer
            documentation for detail.
        """

    @abstractmethod
    def restore_data(self) -> tuple[int, int, int]:
        """Restore internal data if present and return the metadata from existing log for continuation of training.

        If it finds nothing or an error occurs during the recover process, it will
        return the default parameters.

        :return: epoch, env_step, gradient_step.
        """

    @classmethod
    @abstractmethod
    def restore_logged_data(
        cls,
        log_path: str,
    ) -> TRestoredData:
        """Load the logged data from disk for post-processing.

        :return: a dict containing the logged data.
        """


class LazyLogger(BaseLogger):
    """A logger that does nothing. Used as the placeholder in trainer."""

    def __init__(self) -> None:
        super().__init__()

    def write(self, step_type: str, step: int, data: dict[str, VALID_LOG_VALS_TYPE]) -> None:
        """The LazyLogger writes nothing."""

    def save_data(
        self,
        epoch: int,
        env_step: int,
        gradient_step: int,
        save_checkpoint_fn: Callable[[int, int, int], str] | None = None,
    ) -> None:
        pass

    def restore_data(self) -> tuple[int, int, int]:
        return 0, 0, 0

    @classmethod
    def restore_logged_data(cls, log_path: str) -> dict:
        return {}


def to_flat_dict(
    input_dict: dict[str, Any],
    delimiter: str = "/",
    exclude_arrays: bool = False,
    exclude_invalid_entries: bool = True,
    parent_key="",
) -> dict[str, Any]:
    """Flattens a nested dictionary by recursively traversing all levels and compressing the keys.

    :param input_dict: The nested dictionary to be flattened.
    :param delimiter: The delimiter used to separate the keys.
    :param exclude_arrays: Whether to exclude numpy arrays from the output.
    :param exclude_invalid_entries: Whether to exclude entries that are not of type VALID_LOG_VALS.
    :param parent_key: The parent key used as a prefix before the input_dict keys.
    :return: A flattened dictionary where the keys are compressed.
    """
    result = {}

    def add_to_result(
        cur_dict: dict,
        prefix: str = "",
    ) -> None:
        for key, value in cur_dict.items():
            if exclude_arrays and isinstance(value, np.ndarray):
                continue

            new_key = prefix + delimiter + key
            new_key = new_key.lstrip(delimiter)

            if isinstance(value, dict):
                add_to_result(
                    value,
                    new_key,
                )
            elif isinstance(value, VALID_LOG_VALS):
                result[new_key] = value

    add_to_result(input_dict, prefix=parent_key)
    return result
