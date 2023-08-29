from copy import copy
from numbers import Number
from typing import List, Optional, Protocol, Union, overload

import numpy as np
import torch

from tianshou.utils.array_operations import array_clip, array_mean, array_var


class MovAvg(object):
    """Class for moving average.

    It will automatically exclude the infinity and NaN. Usage:
    ::

        >>> stat = MovAvg(size=66)
        >>> stat.add(torch.tensor(5))
        5.0
        >>> stat.add(float('inf'))  # which will not add to stat
        5.0
        >>> stat.add([6, 7, 8])
        6.5
        >>> stat.get()
        6.5
        >>> print(f'{stat.mean():.2f}±{stat.std():.2f}')
        6.50±1.12
    """

    def __init__(self, size: int = 100) -> None:
        super().__init__()
        self.size = size
        self.cache: List[np.number] = []
        self.banned = [np.inf, np.nan, -np.inf]

    def add(
        self, data_array: Union[Number, np.number, list, np.ndarray, torch.Tensor]
    ) -> float:
        """Add a scalar into :class:`MovAvg`.

        You can add ``torch.Tensor`` with only one element, a python scalar, or
        a list of python scalar.
        """
        if isinstance(data_array, torch.Tensor):
            data_array = data_array.flatten().cpu().numpy()
        if np.isscalar(data_array):
            data_array = [data_array]
        for number in data_array:  # type: ignore
            if number not in self.banned:
                self.cache.append(number)
        if self.size > 0 and len(self.cache) > self.size:
            self.cache = self.cache[-self.size:]
        return self.get()

    def get(self) -> float:
        """Get the average."""
        if len(self.cache) == 0:
            return 0.0
        return float(np.mean(self.cache))  # type: ignore

    def mean(self) -> float:
        """Get the average. Same as :meth:`get`."""
        return self.get()

    def std(self) -> float:
        """Get the standard deviation."""
        if len(self.cache) == 0:
            return 0.0
        return float(np.std(self.cache))  # type: ignore

class NormaliserProtocol(Protocol):
    def norm(self, data_array: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
        pass
    def update(self, data_array: Union[np.ndarray, torch.Tensor]) -> None:
        pass


class RunningMeanStd(NormaliserProtocol):
    """Calculates the running mean and std of a data stream.

    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

    :param mean: the initial mean estimation for data array. Default to 0.
    :param std: the initial standard error estimation for data array. Default to 1.
    :param float clip_max: the maximum absolute value for data array. Default to
        10.0.
    :param float epsilon: To avoid division by zero.
    :param update freq: update the mean and var used for standardisation every k steps, 0 means no standardisation,
    1 is normal running mean
    """

    def __init__(
        self,
        mean: Union[float, np.ndarray, torch.Tensor] = 0.0,
        std: Union[float, np.ndarray, torch.Tensor] = 1.0,
        clip_max: Optional[float] = 10.0, #todo remove default and set it where it is used
        epsilon: float = np.finfo(np.float32).eps.item(),
        update_freq: int = 1,
    ) -> None:
        self.mean, self.stale_mean, self.var, self.stale_var = mean,mean,std, std
        self.clip_max = clip_max
        self.count = 0
        self.eps = epsilon
        self.update_freq = update_freq

    @overload
    def norm(self, arr: float) -> float:
        ...

    @overload
    def norm(self, arr: np.ndarray) -> np.ndarray:
        ...

    @overload
    def norm(self, arr: torch.Tensor) -> torch.Tensor:
        ...

    def norm(self, data_array: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
        data_array = (data_array - self.stale_mean) / np.sqrt(self.stale_var + self.eps)
        if self.clip_max:
            data_array = array_clip(data_array, -self.clip_max, self.clip_max)
        return data_array
    @overload
    def unnormalise(self, arr: float) -> float:
        ...

    @overload
    def unnormalise(self, arr: np.ndarray) -> np.ndarray:
        ...

    @overload
    def unnormalise(self, arr: torch.Tensor) -> torch.Tensor:
        ...

    def unnormalise(self, data_array: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
        data_array = data_array * np.sqrt(self.stale_var + self.eps) + self.stale_mean
        return data_array

    @overload
    def update_and_norm(self, arr: float) -> float:
        ...

    @overload
    def update_and_norm(self, arr: np.ndarray) -> np.ndarray:
        ...

    @overload
    def update_and_norm(self, arr: torch.Tensor) -> torch.Tensor:
        ...


    def update_and_norm(self, data_array: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
        self.update(data_array)
        return self.norm(data_array)


    def update(self, data_array: Union[np.ndarray, torch.Tensor]) -> None:
        """Add a batch of item into RMS with the same shape, modify mean/var/count."""
        batch_mean, batch_var = array_mean(data_array), array_var(data_array)
        batch_count = len(data_array)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = m_2 / total_count

        self.mean, self.var = new_mean, new_var
        if self.count//self.update_freq != total_count//self.update_freq:
            self.stale_mean = copy(self.mean)
            self.stale_var = copy(self.var)
        self.count = total_count

