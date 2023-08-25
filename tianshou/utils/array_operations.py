from typing import Union, overload

import numpy as np
import torch


@overload
def array_clip(arr: np.ndarray, min: float, max: float) -> np.ndarray:
    ...
@overload
def array_clip(arr: torch.Tensor, min: float, max: float) -> torch.Tensor:
    ...
@overload
def array_clip(arr: float, min: float, max: float) -> float:
    ...
def array_clip(arr: Union[float, np.ndarray, torch.Tensor], min: float, max: float) -> Union[float, np.ndarray, torch.Tensor]:
    """Clip the value of a scalar or a tensor between min and max."""
    if isinstance(arr, torch.Tensor):
        return torch.clamp(arr, min, max)
    elif isinstance(arr, np.ndarray):
        return np.clip(arr, min, max)
    else:
        return min if arr < min else max if arr > max else arr

@overload
def array_mean(arr: float) -> float:
    ...
@overload
def array_mean(arr: np.ndarray) -> np.ndarray:
    ...
@overload
def array_mean(arr: torch.Tensor) -> torch.Tensor:
    ...

def array_mean(arr: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
    """Compute the mean of a scalar or a tensor."""
    if isinstance(arr, torch.Tensor):
        return torch.mean(arr.float(), dim = 0)
    elif isinstance(arr, np.ndarray):
        return np.mean(arr, axis = 0)
    else:
        return arr


@overload
def array_var(arr: float) -> float:
    ...
@overload
def array_var(arr: np.ndarray) -> np.ndarray:
    ...
@overload
def array_var(arr: torch.Tensor) -> torch.Tensor:
    ...
def array_var(arr: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
    """Compute the standard deviation of a scalar or a tensor."""
    if isinstance(arr, torch.Tensor):
        return torch.var(arr.float(),unbiased=False, dim = 0)
    elif isinstance(arr, np.ndarray):
        return np.var(arr, axis = 0)
    else:
        return arr

