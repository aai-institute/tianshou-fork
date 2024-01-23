from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

import gymnasium as gym
import numpy as np
import torch
from torch.nn.utils.convert_parameters import _check_param_device

from tianshou.data import ReplayBuffer, SequenceSummaryStats
from tianshou.data.batch import Batch, BatchProtocol, arr_type
from tianshou.data.types import ActBatchProtocol, ObsBatchProtocol, RolloutBatchProtocol
from tianshou.policy import BasePolicy, TrainingStats
from tianshou.policy.base import TLearningRateScheduler


def vector_to_gradient(vec: torch.Tensor, parameters: Iterable[torch.Tensor]) -> None:
    r"""Convert one vector to the parameters.

    :param vec: a single vector represents the parameters of a model.
    :param parameters: an iterator of Tensors that are the parameters of a model.
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError(f"expected torch.Tensor, but got: {torch.typename(vec)}")
    # Flag for the device where the parameter is located
    param_device = None

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.grad = vec[pointer : pointer + num_param].view_as(param).data

        # Increment the pointer
        pointer += num_param


class BatchWithReturnsAndDeltasProtocol(RolloutBatchProtocol, Protocol):
    """A batch with episode return and corresponding policy perturbation directions."""

    returns: arr_type
    plus_returns: arr_type
    minus_returns: arr_type
    return_std: float
    deltas: arr_type


@dataclass(kw_only=True)
class ARSTrainingStats(TrainingStats):
    loss: SequenceSummaryStats


class ARSPolicy(BasePolicy[ARSTrainingStats]):
    """Implementation of Augmented Random Search. https://arxiv.org/abs/1803.07055
    :param n_top: Number of top returns to use for gradient update. If None, use all.
    """

    def __init__(
        self,
        actor: torch.nn.Module,
        optim: torch.optim.Optimizer,
        action_space: gym.Space,
        n_top: int | None = None,
        observation_space: gym.Space | None = None,
        action_scaling: bool = True,
        action_bound_method: Literal["clip", "tanh"] | None = None,
        lr_scheduler: TLearningRateScheduler | None = None,
    ) -> None:
        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            lr_scheduler=lr_scheduler,
        )

        self.actor = actor
        self.optim = optim
        self.n_top = n_top

    def process_fn(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> BatchWithReturnsAndDeltasProtocol:
        """Filter and add plus/minus returns, deltas, and return std to batch."""
        pop_size = len(batch.returns)
        returns = torch.Tensor(batch.returns)
        deltas = batch.deltas[: pop_size // 2, :]
        plus_returns = returns[: pop_size // 2, :]
        minus_returns = returns[pop_size // 2 :, :]
        if self.n_top:
            top_returns, _ = torch.max(torch.cat([plus_returns, minus_returns], dim=1), dim=1)
            top_indices = top_returns.argsort(dim=0, descending=True)[: self.n_top]
            plus_returns = plus_returns[top_indices]
            minus_returns = minus_returns[top_indices]
            deltas = deltas[top_indices]
            batch.return_std = torch.cat([plus_returns, minus_returns]).std()
        else:
            batch.return_std = returns.std()

        batch.plus_returns = plus_returns
        batch.minus_returns = minus_returns
        batch.deltas = deltas

        return cast(BatchWithReturnsAndDeltasProtocol, batch)

    def set_params_from_vector(self, params: torch.Tensor) -> None:
        torch.nn.utils.vector_to_parameters(params.flatten(), self.parameters())

    def forward(
        self,
        batch: ObsBatchProtocol,
        state: dict | BatchProtocol | np.ndarray | None = None,
        **kwargs: Any,
    ) -> ActBatchProtocol:
        """Compute action over the given batch data."""
        act = self.actor(batch.obs)
        result = Batch(act=act)
        return cast(ActBatchProtocol, result)

    def learn(self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any) -> ARSTrainingStats:
        self.optim.zero_grad()

        # Compute gradient of the objective function and update the parameters as in line 7 of the ARS algorithm
        grad = -(batch.plus_returns - batch.minus_returns).T @ batch.deltas
        vector_to_gradient(grad / (self.n_top * batch.return_std + 1e-6), self.parameters())
        self.optim.step()

        return ARSTrainingStats(loss=SequenceSummaryStats.from_sequence(batch.plus_returns.numpy()))
