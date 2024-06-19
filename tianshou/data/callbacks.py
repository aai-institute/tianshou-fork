from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
import torch

from tianshou.data import to_numpy, ReplayBuffer
from tianshou.data.types import RolloutBatchProtocol, CollectActionComputationBatchProtocol, DistBatchProtocol
from tianshou.policy.base import episode_mc_return, _gae_return
from tianshou.utils.net.continuous import Critic


class RolloutHookProtocol(Protocol):
    """A protocol for rollout hooks."""

    def __call__(self, rollout_batch: RolloutBatchProtocol) -> dict[str, np.ndarray]:
        """The function to call when the hook is executed."""
        ...


class RolloutHook(RolloutHookProtocol, ABC):
    @abstractmethod
    def __call__(self, rollout_batch: RolloutBatchProtocol) -> dict[str, np.ndarray]:
        ...


class CombinedRolloutHook(RolloutHook):
    def __init__(self, *rollout_hooks: RolloutHookProtocol):
        self.rollout_hooks = rollout_hooks

    def __call__(self, rollout_batch: RolloutBatchProtocol) -> dict[str, np.ndarray]:
        result = {}
        for rollout_hook in self.rollout_hooks:
            new_entries_dict = rollout_hook(rollout_batch)
            if duplicated_entries := set(new_entries_dict).difference(result):
                raise RuntimeError(
                    f"Combined rollout hook lead to previously "
                    f"computed entries that would be overwritten: {duplicated_entries=}. "
                    f"Consider combining only hooks which will deliver non-overlapping entries to solve this.",
                )
            result.update(new_entries_dict)
        return result


class StepHookProtocol(Protocol):
    """A protocol for step hooks."""

    def __call__(
        self,
        rollout_batch: RolloutBatchProtocol,
        action_batch: CollectActionComputationBatchProtocol,
    ) -> dict[str, np.ndarray]:
        """The function to call when the hook is executed."""
        ...


class StepHook(StepHookProtocol, ABC):
    @abstractmethod
    def __call__(
        self,
        rollout_batch: RolloutBatchProtocol,
        action_batch: CollectActionComputationBatchProtocol,
    ) -> dict[str, np.ndarray]:
        ...


class CombinedStepHook(StepHook):
    def __init__(self, *step_hooks: StepHookProtocol):
        self.step_hooks = step_hooks

    def __call__(
        self,
        rollout_batch: RolloutBatchProtocol,
        action_batch: CollectActionComputationBatchProtocol,
    ) -> dict[str, np.ndarray]:
        result = {}
        for step_hook in self.step_hooks:
            new_entries_dict = step_hook(rollout_batch, action_batch)
            if duplicated_entries := set(new_entries_dict).difference(result):
                raise RuntimeError(
                    f"Combined step hook lead to previously "
                    f"computed entries that would be overwritten: {duplicated_entries=}. "
                    f"Consider combining only hooks which will deliver non-overlapping entries to solve this.",
                )
            result.update(new_entries_dict)
        return result


class StepHookActionDistribution(StepHook):
    ACTION_DIST_KEY = "action_dist"

    def __call__(
        self,
        rollout_batch: RolloutBatchProtocol,
        action_batch: CollectActionComputationBatchProtocol,
    ) -> dict[str, np.ndarray]:
        return {self.ACTION_DIST_KEY: action_batch.dist}


class EpisodeRolloutHook(RolloutHook, ABC):
    """Marker interface, hooks that operate on a rollout of a single episode should inherit from this."""


class EpisodeRolloutHookMCReturn(EpisodeRolloutHook):
    MC_RETURN_KEY = "mc_return"
    FULL_EPISODE_MC_RETURN_KEY = "full_episode_mc_return"

    def __init__(self, gamma: float):
        self.gamma = gamma

    def __call__(self, rollout_batch: RolloutBatchProtocol) -> dict[str, np.ndarray]:
        mc_returns = episode_mc_return(rollout_batch.rew, self.gamma)
        full_episode_mc_return = mc_returns[0]
        return {
            self.MC_RETURN_KEY: mc_returns,
            self.FULL_EPISODE_MC_RETURN_KEY: np.full_like(
                rollout_batch.rew,
                full_episode_mc_return,
            ),
        }


class EpisodeHookAddValuesFromContinuousCritic(EpisodeRolloutHook):
    RETURNS_KEY = "returns"
    ADVANTAGES_KEY = "advantages"

    def __init__(self, critic: Critic, batchsize: int, gamma: float, gae_lambda: float = 0.95, ret_rms=None):
        self.critic = critic
        self.batchsize = batchsize
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.rew_norm = True
        self.ret_rms = ret_rms
        self._eps = 1e-8

    # TODO: WIP
    def __call__(self, buffer: ReplayBuffer) -> dict[str, np.ndarray]:
        batch, indices = buffer.sample(sample_size)
        v_s, v_s_ = [], []
        with torch.no_grad():
            for minibatch in rollout_batch.split(self.batchsize, shuffle=False, merge_last=True):
                v_s.append(self.critic(minibatch.obs))
                v_s_.append(self.critic(minibatch.obs_next))
        v_s = torch.cat(v_s, dim=0).flatten().cpu().numpy()  # old value
        v_s = to_numpy(v_s.flatten())

        v_s_ = torch.cat(v_s_, dim=0).flatten().cpu().numpy()
        if self.rew_norm:  # unnormalize v_s & v_s_
            v_s = v_s * np.sqrt(self.ret_rms.var + self._eps)
            v_s_ = v_s_ * np.sqrt(self.ret_rms.var + self._eps)

        v_s_ = to_numpy(v_s_.flatten())
        v_s_ = v_s_ * ~rollout_batch.terminated

        rew = rollout_batch.rew
        end_flag = np.logical_or(rollout_batch.terminated, rollout_batch.truncated)
        end_flag[np.isin(indices, buffer.unfinished_index())] = True
        advantage = _gae_return(v_s, v_s_, rew, end_flag, self.gamma, self.gae_lambda)
        unnormalized_returns = advantage + v_s
        if self.rew_norm:
            returns = unnormalized_returns / np.sqrt(self.ret_rms.var + self._eps)
            self.ret_rms.update(unnormalized_returns)
        else:
            returns = unnormalized_returns
        # returns = to_torch_as(batch.returns, batch.v_s)
        # adv = to_torch_as(advantages, batch.v_s)
        return {self.RETURNS_KEY: returns, self.ADVANTAGES_KEY: advantage}


class HookFilterEpisodeRolloutMCReturn(EpisodeRolloutHook):
    BATCH_KEY = "filter_optimality"

    def __init__(self, threshold: float):
        self.optimality_return_threshold = threshold

    def __call__(self, rollout_batch: RolloutBatchProtocol) -> dict[str, np.ndarray]:
        return {
            self.BATCH_KEY: np.ones_like(rollout_batch.rew)
            if sum(rollout_batch.rew) >= self.optimality_return_threshold
            else np.zeros_like(rollout_batch.rew),
        }


@dataclass
class CollectCallbacks:
    """Container for callbacks used during collection."""

    episode_done_callback: EpisodeRolloutHook | Callable[[RolloutBatchProtocol], dict[str, np.ndarray]] | None = None
    step_callback: StepHook | Callable[[RolloutBatchProtocol, DistBatchProtocol], dict[str, np.ndarray]] | None = None
    collect_end_callback: Callable[[RolloutBatchProtocol], None] | None = None

    def run_on_episode_done(
        self,
        episode_batch: RolloutBatchProtocol,
    ) -> dict[str, np.ndarray] | None:
        """Executes the `on_episode_done_hook` that was passed on init.

        The raison d'être of this method is to allow for a cleaner implementation
        of the hook for users who want to subclass the Collector. These users can
        then override this method and also override the init to no longer accept
        the `on_episode_done_hook` provider.
        """
        if self.episode_done_callback is not None:
            return self.episode_done_callback(episode_batch)
        return None

    def run_on_step(
        self,
        rollout_batch: RolloutBatchProtocol,
        action_batch: CollectActionComputationBatchProtocol,
    ) -> dict[str, np.ndarray] | None:
        """Executes the `on_step_hook` that was passed on init."""
        if self.step_callback is not None:
            step_hook_additions = self.step_callback(rollout_batch, action_batch)
            if step_hook_additions is not None:
                for key, array in step_hook_additions.items():
                    rollout_batch.info.set_array_at_key(
                        array,
                        key,
                    )
        return None

    def run_on_collect_end(self, rollout_batch: RolloutBatchProtocol) -> None:
        if self.collect_end_callback is not None:
            self.collect_end_callback(rollout_batch)
