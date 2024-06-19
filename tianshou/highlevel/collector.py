from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from tianshou.data import CollectStats, SequenceSummaryStats, ReplayBuffer
from tianshou.data.callbacks import EpisodeHookAddValuesFromContinuousCritic, StepHook, RolloutHook, CollectCallbacks, \
    EpisodeRolloutHookMCReturn
# from tianshou.data.types import CollectActionComputationBatchProtocol, RolloutBatchProtocol
# from tianshou.data.collector import StepHookActionDistribution
from tianshou.utils.string import ToStringMixin


class CollectorContext:
    def __init__(self, policy):
        self.policy = policy


class CollectorCallback(ToStringMixin, ABC):
    """Callback which is called after an episode has been completed."""

    @abstractmethod
    def get_collector_callback(self, context: CollectorContext) -> StepHook | RolloutHook:
        pass


class CustomCollectStatsAndCallback(ABC):
    def _get_on_step_callback(self, context: CollectorContext) -> Optional[StepHook]:
        pass

    def _get_on_episode_done_callback(self, context: CollectorContext) -> Optional[RolloutHook]:
        pass

    def _get_on_collect_end_callback(self, context: CollectorContext) -> Optional[RolloutHook]:
        pass

    def get_collect_callbacks(self, context: CollectorContext) -> CollectCallbacks:
        collect_callbacks = CollectCallbacks(
            collect_end_callback=self._get_on_collect_end_callback(context),
            episode_done_callback=self._get_on_episode_done_callback(context),
            step_callback=self._get_on_step_callback(context),
        )
        return collect_callbacks

    @abstractmethod
    def get_collect_stats(self) -> type[CollectStats]:
        pass


class CollectStatsWithMCReturn(CustomCollectStatsAndCallback):
    def __init__(self, gamma):
        self.gamma = gamma

    def _get_on_episode_done_callback(self, context: CollectorContext) -> Optional[RolloutHook]:
        return EpisodeRolloutHookMCReturn(gamma=self.gamma)

    def get_collect_stats(self) -> type[CollectStats]:
        @dataclass
        class CustomCollectStats(CollectStats):
            mc_returns: SequenceSummaryStats = None

            def add_stats_from_buffer(self, buffer: ReplayBuffer) -> None:
                all_buffer_data = buffer.sample(0)[0]
                self.mc_returns = SequenceSummaryStats.from_sequence(all_buffer_data.mc_return)
        return CustomCollectStats


class CollectStatsWithGAEReturn(CustomCollectStatsAndCallback):
    def __init__(self, batchsize, gamma, gae_lambda):
        self.batchsize = batchsize
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def _get_on_collect_end_callback(self, context: CollectorContext) -> callable:
        return EpisodeHookAddValuesFromContinuousCritic(context.policy.critic, batchsize=64, gamma=self.gamma,
                                                        gae_lambda=self.gae_lambda,
                                                        ret_rms=context.policy.ret_rms)

    def get_collect_stats(self) -> type[CollectStats]:
        @dataclass
        class CustomCollectStats(CollectStats):
            advantage: SequenceSummaryStats = None
            normalized_returns: SequenceSummaryStats = None

            def add_stats_from_buffer(self, buffer: ReplayBuffer) -> None:
                all_buffer_data = buffer.sample(0)[0]
                self.advantage = SequenceSummaryStats.from_sequence(all_buffer_data.advantages)
        return CustomCollectStats


# class OnStepCallback(ToStringMixin, ABC):
#     """Callback which is called at the beginning of each epoch, i.e. prior to the data collection phase
#     of each epoch.
#     """
#
#     @abstractmethod
#     def callback(self,
#                  rollout_batch: RolloutBatchProtocol,
#                  action_computation_batch: CollectActionComputationBatchProtocol) -> None:
#         pass
#
#     def get_trainer_fn(self, context: TrainingContext) -> Callable[[int, int], None]:
#         def fn(epoch: int, env_step: int) -> None:
#             return self.callback(epoch, env_step, context)
#
#         return fn


@dataclass
class CollectorCallbacks:
    """Callbacks to be called at various points during the collection phase of an epoch."""
    collect_stat_and_callback: CustomCollectStatsAndCallback | None = Non


# class OnStepCallbackGetActionDistribution(OnStepCallback):
#     def __init__(self):
#         self.hook = StepHookActionDistribution()
#
#     def callback(self, epoch: int, env_step: int, context: TrainingContext) -> None:
