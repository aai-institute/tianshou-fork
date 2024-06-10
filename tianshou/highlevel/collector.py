from abc import ABC, abstractmethod
from dataclasses import dataclass

from tianshou.data.collector import StepHookActionDistribution
from tianshou.utils.string import ToStringMixin


# class OnStepCallback(ToStringMixin, ABC):
#     """Callback which is called at the beginning of each epoch, i.e. prior to the data collection phase
#     of each epoch.
#     """
#
#     @abstractmethod
#     def callback(self, epoch: int, env_step: int, context: TrainingContext) -> None:
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

    on_step_hook: list[callable] = None
    """Callbacks to be called at the beginning of each step."""
    on_collect: list[callable] = None
    """Callbacks to be called after collecting a transition."""
    on_collect_end: list[callable] = None
    """Callbacks to be called at the end of the collection phase of an epoch."""
    on_collect_start: list[callable] = None
    """Callbacks to be called at the beginning of the collection phase of an epoch."""
    on_step_end: list[callable] = None
    """Callbacks to be called at the end of each step."""
    on_step_start: list[callable] = None
    """Callbacks to be called at the beginning of each step."""


# class OnStepCallbackGetActionDistribution(OnStepCallback):
#     def __init__(self):
#         self.hook = StepHookActionDistribution()
#
#     def callback(self, epoch: int, env_step: int, context: TrainingContext) -> None:
