from abc import ABC, abstractmethod
from typing import Sequence

from tianshou.highlevel.experiment import Experiment


class SeededExperimentFactory(ABC):
    @abstractmethod
    def create_experiment(self, seed: int) -> Experiment:
        pass
