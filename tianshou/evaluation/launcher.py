"""Provides a basic interface for launching experiments. The API is experimental and subject to change!."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from copy import copy
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Literal

from joblib import Parallel, delayed

from tianshou.data import InfoStats
from tianshou.highlevel.experiment import Experiment

log = logging.getLogger(__name__)


@dataclass
class JoblibConfig:
    n_jobs: int = -1
    """The maximum number of concurrently running jobs. If -1, all CPUs are used."""
    backend: Literal["loky", "multiprocessing", "threading"] | None = "loky"
    """Allows to hard-code backend, otherwise inferred based on prefer and require."""
    verbose: int = 10
    """If greater than zero, prints progress messages."""


class ExpLauncher(ABC):
    def __init__(
        self,
        experiment_runner: Callable[
            [Experiment],
            InfoStats | None,
        ] = lambda exp: exp.run().trainer_result,
    ):
        self.experiment_runner = experiment_runner

    @abstractmethod
    def launch(self, experiments: Sequence[Experiment]) -> None:
        pass


class SequentialExpLauncher(ExpLauncher):
    def launch(self, experiments: Sequence[Experiment]) -> None:
        for exp in experiments:
            self.experiment_runner(exp)


class JoblibExpLauncher(ExpLauncher):
    def __init__(
        self,
        joblib_cfg: JoblibConfig | None = None,
        experiment_runner: Callable[
            [Experiment],
            InfoStats | None,
        ] = lambda exp: exp.run().trainer_result,
    ) -> None:
        super().__init__(experiment_runner=experiment_runner)
        self.joblib_cfg = copy(joblib_cfg) if joblib_cfg is not None else JoblibConfig()
        # Joblib's backend is hard-coded to loky since the threading backend produces different results
        if self.joblib_cfg.backend != "loky":
            log.warning(
                f"Ignoring the user provided joblib backend {self.joblib_cfg.backend} and using loky instead. "
                f"The current implementation requires loky to work and will be relaxed soon",
            )
            self.joblib_cfg.backend = "loky"

    def launch(self, experiments: Sequence[Experiment]) -> None:
        Parallel(**asdict(self.joblib_cfg))(delayed(self._safe_execute)(exp) for exp in experiments)

    def _safe_execute(self, exp: Experiment) -> None:
        try:
            self.experiment_runner(exp)
        except BaseException as e:
            log.error(e)


class RegisteredExpLauncher(Enum):
    joblib = "joblib"
    sequential = "sequential"

    def create_launcher(self) -> ExpLauncher:
        match self:
            case RegisteredExpLauncher.joblib:
                return JoblibExpLauncher()
            case RegisteredExpLauncher.sequential:
                return SequentialExpLauncher()
            case _:
                raise NotImplementedError(
                    f"Launcher {self} is not yet implemented.",
                )
