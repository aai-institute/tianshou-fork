"""The rliable-evaluation module provides a high-level interface to evaluate the results of an experiment with multiple runs
on different seeds using the rliable library. The API is experimental and subject to change!.
"""

import os
from dataclasses import dataclass, fields
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sst
from rliable import library as rly
from rliable import plot_utils

from tianshou.highlevel.experiment import Experiment
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.utils import logging
from tianshou.utils.logger.base import DataScope

log = logging.getLogger(__name__)


@dataclass
class LoggedSummaryData:
    mean: np.ndarray
    std: np.ndarray
    max: np.ndarray
    min: np.ndarray


@dataclass
class LoggedCollectStats:
    env_step: np.ndarray | None = None
    n_collected_episodes: np.ndarray | None = None
    n_collected_steps: np.ndarray | None = None
    collect_time: np.ndarray | None = None
    collect_speed: np.ndarray | None = None
    returns_stat: LoggedSummaryData | None = None
    lens_stat: LoggedSummaryData | None = None

    @classmethod
    def from_data_dict(cls, data: dict) -> "LoggedCollectStats":
        """Create a LoggedCollectStats object from a dictionary.

        Converts SequenceSummaryStats from dict format to dataclass format and ignores fields that are not present.
        """
        field_names = [f.name for f in fields(cls)]
        for k, v in data.items():
            if k not in field_names:
                data.pop(k)
            if isinstance(v, dict):
                data[k] = LoggedSummaryData(**v)
        return cls(**data)


@dataclass
class RLiableExperimentResult:
    """The result of an experiment that can be used with the rliable library."""

    exp_dir: str
    """The base directory where each sub-directory contains the results of one experiment run."""

    test_episode_returns_RE: np.ndarray
    """The test episodes for each run of the experiment where each row corresponds to one run."""

    train_episode_returns_RE: np.ndarray
    """The training episodes for each run of the experiment where each row corresponds to one run."""

    env_steps_E: np.ndarray
    """The number of environment steps at which the test episodes were evaluated."""

    env_steps_train_E: np.ndarray
    """The number of environment steps at which the training episodes were evaluated."""

    @classmethod
    def load_from_disk(cls, exp_dir: str) -> "RLiableExperimentResult":
        """Load the experiment result from disk.

        :param exp_dir: The directory from where the experiment results are restored.
        """
        test_episode_returns = []
        train_episode_returns = []
        env_step_at_test = None
        env_step_at_train = None

        # TODO: env_step_at_test should not be defined in a loop and overwritten at each iteration
        #  just for retrieving them. We might need a cleaner directory structure.
        for entry in os.scandir(exp_dir):
            if entry.name.startswith(".") or not entry.is_dir():
                continue

            try:
                logger_factory = Experiment.from_directory(entry.path).logger_factory
            # Usually this means from low-level API
            except FileNotFoundError:
                log.info(
                    f"Could not find persisted experiment in {entry.path}, using default logger.",
                )
                logger_factory = LoggerFactoryDefault()

            logger = logger_factory.create_logger(
                entry.path,
                entry.name,
                None,
            )
            data = logger.restore_logged_data(entry.path)

            if DataScope.TEST.value not in data or not data[DataScope.TEST.value]:
                continue
            restored_test_data = data[DataScope.TEST.value]
            restored_train_data = data[DataScope.TRAIN.value]
            for restored_data, scope in zip(
                [restored_test_data, restored_train_data],
                [DataScope.TEST, DataScope.TRAIN],
                strict=True,
            ):
                if not isinstance(restored_data, dict):
                    raise RuntimeError(
                        f"Expected entry with key {scope.value} data to be a dictionary, "
                        f"but got {restored_data=}.",
                    )
            test_data = LoggedCollectStats.from_data_dict(restored_test_data)
            train_data = LoggedCollectStats.from_data_dict(restored_train_data)

            if test_data.returns_stat is not None:
                test_episode_returns.append(test_data.returns_stat.mean)
                env_step_at_test = test_data.env_step

            if train_data.returns_stat is not None:
                train_episode_returns.append(train_data.returns_stat.mean)
                env_step_at_train = train_data.env_step

        test_data_found = True
        train_data_found = True
        if not test_episode_returns or env_step_at_test is None:
            log.warning(f"No test experiment data found in {exp_dir}.")
            test_data_found = False
        if not train_episode_returns or env_step_at_train is None:
            log.warning(f"No train experiment data found in {exp_dir}.")
            train_data_found = False

        if not test_data_found and not train_data_found:
            raise RuntimeError(f"No test or train data found in {exp_dir}.")

        return cls(
            test_episode_returns_RE=np.array(test_episode_returns),
            env_steps_E=np.array(env_step_at_test),
            exp_dir=exp_dir,
            train_episode_returns_RE=np.array(train_episode_returns),
            env_steps_train_E=np.array(env_step_at_train),
        )

    def _get_rliable_data(
        self,
        algo_name: str | None = None,
        score_thresholds: np.ndarray | None = None,
        scope: DataScope | Literal["train", "test"] = DataScope.TEST,
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        """Return the data in the format expected by the rliable library.

        :param algo_name: The name of the algorithm to be shown in the figure legend. If None, the name of the algorithm
            is set to the experiment dir.
        :param score_thresholds: The score thresholds for the performance profile. If None, the thresholds are inferred
            from the minimum and maximum test episode returns.

        :return: A tuple score_dict, env_steps, and score_thresholds.
        """
        if isinstance(scope, DataScope):
            scope = scope.value
        if scope == DataScope.TEST.value:
            env_steps, returns = self.env_steps_E, self.test_episode_returns_RE
        elif scope == DataScope.TRAIN.value:
            env_steps, returns = self.env_steps_train_E, self.train_episode_returns_RE
        else:
            raise ValueError(f"Invalid scope {scope}, should be either 'TEST' or 'TRAIN'.")
        if score_thresholds is None:
            score_thresholds = np.linspace(
                np.min(returns),
                np.max(returns),
                101,
            )

        if algo_name is None:
            algo_name = os.path.basename(self.exp_dir)

        score_dict = {algo_name: returns}

        return score_dict, env_steps, score_thresholds

    def eval_results(
        self,
        algo_name: str | None = None,
        score_thresholds: np.ndarray | None = None,
        save_plots: bool = False,
        show_plots: bool = True,
        scope: DataScope | Literal["train", "test"] = DataScope.TEST,
    ) -> tuple[plt.Figure, plt.Axes, plt.Figure, plt.Axes]:
        """Evaluate the results of an experiment and create a sample efficiency curve and a performance profile.

        :param algo_name: The name of the algorithm to be shown in the figure legend. If None, the name of the algorithm
            is set to the experiment dir.
        :param score_thresholds: The score thresholds for the performance profile. If None, the thresholds are inferred
            from the minimum and maximum test episode returns.
        :param save_plots: If True, the figures are saved to the experiment directory.
        :param show_plots: If True, the figures are shown.
        :param scope: The scope of the evaluation, either 'TEST' or 'TRAIN'.

        :return: The created figures and axes.
        """
        score_dict, env_steps, score_thresholds = self._get_rliable_data(
            algo_name,
            score_thresholds,
            scope,
        )

        iqm = lambda scores: sst.trim_mean(scores, proportiontocut=0.25, axis=0)
        iqm_scores, iqm_cis = rly.get_interval_estimates(score_dict, iqm)

        # Plot IQM sample efficiency curve
        fig_iqm, ax_iqm = plt.subplots(ncols=1, figsize=(7, 5), constrained_layout=True)
        plot_utils.plot_sample_efficiency_curve(
            env_steps,
            iqm_scores,
            iqm_cis,
            algorithms=None,
            xlabel="env step",
            ylabel="IQM episode return",
            ax=ax_iqm,
        )
        if show_plots:
            plt.show(block=False)

        if save_plots:
            iqm_sample_efficiency_curve_path = os.path.abspath(
                os.path.join(
                    self.exp_dir,
                    "iqm_sample_efficiency_curve.png",
                ),
            )
            log.info(f"Saving iqm sample efficiency curve to {iqm_sample_efficiency_curve_path}.")
            fig_iqm.savefig(iqm_sample_efficiency_curve_path)

        final_score_dict = {algo: returns[:, [-1]] for algo, returns in score_dict.items()}
        score_distributions, score_distributions_cis = rly.create_performance_profile(
            final_score_dict,
            score_thresholds,
        )

        # Plot score distributions
        fig_profile, ax_profile = plt.subplots(ncols=1, figsize=(7, 5), constrained_layout=True)
        plot_utils.plot_performance_profiles(
            score_distributions,
            score_thresholds,
            performance_profile_cis=score_distributions_cis,
            xlabel=r"Episode return $(\tau)$",
            ax=ax_profile,
        )

        if save_plots:
            profile_curve_path = os.path.abspath(
                os.path.join(self.exp_dir, "performance_profile.png"),
            )
            log.info(f"Saving performance profile curve to {profile_curve_path}.")
            fig_profile.savefig(profile_curve_path)
        if show_plots:
            plt.show(block=False)

        return fig_iqm, ax_iqm, fig_profile, ax_profile


def load_and_eval_experiments(
    log_dir: str,
    show_plots: bool = True,
    save_plots: bool = True,
    scope: DataScope | Literal["train", "test", "both"] = DataScope.TEST,
) -> RLiableExperimentResult:
    """Evaluate the experiments in the given log directory using the rliable API and return the loaded results object.

    If neither show_plots nor save_plots is set to True, this is equivalent to just loading the results from disk.

    :param log_dir: The directory containing the experiment results.
    :param show_plots: If True, the plots are shown.
    :param save_plots: If True, the plots are saved to the log directory.
    :param scope: The scope of the evaluation, either 'TEST' or 'TRAIN'.
    """
    rliable_result = RLiableExperimentResult.load_from_disk(log_dir)
    if scope == "both":
        for scope in [DataScope.TEST, DataScope.TRAIN]:
            rliable_result.eval_results(show_plots=True, save_plots=True, scope=scope)
    else:
        rliable_result.eval_results(show_plots=show_plots, save_plots=save_plots, scope=scope)
    return rliable_result
