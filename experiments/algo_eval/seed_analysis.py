from dataclasses import dataclass, field, asdict
from typing import Sequence, Literal

import numpy as np
from joblib import Parallel, delayed

from experiments.algo_eval.utils import shortener
from experiments.exp_builders import SeededExperimentFactory
from tianshou.highlevel.experiment import Experiment


@dataclass
class JoblibConfig:
    n_jobs: int = 1
    backend: Literal["loky", "multiprocessing", "threading"] | None = None
    verbose: int = 10
    return_as: Literal["list", "generator"] = "list"
    prefer: Literal["processes", "threads"] | None = None
    require: Literal["sharedmem"] | None = None


@dataclass
class ExperimentResults:
    algorithms: list[str]
    score_dict: dict[str, np.ndarray]  # (n_runs x n_epochs)
    env_steps: np.ndarray  # (n_epochs)
    score_thresholds: np.ndarray

    def load_from_disk(self):
        pass


@dataclass
class SeedConfiguration:
    seeds: Sequence[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])


class SeedVariabilityAnalysis:
    def __init__(self,
                 seed_config: SeedConfiguration,
                 seeded_experiment_factory: SeededExperimentFactory):
        self.seed_config = seed_config
        self.seeded_experiment_factory = seeded_experiment_factory

    def build_experiments(self):
        for policy_seed in self.seed_config.seeds:
            experiment = self.seeded_experiment_factory.create_experiment(policy_seed)
            full_name = f"seed={policy_seed}"
            experiment_name = shortener(full_name, 3)
            yield experiment_name, experiment

    def run_sequential(self):
        results = []

        for experiment_name, experiment in self.build_experiments():
            results.append(experiment.run(experiment_name))

    def run_joblib_local(self, joblib_config: JoblibConfig):
        results = Parallel(**asdict(joblib_config))(
            delayed(self.execute_task)(exp, exp_name) for exp_name, exp in self.build_experiments())
        return results

    def execute_task(self, exp: Experiment, name: str):
        try:
            exp.run(name)
        except Exception as e:
            print(e)

    @staticmethod
    def eval_results(results: ExperimentResults):
        import matplotlib.pyplot as plt
        import scipy.stats as sst
        import seaborn as sns
        from rliable import library as rly
        from rliable import plot_utils

        iqm = lambda scores: sst.trim_mean(scores, proportiontocut=0.25, axis=0)
        iqm_scores, iqm_cis = rly.get_interval_estimates(
            results.score_dict, iqm, reps=50000)

        # Plot IQM sample efficiency curve
        fig, ax = plt.subplots(ncols=1, figsize=(7, 5))
        plot_utils.plot_sample_efficiency_curve(
            results.env_steps, iqm_scores, iqm_cis, algorithms=results.algorithms,
            xlabel=r'Number of env steps',
            ylabel='IQM episode return',
            ax=ax)
        plt.savefig('iqm_sample_efficiency_curve.png')

        final_score_dict = {algo: returns[:, [-1]] for algo, returns in results.score_dict.items()}
        score_distributions, score_distributions_cis = rly.create_performance_profile(
            final_score_dict, results.score_thresholds)

        # Plot score distributions
        fig, ax = plt.subplots(ncols=1, figsize=(7, 5))
        plot_utils.plot_performance_profiles(
            score_distributions, results.score_thresholds,
            performance_profile_cis=score_distributions_cis,
            colors=dict(zip(results.algorithms, sns.color_palette('colorblind'))),
            xlabel=r'Episode return $(\tau)$',
            ax=ax)
        plt.savefig('performance_profile.png')
