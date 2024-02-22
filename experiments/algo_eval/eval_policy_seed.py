import os

import numpy as np
import scipy.stats as sst
import seaborn as sns
from matplotlib import pyplot as plt
from rliable import library as rly
from rliable import plot_utils

from experiments.algo_eval.ppo_eval_protocol import SeedConfiguration
from experiments.algo_eval.utils import shortener
from tianshou.utils.logger.pandas_logger import PandasLogger

# custom path to the experiment directory
exp_dir = "/private/tmp/log/2024-02-20/13-00-20/ppo/HalfCheetah-v4/"


def eval_exp():
    seed_config = SeedConfiguration(
        policy_seeds=list(range(10)),
        train_env_seeds=[0, 10, 100, 1000],
        test_env_seeds=[1337],
    )

    test_episode_returns = []

    for policy_seed in seed_config.policy_seeds:
        for train_seed in seed_config.train_env_seeds:
            full_name = f"seed={policy_seed},train_seed={train_seed}"
            experiment_name = shortener(full_name, 3)
            log_dir = os.path.join(exp_dir, experiment_name)
            print(log_dir)

            # experiment = Experiment.from_directory(log_dir)
            logger = PandasLogger(log_dir)
            logger.restore_data()

            train_data = logger.data['train']
            test_data = logger.data['test']

            test_episode_returns.append([d['returns_stat']['mean'] for d in test_data])

    test_episode_returns = np.stack(test_episode_returns)
    env_step_test = [d['env_step'] for d in test_data]

    algorithms = ['PPO']
    ppo_score_dict = {'PPO': test_episode_returns}
    iqm = lambda scores: sst.trim_mean(scores, proportiontocut=0.25, axis=0)
    iqm_scores, iqm_cis = rly.get_interval_estimates(
        ppo_score_dict, iqm, reps=50000)

    plot_utils.plot_sample_efficiency_curve(
        env_step_test, iqm_scores, iqm_cis, algorithms=algorithms,
        xlabel=r'Number of env steps',
        ylabel='IQM Epsiode Return',)

    half_cheetah_thresholds = np.linspace(0.0, 8000.0, 81)
    ppo_final_score_dict = {'PPO': test_episode_returns[:, [-1]]}
    score_distributions, score_distributions_cis = rly.create_performance_profile(
        ppo_final_score_dict, half_cheetah_thresholds)
    # Plot score distributions
    fig, ax = plt.subplots(ncols=1, figsize=(7, 5))
    plot_utils.plot_performance_profiles(
        score_distributions, half_cheetah_thresholds,
        performance_profile_cis=score_distributions_cis,
        colors=dict(zip(algorithms, sns.color_palette('colorblind'))),
        xlabel=r'Human Normalized Score $(\tau)$',
        ax=ax)

    print()


if __name__ == "__main__":
    eval_exp()
