import os

import numpy as np
import scipy.stats as sst
import seaborn as sns
from matplotlib import pyplot as plt
from rliable import library as rly, metrics
from rliable import plot_utils

from experiments.algo_eval.ppo_eval_protocol import SeedConfiguration
from experiments.algo_eval.utils import shortener
from tianshou.utils.logger.pandas_logger import PandasLogger


def get_exp_dir(seed_config):
    if seed_config.train_env_seeds:
        for policy_seed in seed_config.policy_seeds:
            for train_seed in seed_config.train_env_seeds:
                full_name = f"seed={policy_seed},train_seed={train_seed}"
                experiment_name = shortener(full_name, 3)
                yield experiment_name
    else:
        for policy_seed in seed_config.policy_seeds:
            full_name = f"seed={policy_seed}"
            experiment_name = shortener(full_name, 3)
            yield experiment_name


def load_exp(log_dir_base, seed_config):
    test_episode_returns = []
    exp_dirs = get_exp_dir(seed_config)

    for experiment_name in exp_dirs:
        log_dir = os.path.join(log_dir_base, experiment_name)
        print(log_dir)

        # experiment = Experiment.from_directory(log_dir)
        logger = PandasLogger(log_dir)
        logger.restore_data()

        train_data = logger.data['train']
        test_data = logger.data['test']

        test_episode_returns.append([d['returns_stat']['mean'] for d in test_data])

    test_episode_returns = np.stack(test_episode_returns)
    env_step_test = [d['env_step'] for d in test_data]

    return env_step_test, test_episode_returns


def eval_exp(score_dict, env_step_test, algorithms, thresholds):

    iqm = lambda scores: sst.trim_mean(scores, proportiontocut=0.25, axis=0)
    iqm_scores, iqm_cis = rly.get_interval_estimates(
        score_dict, iqm, reps=50000)

    plot_utils.plot_sample_efficiency_curve(
        env_step_test, iqm_scores, iqm_cis, algorithms=algorithms,
        xlabel=r'Number of env steps',
        ylabel='IQM Epsiode Return',)
    plt.legend()

    final_score_dict = {k: v[:, [-1]] for k, v in score_dict.items()}
    score_distributions, score_distributions_cis = rly.create_performance_profile(
        final_score_dict, thresholds)
    # Plot score distributions
    fig, ax = plt.subplots(ncols=1, figsize=(7, 5))
    plot_utils.plot_performance_profiles(
        score_distributions, thresholds,
        performance_profile_cis=score_distributions_cis,
        colors=dict(zip(algorithms, sns.color_palette('colorblind'))),
        xlabel=r'Episode return $(\tau)$',
        ax=ax)

    if len(algorithms) > 1:
        aggregate_func = lambda x: np.array([
            metrics.aggregate_median(x),
            metrics.aggregate_iqm(x),
            metrics.aggregate_mean(x),
            # metrics.aggregate_optimality_gap(x),
        ])

        aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
            final_score_dict, aggregate_func, reps=50000)

        fig, axes = plot_utils.plot_interval_estimates(
            aggregate_scores, aggregate_score_cis,
            metric_names=['Median', 'IQM', 'Mean'],
            algorithms=algorithms, xlabel='Episode return')

    print()


if __name__ == "__main__":
    # custom path to the experiment directory
    exp_dir = "/private/tmp/log/2024-02-20/13-00-20/ppo/HalfCheetah-v4/"
    seed_config = SeedConfiguration(
        policy_seeds=list(range(10)),
        train_env_seeds=[0, 10, 100, 1000],
        test_env_seeds=[1337],
    )
    ppo_results = load_exp(exp_dir, seed_config)

    exp_dir = "/private/tmp/log/2024-02-24/15-35-26/ppo/HalfCheetah-v4/"
    seed_config = SeedConfiguration(
        policy_seeds=list(range(40)),
        train_env_seeds=[],
        test_env_seeds=[1337],
    )
    ppo_results_global_seed = load_exp(exp_dir, seed_config)

    algorithms = ['PPO', 'PPO_global_seed', 'PPO_global_10']
    score_dict = {'PPO': ppo_results[1], 'PPO_global_seed': ppo_results_global_seed[1],
                  'PPO_global_10': ppo_results_global_seed[1][:10]}
    half_cheetah_thresholds = np.linspace(0.0, 8000.0, 81)

    eval_exp(score_dict, ppo_results[0], algorithms, half_cheetah_thresholds)
