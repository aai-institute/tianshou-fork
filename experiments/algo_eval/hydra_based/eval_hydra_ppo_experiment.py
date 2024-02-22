import os
from abc import ABC, abstractmethod

import hydra
import numpy as np
import scipy.stats as sst
from matplotlib import pyplot as plt
from omegaconf import DictConfig

from tianshou.highlevel.experiment import Experiment
from tianshou.utils.logger.pandas_logger import PandasLogger

# custom path to the experiment directory, couldn't figure out how to get it from hydra yet
exp_dir = "../log/24-02-19/15-48-56/ppo/HalfCheetah-v4"


@hydra.main(version_base=None, config_path=os.path.join(exp_dir, ".hydra"), config_name="config")
def eval_exp(cfg: DictConfig):
    print(os.getcwd())
    log_dir = os.path.join(os.getcwd(), exp_dir)

    # experiment = Experiment.from_directory(log_dir)
    logger = PandasLogger(log_dir)
    logger.restore_data()

    train_data = logger.data['train']
    test_data = logger.data['test']

    n_train_envs = cfg.sampling_config.num_train_envs
    n_iter = len(test_data)
    n_test_envs = cfg.sampling_config.num_test_envs
    n_ret_per_env = cfg.sampling_config.episode_per_test // n_test_envs

    test_seed_set_size = 4
    split = n_test_envs // test_seed_set_size

    env_step_train = [d['env_step'] for d in train_data]
    env_step_test = [d['env_step'] for d in test_data]

    test_episode_returns_per_env = np.zeros((n_iter, n_test_envs, n_ret_per_env))
    for i in range(cfg.sampling_config.num_test_envs):
        test_episode_returns_per_env[:, i, :] = np.stack([iter_data['episode_returns_per_env'][f'env_{i}'] for iter_data in test_data])

    train_avg_return = np.array([d['returns_stat']['mean'] for d in train_data])
    train_std_return = np.array([d['returns_stat']['std'] for d in train_data])

    test_return_sets = {i: {} for i in range(split)}

    for i in range(split):
        test_return_set = test_episode_returns_per_env[:, i * test_seed_set_size: i * test_seed_set_size + test_seed_set_size, :]
        test_return_sets[i]['test_return_set'] = test_return_set
        test_return_sets[i]['test_avg_return'] = np.mean(test_return_set, axis=(1, 2))
        test_return_sets[i]['test_std_return'] = np.std(test_return_set, axis=(1, 2))

    fig, ax = plt.subplots()
    ax.plot(env_step_train, train_avg_return, lw=2, label=f'avg return train (mean seed 0:{n_train_envs})')
    ax.fill_between(env_step_train, train_avg_return + train_std_return, train_avg_return - train_std_return, facecolor='C0', alpha=0.4)
    for i in range(split):
        ax.plot(env_step_test, test_return_sets[i]['test_avg_return'], lw=2, label=f'avg return test (mean seed {i * test_seed_set_size}:{i * test_seed_set_size + test_seed_set_size})')
        ax.fill_between(env_step_test, test_return_sets[i]['test_avg_return'] + test_return_sets[i]['test_std_return'],
                        test_return_sets[i]['test_avg_return'] - test_return_sets[i]['test_std_return'], facecolor=f'C{i+1}', alpha=0.4)
    ax.legend()
    # plt.show()

    # 2 sample Kolmogorov-Smirnov test
    p_values_all = {i: {} for i in range(split - 1)}
    for j in range(split - 1):
        p_values = []
        for i in range(n_iter):
            p_values.append(sst.ks_2samp(test_return_sets[0]['test_return_set'][i].flatten(),
                                         test_return_sets[j + 1]['test_return_set'][i].flatten()).pvalue)
        p_values_all[j] = p_values

    fig, ax = plt.subplots()
    for i in range(split - 1):
        ax.plot(env_step_test, p_values_all[i], lw=2, label=f'p-value test set 0 vs test set {i + 1}')
    ax.legend()
    print()


if __name__ == "__main__":
    eval_exp()
