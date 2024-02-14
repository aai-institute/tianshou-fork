import os

import hydra
import numpy as np
from omegaconf import DictConfig

from tianshou.highlevel.experiment import Experiment
from experiments.algo_eval.run_ppo_experiment import PandasLoggerFactory  # needed to call from_directory
# from tianshou.highlevel.logger import LoggerFactory, TLogger
# from tianshou.utils.logger.pandas_logger import PandasLogger

# custom path to the experiment directory, couldn't figure out how to get it from hydra yet
exp_dir = "log/24-02-14/18-55-09/ppo/Pendulum-v1"

# class PandasLoggerFactory(LoggerFactory):
#     def create_logger(self, log_dir: str,
#                       experiment_name: str,
#                       run_id: str | None,
#                       config_dict: dict) -> TLogger:
#         return PandasLogger(log_dir,
#                             exclude_arrays=False)


@hydra.main(version_base=None, config_path=os.path.join(exp_dir, ".hydra"), config_name="config")
def eval_exp(cfg: DictConfig):
    print(os.getcwd())
    log_dir = os.path.join(os.getcwd(), exp_dir)

    experiment = Experiment.from_directory(log_dir)

    logger = experiment.logger_factory.create_logger(log_dir, None, None, None)
    logger.restore_data()

    test_data = logger.data['test']

    n_iter = len(test_data)
    n_test_envs = cfg.sampling_config.num_test_envs
    n_ret_per_env = cfg.sampling_config.episode_per_test // n_test_envs

    episode_returns_per_env = np.zeros((n_iter, n_test_envs, n_ret_per_env))
    for i in range(cfg.sampling_config.num_test_envs):
        episode_returns_per_env[:, i, :] = np.stack([iter_data[f'test/episode_returns_per_env/env_{i}'] for iter_data in test_data])


if __name__ == "__main__":
    eval_exp()
