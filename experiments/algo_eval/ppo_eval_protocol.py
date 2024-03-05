import os
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from examples.mujoco.mujoco_env import MujocoEnvFactory
from experiments.algo_eval.seed_analysis import SeedConfiguration, SeedVariabilityAnalysis, ExperimentResults, \
    JoblibConfig
from experiments.algo_eval.utils import shortener
from experiments.exp_builders.ppo_exp_builder import PPOSeededExperimentFactory
from tianshou.highlevel.config import SamplingConfig
from tianshou.highlevel.env import EnvFactoryRegistered, VectorEnvType
from tianshou.highlevel.experiment import ExperimentConfig
from tianshou.highlevel.logger import LoggerManagerFactory
from tianshou.highlevel.params.dist_fn import DistributionFunctionFactoryIndependentGaussians
from tianshou.highlevel.params.lr_scheduler import LRSchedulerFactoryLinear
from tianshou.highlevel.params.policy_params import PPOParams, Params
from tianshou.utils.logger.pandas_logger import PandasLogger
from tianshou.utils.logging import datetime_tag


# policy_eval_config_space = ConfigurationSpace(
#     name="eval_config",
#     space={
#         "policy_seeds":  [0, 1, 2, 3, 4],
#         "base_train_env_seed": [42, 1000, 2000, 3000, 4000],
#         "base_test_env_seed": 1337
#     }
# )


@dataclass
class PPOEvalConfig:
    hidden_sizes: Sequence[int]
    seed_config: SeedConfiguration
    experiment_config: ExperimentConfig
    sampling_config: SamplingConfig
    policy_params: Params
    env_factory: EnvFactoryRegistered

    def __post_init__(self):
        for train_env_seed in self.seed_config.train_env_seeds:
            if isinstance(train_env_seed, Sequence):
                assert len(train_env_seed) == self.sampling_config.num_train_envs

        for test_env_seed in self.seed_config.test_env_seeds:
            if isinstance(test_env_seed, Sequence):
                assert len(test_env_seed) == self.sampling_config.num_test_envs

        self.log_dir = os.path.join("log", datetime_tag(), "ppo", self.env_factory.task)
        self.experiment_config.persistence_base_dir = self.log_dir
        # subdir: ${short_dir:${hydra.job.override_dirname}, 3}


if __name__ == "__main__":

    seed_config = SeedConfiguration(
        seeds=list(range(2)),
    )

    experiment_config: ExperimentConfig = ExperimentConfig(
        seed=0,
        watch=False,
        watch_render=0.0,
        log_file_enabled=True
    )

    sampling_config = SamplingConfig(
        step_per_collect=2048,
        buffer_size=2048,
        num_epochs=1,
        step_per_epoch=2048,
        batch_size=64,
        num_train_envs=2,
        train_seed=0,
        num_test_envs=2,
        episode_per_test=32,
        sample_equal_from_each_env=True,
        repeat_per_collect=10,
        )

    policy_params: Params = PPOParams(
        discount_factor=0.99,
        gae_lambda=0.95,
        action_bound_method="clip",
        reward_normalization=True,
        ent_coef=0.0,
        vf_coef=0.25,
        max_grad_norm=0.5,
        value_clip=False,
        advantage_normalization=False,
        eps_clip=0.2,
        dual_clip=None,
        recompute_advantage=True,
        lr=3e-4,
        lr_scheduler_factory=LRSchedulerFactoryLinear(sampling_config=sampling_config),
        dist_fn=DistributionFunctionFactoryIndependentGaussians(),
        )

    env_factory: EnvFactoryRegistered = MujocoEnvFactory(
        task="HalfCheetah-v4",
        seed=0,
        obs_norm=True,
        venv_kwargs={'context': 'fork'},
        # venv_type=VectorEnvType.DUMMY
    )

    joblib_config = JoblibConfig(
        n_jobs=2,
        # backend="multiprocessing",
        # require="sharedmem",
        # prefer="processes",
        verbose=10,
    )

    algo_config = PPOEvalConfig(seed_config=seed_config,
                                env_factory=env_factory,
                                experiment_config=experiment_config,
                                policy_params=policy_params,
                                hidden_sizes=[64, 64],
                                sampling_config=sampling_config)

    seeded_experiment_factory = PPOSeededExperimentFactory(algo_config.env_factory,
                                                           algo_config.experiment_config,
                                                           algo_config.sampling_config,
                                                           algo_config.policy_params,
                                                           algo_config.hidden_sizes,
                                                           LoggerManagerFactory(['tensorboard', 'pandas'],
                                                                                'ppo_eval'))

    seed_variability_analysis = SeedVariabilityAnalysis(seed_config, seeded_experiment_factory)
    results = seed_variability_analysis.run_joblib_local(joblib_config)

    test_episode_returns = []

    for seed in seed_config.seeds:
        full_name = f"seed={seed}"
        experiment_name = shortener(full_name, 3)
        log_dir = os.path.join(algo_config.log_dir, experiment_name)
        print(log_dir)

        logger = PandasLogger(log_dir, exclude_arrays=False)
        logger.restore_data()

        train_data = logger.data['train']
        test_data = logger.data['test']

        test_episode_returns.append([d['returns_stat']['mean'] for d in test_data])

    results = ExperimentResults(algorithms=['PPO'],
                                score_dict={'PPO': np.array(test_episode_returns)},
                                env_steps=np.array([d['env_step'] for d in test_data]),
                                score_thresholds=np.linspace(0.0, 8000.0, 81))

    seed_variability_analysis.eval_results(results)

    print()
