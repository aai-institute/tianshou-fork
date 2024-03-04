from copy import copy
from typing import Sequence

import torch

from experiments.exp_builders import SeededExperimentFactory
from tianshou.highlevel.config import SamplingConfig
from tianshou.highlevel.env import EnvFactory
from tianshou.highlevel.experiment import PPOExperimentBuilder, ExperimentConfig, Experiment
from tianshou.highlevel.logger import LoggerFactory
from tianshou.highlevel.params.policy_params import PPOParams


class PPOSeededExperimentFactory(SeededExperimentFactory):
    def __init__(self,
                 env_factory: EnvFactory,
                 experiment_config: ExperimentConfig,
                 sampling_config: SamplingConfig,
                 policy_params: PPOParams,
                 hidden_sizes: Sequence[int],
                 logger_factory: LoggerFactory,
                 ):
        self.hidden_sizes = hidden_sizes
        self.experiment_config = experiment_config
        self.sampling_config = sampling_config
        self.policy_params = policy_params
        self.logger_factory = logger_factory
        self.env_factory = env_factory

    def create_experiment(self, policy_seed: int | None, train_seed: int | None, test_seed: int | None) -> Experiment:
        experiment_config = copy(self.experiment_config)
        if policy_seed is not None:
            experiment_config.seed = policy_seed

        sampling_config = copy(self.sampling_config)
        if train_seed is not None:
            sampling_config.train_seed = train_seed
        if test_seed is not None:
            sampling_config.test_seed = test_seed

        return PPOExperimentBuilder(self.env_factory, experiment_config, sampling_config) \
            .with_ppo_params(self.policy_params) \
            .with_actor_factory_default(
                self.hidden_sizes,
                hidden_activation=torch.nn.Tanh,
                continuous_unbounded=True,
                continuous_conditioned_sigma=False,
            ) \
            .with_critic_factory_default(
                self.hidden_sizes,
                hidden_activation=torch.nn.Tanh,
            ) \
            .with_logger_factory(self.logger_factory) \
            .build()
