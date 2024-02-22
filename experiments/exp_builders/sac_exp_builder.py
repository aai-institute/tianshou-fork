from copy import copy
from typing import Sequence

from experiments.exp_builders import SeededExperimentFactory
from tianshou.highlevel.config import SamplingConfig
from tianshou.highlevel.env import EnvFactory
from tianshou.highlevel.experiment import SACExperimentBuilder, ExperimentConfig, Experiment
from tianshou.highlevel.logger import LoggerFactory
from tianshou.highlevel.params.policy_params import SACParams


class SACSeededExperimentFactory(SeededExperimentFactory):
    def __init__(self,
                 env_factory: EnvFactory,
                 experiment_config: ExperimentConfig,
                 sampling_config: SamplingConfig,
                 policy_params: SACParams,
                 logger_factory: LoggerFactory,
                 hidden_sizes: Sequence[int]):
        self.hidden_sizes = hidden_sizes
        self.experiment_config = experiment_config
        self.sampling_config = sampling_config
        self.policy_params = policy_params
        self.logger_factory = logger_factory
        self.env_factory = env_factory

    def create_experiment(self, policy_seed: int, train_seed, test_seed) -> Experiment:
        experiment_config = copy(self.experiment_config)
        experiment_config.seed = policy_seed

        sampling_config = copy(self.sampling_config)
        sampling_config.train_seed = train_seed
        sampling_config.test_seed = test_seed

        return SACExperimentBuilder(self.env_factory, experiment_config, sampling_config) \
            .with_sac_params(self.policy_params) \
            .with_actor_factory_default(
                self.hidden_sizes,
                # hidden_activation=torch.nn.Tanh,
                continuous_unbounded=True,
                continuous_conditioned_sigma=True,
            ) \
            .with_common_critic_factory_default(
                self.hidden_sizes,
                # hidden_activation=torch.nn.Tanh,
            ) \
            .with_logger_factory(self.logger_factory) \
            .build()
