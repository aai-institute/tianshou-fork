from typing import Sequence

import torch

from tianshou.highlevel.config import SamplingConfig
from tianshou.highlevel.env import EnvFactory
from tianshou.highlevel.experiment import PPOExperimentBuilder, ExperimentConfig
from tianshou.highlevel.logger import LoggerFactory
from tianshou.highlevel.params.policy_params import PPOParams


class PPOWithDefaultActorAndDefaultCriticExperimentBuilder(PPOExperimentBuilder):
    def __init__(self,
                 env_factory: EnvFactory,
                 experiment_config: ExperimentConfig,
                 sampling_config: SamplingConfig,
                 policy_params: PPOParams,
                 hidden_sizes: Sequence[int],
                 logger_factory: LoggerFactory,
                 ):
        super().__init__(env_factory, experiment_config, sampling_config)
        self.with_ppo_params(policy_params)
        self.with_actor_factory_default(
            hidden_sizes,
            hidden_activation=torch.nn.Tanh,
            continuous_unbounded=True,
            continuous_conditioned_sigma=False,
        )
        self.with_critic_factory_default(
            hidden_sizes,
            hidden_activation=torch.nn.Tanh,
        )
        self.with_logger_factory(logger_factory)
