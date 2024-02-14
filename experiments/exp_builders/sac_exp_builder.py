from typing import Sequence

from tianshou.highlevel.config import SamplingConfig
from tianshou.highlevel.env import EnvFactory
from tianshou.highlevel.experiment import SACExperimentBuilder, ExperimentConfig
from tianshou.highlevel.params.policy_params import SACParams


class SACWithDefaultActorAndCommonCriticExperimentBuilder(SACExperimentBuilder):
    def __init__(self,
                 env_factory: EnvFactory,
                 experiment_config: ExperimentConfig,
                 sampling_config: SamplingConfig,
                 policy_params: SACParams,
                 hidden_sizes: Sequence[int]):
        super().__init__(env_factory, experiment_config, sampling_config)
        self.with_sac_params(policy_params)
        self.with_actor_factory_default(
            hidden_sizes,
            # hidden_activation=torch.nn.Tanh,
            continuous_unbounded=True,
            continuous_conditioned_sigma=True,
        )
        self.with_common_critic_factory_default(
            hidden_sizes,
            # hidden_activation=torch.nn.Tanh,
        )