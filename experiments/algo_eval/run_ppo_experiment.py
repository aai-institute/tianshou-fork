import os
import pickle
from dataclasses import dataclass, field
from typing import Sequence

import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch

from tianshou.highlevel.experiment import PPOExperimentBuilder
from tianshou.highlevel.logger import PandasLoggerFactory
from tianshou.highlevel.params.lr_scheduler import LRSchedulerFactoryLinear

OmegaConf.register_new_resolver("format", lambda inpt, formatter: formatter.format(inpt))


@dataclass
class PPOEvalConfig:
    task: str = "Pendulum-v1"
    policy_seeds: Sequence[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    base_train_env_seed: int = 42
    base_test_env_seed: int = 1337


@dataclass
class PPOEvalTrainSeedConfig:
    task: str = "Pendulum-v1"
    policy_seeds: int = 0
    base_train_env_seed: Sequence[int] = field(default_factory=lambda: [42, 1000, 2000, 3000, 4000])
    base_test_env_seed: int = 1337


# cs = ConfigStore.instance()
# # Registering the Config class with the name `ppo_eval_config` with the config group `eval_config`
# cs.store(name="ppo_eval_config", group="eval_config", node=PPOEvalConfig)


@hydra.main(version_base=None, config_path="../configs", config_name="ppo_experiment_config")
def run_exp(cfg: DictConfig):
    print(cfg)
    experiment_config = hydra.utils.instantiate(cfg.experiment_config)
    sampling_config = hydra.utils.instantiate(cfg.sampling_config)

    lr_scheduler = LRSchedulerFactoryLinear(sampling_config) if cfg.lr_decay else None
    policy_params = hydra.utils.instantiate(cfg.policy_params,
                                            lr_scheduler_factory=lr_scheduler)
    env_factory = hydra.utils.instantiate(cfg.env_factory)

    experiment = (
        PPOExperimentBuilder(env_factory, experiment_config, sampling_config)
        .with_ppo_params(policy_params)
        .with_actor_factory_default(
            cfg.hidden_sizes,
            hidden_activation=torch.nn.Tanh,
            continuous_unbounded=True,
            continuous_conditioned_sigma=False,
        )
        .with_critic_factory_default(
            cfg.hidden_sizes,
            hidden_activation=torch.nn.Tanh,
        )
        .with_logger_factory(PandasLoggerFactory())
        .build())

    log_name = HydraConfig.get().runtime.output_dir
    print(log_name)
    experiment_result = experiment.run(log_name)
    print(experiment_result)


if __name__ == "__main__":
    run_exp()
