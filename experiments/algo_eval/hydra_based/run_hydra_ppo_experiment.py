import os
import pickle
from dataclasses import dataclass, field
from typing import Sequence

import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch

from experiments.algo_eval.utils import shortener
from tianshou.highlevel.experiment import PPOExperimentBuilder
from tianshou.highlevel.logger import LoggerManagerFactory
from tianshou.highlevel.params.lr_scheduler import LRSchedulerFactoryLinear


OmegaConf.register_new_resolver("short_dir", shortener)
OmegaConf.register_new_resolver("format", lambda inpt, formatter: formatter.format(inpt))


# cs = ConfigStore.instance()
# # Registering the Config class with the name `ppo_eval_config` with the config group `eval_config`
# cs.store(name="ppo_eval_config", group="eval_config", node=PPOEvalConfig)


@hydra.main(version_base=None, config_path="../../configs", config_name="ppo_experiment_config")
def run_exp(cfg: DictConfig):
    print(cfg)
    log_dir = HydraConfig.get().runtime.output_dir
    log_base_dir, experiment_name = os.path.split(log_dir)
    print(log_base_dir)
    experiment_config = hydra.utils.instantiate(cfg.experiment_config,
                                                persistence_base_dir=log_base_dir,
                                                # device=f"cuda:{HydraConfig.get().job.num % 4}"
                                                )
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
        .with_logger_factory(LoggerManagerFactory(['tensorboard', 'pandas'], HydraConfig.get().job.name))
        .build())

    experiment_result = experiment.run(experiment_name)
    print(experiment_result)


if __name__ == "__main__":
    run_exp()
