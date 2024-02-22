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


@hydra.main(version_base=None, config_path="configs", config_name="eval_sac_on_ant_config")
def run_exp(cfg: DictConfig):
    print(cfg)
    log_dir = HydraConfig.get().runtime.output_dir
    log_base_dir, experiment_name = os.path.split(log_dir)
    print(log_base_dir)
    cfg.experiment_config.persistence_base_dir = log_base_dir

    experiment_factory = hydra.utils.instantiate(cfg)
    experiment = experiment_factory.create_experiment(cfg.experiment_config.seed,
                                                      cfg.sampling_config.train_seed,
                                                      cfg.sampling_config.test_seed)

    experiment_result = experiment.run(experiment_name)
    print(experiment_result)


if __name__ == "__main__":
    run_exp()
