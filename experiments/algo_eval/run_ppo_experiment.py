from dataclasses import dataclass, field
from typing import Sequence

import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from experiments.exp_builders.ppo_exp_builder import PPOWithDefaultActorAndDefaultCriticExperimentBuilder
from tianshou.highlevel.logger import LoggerFactory, TLogger
from tianshou.highlevel.params.lr_scheduler import LRSchedulerFactoryLinear
from tianshou.utils.logger.pandas_logger import PandasLogger

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


class PandasLoggerFactory(LoggerFactory):
    def create_logger(self, log_dir: str,
                      experiment_name: str,
                      run_id: str | None,
                      config_dict: dict) -> TLogger:
        return PandasLogger(log_dir)


@hydra.main(version_base=None, config_path="../configs", config_name="ppo_experiment_config")
def run_exp(cfg: DictConfig):
    print(cfg)
    experiment_config = hydra.utils.instantiate(cfg.experiment_config)
    sampling_config = hydra.utils.instantiate(cfg.sampling_config)

    lr_scheduler = LRSchedulerFactoryLinear(sampling_config) if cfg.lr_decay else None
    policy_params = hydra.utils.instantiate(cfg.policy_params,
                                            lr_scheduler_factory=lr_scheduler)
    env_factory = hydra.utils.instantiate(cfg.env_factory)

    experiment_builder = PPOWithDefaultActorAndDefaultCriticExperimentBuilder(env_factory,
                                                                              experiment_config,
                                                                              sampling_config,
                                                                              policy_params,
                                                                              cfg.hidden_sizes,
                                                                              logger_factory=PandasLoggerFactory(),
                                                                              )
    experiment = experiment_builder.build()
    log_name = HydraConfig.get().runtime.output_dir
    print(log_name)
    result = experiment.run(log_name)
    print(result)


if __name__ == "__main__":
    run_exp()
