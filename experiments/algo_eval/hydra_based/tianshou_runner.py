import os

import hydra
import numpy as np
import scipy.stats as sst
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from experiments.algo_eval.utils import shortener
from experiments.algo_eval.seed_analysis import SeedConfiguration, SeedVariabilityAnalysis
from tianshou.trainer.utils import test_episode


OmegaConf.register_new_resolver("short_dir", shortener)
OmegaConf.register_new_resolver("format", lambda inpt, formatter: formatter.format(inpt))


@hydra.main(version_base=None, config_path="configs", config_name="hpo_ppo_pendulum_config")
def run_exp(cfg: DictConfig):
    print(cfg)
    log_dir = HydraConfig.get().runtime.output_dir
    log_base_dir, experiment_name = os.path.split(log_dir)
    print(log_base_dir)
    cfg.experiment.experiment_config.persistence_base_dir = log_base_dir

    experiment_factory = hydra.utils.instantiate(cfg.experiment)

    # Use the seed configuration from DEHB sweeper
    # experiment = experiment_factory.create_experiment(cfg.seed,
    #                                                   cfg.seed,
    #                                                   cfg.test_seed)
    # experiment_result = experiment.run(experiment_name)
    # try:
    #     result = test_episode(policy=experiment_result.world.policy,
    #                           collector=experiment_result.world.test_collector,
    #                           n_episode=cfg.n_eval_episodes,
    #                           epoch=experiment_result.world.trainer.epoch,
    #                           test_fn=None,
    #                           )
    #
    #     mean_rewards = -result.returns_stat.mean  # these should be sufficiently many as not to need IQM
    # except Exception as e:
    #     mean_rewards = float("nan")
    # return mean_rewards

    # Use explicit seed configuration
    seed_config = SeedConfiguration(**cfg.seed_config)
    seed_analysis = SeedVariabilityAnalysis(seed_config=seed_config,
                                            seeded_experiment_factory=experiment_factory)

    mean_reward_over_seeds = []
    for sub_exp_name, experiment in seed_analysis.build_experiments():
        try:
            experiment_result = experiment.run(os.path.join(experiment_name, sub_exp_name))

            result = test_episode(policy=experiment_result.world.policy,
                                  collector=experiment_result.world.test_collector,
                                  n_episode=cfg.n_eval_episodes,
                                  epoch=experiment_result.world.trainer.epoch,
                                  test_fn=None,
                                  )

            mean_rewards = -result.returns_stat.mean  # these should be sufficiently many as not to need IQM
        except Exception as e:
            print(f"An exception occurred: {e}")
            mean_rewards = float("nan")
        mean_reward_over_seeds.append(mean_rewards)

    mean_reward_over_seeds = [m_r for m_r in mean_reward_over_seeds if not np.isnan(m_r)]
    if mean_reward_over_seeds:
        if cfg.aggregate_fn == "mean":
            return np.mean(mean_reward_over_seeds)
        elif cfg.aggregate_fn == "iqm":
            return sst.trim_mean(mean_reward_over_seeds, proportiontocut=0.25, axis=0)
        else:
            raise ValueError(f"Unknown aggregation function: {cfg.aggregate_fn}")
    else:
        return float("nan")


if __name__ == "__main__":
    run_exp()
