#!/usr/bin/env python3

import datetime
import os
from collections.abc import Sequence

from jsonargparse import CLI

from examples.mujoco.mujoco_env import MujocoEnvFactory
from tianshou.highlevel.config import RLSamplingConfig
from tianshou.highlevel.experiment import (
    RLExperimentConfig,
    SACExperimentBuilder,
)
from tianshou.highlevel.params.alpha import AutoAlphaFactoryDefault
from tianshou.highlevel.params.policy_params import SACParams
from tianshou.utils import logging


def main(
    experiment_config: RLExperimentConfig,
    task: str = "Ant-v3",
    buffer_size: int = 1000000,
    hidden_sizes: Sequence[int] = (256, 256),
    actor_lr: float = 1e-3,
    critic_lr: float = 1e-3,
    gamma: float = 0.99,
    tau: float = 0.005,
    alpha: float = 0.2,
    auto_alpha: bool = False,
    alpha_lr: float = 3e-4,
    start_timesteps: int = 10000,
    epoch: int = 200,
    step_per_epoch: int = 5000,
    step_per_collect: int = 1,
    update_per_step: int = 1,
    n_step: int = 1,
    batch_size: int = 256,
    training_num: int = 1,
    test_num: int = 10,
):
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    log_name = os.path.join(task, "sac", str(experiment_config.seed), now)

    sampling_config = RLSamplingConfig(
        num_epochs=epoch,
        step_per_epoch=step_per_epoch,
        num_train_envs=training_num,
        num_test_envs=test_num,
        buffer_size=buffer_size,
        batch_size=batch_size,
        step_per_collect=step_per_collect,
        update_per_step=update_per_step,
        start_timesteps=start_timesteps,
        start_timesteps_random=True,
    )

    env_factory = MujocoEnvFactory(task, experiment_config.seed, sampling_config)

    experiment = (
        SACExperimentBuilder(experiment_config, env_factory, sampling_config)
        .with_sac_params(
            SACParams(
                tau=tau,
                gamma=gamma,
                alpha=AutoAlphaFactoryDefault(lr=alpha_lr) if auto_alpha else alpha,
                estimation_step=n_step,
                actor_lr=actor_lr,
                critic1_lr=critic_lr,
                critic2_lr=critic_lr,
            ),
        )
        .with_actor_factory_default(
            hidden_sizes,
            continuous_unbounded=True,
            continuous_conditioned_sigma=True,
        )
        .with_common_critic_factory_default(hidden_sizes)
        .build()
    )
    experiment.run(log_name)


if __name__ == "__main__":
    logging.run_main(lambda: CLI(main))
