# import os
# from pprint import pprint

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import torch

# from examples.mujoco.mujoco_env import MujocoEnvFactory
# from tianshou.highlevel.config import SamplingConfig
from tianshou.highlevel.experiment import PPOExperimentBuilder, ExperimentConfig
# from tianshou.highlevel.module.actor import ActorFactoryDefault
# from tianshou.highlevel.module.critic import CriticFactoryDefault
# from tianshou.highlevel.params.dist_fn import DistributionFunctionFactoryIndependentGaussians
# from tianshou.highlevel.params.lr_scheduler import LRSchedulerFactoryLinear
# from tianshou.highlevel.params.policy_params import PPOParams
# from tianshou.utils.logging import datetime_tag


OmegaConf.register_new_resolver("format", lambda inpt, formatter: formatter.format(inpt))


def get_experiment(config_dict):

    experiment_config = hydra.utils.instantiate(config_dict.experiment_config)
    # experiment_config = ExperimentConfig(seed=config_dict.seed,
    #                                      watch_render=False)

    sampling_config = hydra.utils.instantiate(config_dict.sampling_config)
    # sampling_config = SamplingConfig(
    #     num_epochs=config_dict.epoch,
    #     step_per_epoch=config_dict.step_per_epoch,
    #     batch_size=config_dict.batch_size,
    #     num_train_envs=config_dict.training_num,
    #     train_seed=config_dict.train_seed,
    #     num_test_envs=config_dict.test_num,
    #     test_seed=config_dict.test_seed,
    #     buffer_size=config_dict.buffer_size,
    #     step_per_collect=config_dict.step_per_collect,
    #     repeat_per_collect=config_dict.repeat_per_collect,
    # )

    ppo_config = hydra.utils.instantiate(config_dict.ppo_config)
    # ppo_config = PPOParams(
    #             discount_factor=config_dict.gamma,
    #             gae_lambda=config_dict.gae_lambda,
    #             action_bound_method=config_dict.bound_action_method,
    #             reward_normalization=config_dict.rew_norm,
    #             ent_coef=config_dict.ent_coef,
    #             vf_coef=config_dict.vf_coef,
    #             max_grad_norm=config_dict.max_grad_norm,
    #             value_clip=config_dict.value_clip,
    #             advantage_normalization=config_dict.norm_adv,
    #             eps_clip=config_dict.eps_clip,
    #             dual_clip=config_dict.dual_clip,
    #             recompute_advantage=config_dict.recompute_adv,
    #             lr=config_dict.lr,
    #             lr_scheduler_factory=LRSchedulerFactoryLinear(sampling_config)
    #             if config_dict.lr_decay
    #             else None,
    #             dist_fn=DistributionFunctionFactoryIndependentGaussians(),
    #         )

    env_factory = hydra.utils.instantiate(config_dict.env_factory)
    # env_factory = MujocoEnvFactory(task, experiment_config.seed, obs_norm=True)

    experiment = (
        PPOExperimentBuilder(env_factory, experiment_config, sampling_config)
        .with_ppo_params(ppo_config,
        )
        .with_actor_factory_default(config_dict.hidden_sizes, torch.nn.Tanh, continuous_unbounded=True)
        .with_critic_factory_default(config_dict.hidden_sizes, torch.nn.Tanh)
        .build()
    )
    return experiment


@hydra.main(version_base=None, config_path="configs", config_name="ppo_experiment_config")
def run_exp(cfg):
    print(cfg)
    experiment_builder = hydra.utils.instantiate(cfg)
    experiment = experiment_builder.build()
    # # experiment = get_experiment(cfg)
    # log_name = os.path.join(cfg.task, "ppo", str(cfg.seed), datetime_tag())
    log_name = HydraConfig.get().runtime.output_dir
    experiment.run(log_name)


if __name__ == "__main__":
    run_exp()
