from abc import ABC, abstractmethod
from collections.abc import Callable
from os import PathLike
from typing import Any, Generic, TypeVar, cast

import gymnasium
import torch

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.highlevel.config import SamplingConfig
from tianshou.highlevel.env import Environments
from tianshou.highlevel.logger import Logger
from tianshou.highlevel.module.actor import (
    ActorFactory,
)
from tianshou.highlevel.module.core import TDevice
from tianshou.highlevel.module.critic import CriticEnsembleFactory, CriticFactory
from tianshou.highlevel.module.module_opt import (
    ActorCriticModuleOpt,
)
from tianshou.highlevel.optim import OptimizerFactory
from tianshou.highlevel.params.policy_params import (
    A2CParams,
    DDPGParams,
    DiscreteSACParams,
    DQNParams,
    NPGParams,
    Params,
    ParamsMixinActorAndDualCritics,
    ParamsMixinLearningRateWithScheduler,
    ParamTransformerData,
    PGParams,
    PPOParams,
    REDQParams,
    SACParams,
    TD3Params,
    TRPOParams,
)
from tianshou.highlevel.params.policy_wrapper import PolicyWrapperFactory
from tianshou.highlevel.trainer import TrainerCallbacks, TrainingContext
from tianshou.policy import (
    A2CPolicy,
    BasePolicy,
    DDPGPolicy,
    DiscreteSACPolicy,
    DQNPolicy,
    NPGPolicy,
    PGPolicy,
    PPOPolicy,
    REDQPolicy,
    SACPolicy,
    TD3Policy,
    TRPOPolicy,
)
from tianshou.trainer import BaseTrainer, OffpolicyTrainer, OnpolicyTrainer
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.string import ToStringMixin

CHECKPOINT_DICT_KEY_MODEL = "model"
CHECKPOINT_DICT_KEY_OBS_RMS = "obs_rms"
TParams = TypeVar("TParams", bound=Params)
TActorCriticParams = TypeVar("TActorCriticParams", bound=ParamsMixinLearningRateWithScheduler)
TActorDualCriticsParams = TypeVar("TActorDualCriticsParams", bound=ParamsMixinActorAndDualCritics)
TPolicy = TypeVar("TPolicy", bound=BasePolicy)


class AgentFactory(ABC, ToStringMixin):
    def __init__(self, sampling_config: SamplingConfig, optim_factory: OptimizerFactory):
        self.sampling_config = sampling_config
        self.optim_factory = optim_factory
        self.policy_wrapper_factory: PolicyWrapperFactory | None = None
        self.trainer_callbacks: TrainerCallbacks = TrainerCallbacks()

    def create_train_test_collector(
        self,
        policy: BasePolicy,
        envs: Environments,
    ) -> tuple[Collector, Collector]:
        buffer_size = self.sampling_config.buffer_size
        train_envs = envs.train_envs
        buffer: ReplayBuffer
        if len(train_envs) > 1:
            buffer = VectorReplayBuffer(
                buffer_size,
                len(train_envs),
                stack_num=self.sampling_config.replay_buffer_stack_num,
                save_only_last_obs=self.sampling_config.replay_buffer_save_only_last_obs,
                ignore_obs_next=self.sampling_config.replay_buffer_ignore_obs_next,
            )
        else:
            buffer = ReplayBuffer(
                buffer_size,
                stack_num=self.sampling_config.replay_buffer_stack_num,
                save_only_last_obs=self.sampling_config.replay_buffer_save_only_last_obs,
                ignore_obs_next=self.sampling_config.replay_buffer_ignore_obs_next,
            )
        train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
        test_collector = Collector(policy, envs.test_envs)
        if self.sampling_config.start_timesteps > 0:
            train_collector.collect(n_step=self.sampling_config.start_timesteps, random=True)
        return train_collector, test_collector

    def set_policy_wrapper_factory(
        self,
        policy_wrapper_factory: PolicyWrapperFactory | None,
    ) -> None:
        self.policy_wrapper_factory = policy_wrapper_factory

    def set_trainer_callbacks(self, callbacks: TrainerCallbacks) -> None:
        self.trainer_callbacks = callbacks

    @abstractmethod
    def _create_policy(self, envs: Environments, device: TDevice) -> BasePolicy:
        pass

    def create_policy(self, envs: Environments, device: TDevice) -> BasePolicy:
        policy = self._create_policy(envs, device)
        if self.policy_wrapper_factory is not None:
            policy = self.policy_wrapper_factory.create_wrapped_policy(
                policy,
                envs,
                self.optim_factory,
                device,
            )
        return policy

    @staticmethod
    def _create_save_best_fn(envs: Environments, log_path: str) -> Callable:
        def save_best_fn(pol: torch.nn.Module) -> None:
            pass
            # TODO: Fix saving in general (code works only for mujoco)
            # state = {
            #    CHECKPOINT_DICT_KEY_MODEL: pol.state_dict(),
            #    CHECKPOINT_DICT_KEY_OBS_RMS: envs.train_envs.get_obs_rms(),
            # }
            # torch.save(state, os.path.join(log_path, "policy.pth"))

        return save_best_fn

    @staticmethod
    def load_checkpoint(
        policy: torch.nn.Module,
        path: str | PathLike,
        envs: Environments,
        device: TDevice,
    ) -> None:
        ckpt = torch.load(path, map_location=device)
        policy.load_state_dict(ckpt[CHECKPOINT_DICT_KEY_MODEL])
        if envs.train_envs:
            envs.train_envs.set_obs_rms(ckpt[CHECKPOINT_DICT_KEY_OBS_RMS])
        if envs.test_envs:
            envs.test_envs.set_obs_rms(ckpt[CHECKPOINT_DICT_KEY_OBS_RMS])
        print("Loaded agent and obs. running means from: ", path)  # TODO logging

    @abstractmethod
    def create_trainer(
        self,
        policy: BasePolicy,
        train_collector: Collector,
        test_collector: Collector,
        envs: Environments,
        logger: Logger,
    ) -> BaseTrainer:
        pass


class OnpolicyAgentFactory(AgentFactory, ABC):
    def create_trainer(
        self,
        policy: BasePolicy,
        train_collector: Collector,
        test_collector: Collector,
        envs: Environments,
        logger: Logger,
    ) -> OnpolicyTrainer:
        sampling_config = self.sampling_config
        callbacks = self.trainer_callbacks
        context = TrainingContext(policy, envs, logger)
        train_fn = (
            callbacks.epoch_callback_train.get_trainer_fn(context)
            if callbacks.epoch_callback_train
            else None
        )
        test_fn = (
            callbacks.epoch_callback_test.get_trainer_fn(context)
            if callbacks.epoch_callback_test
            else None
        )
        stop_fn = (
            callbacks.stop_callback.get_trainer_fn(context) if callbacks.stop_callback else None
        )
        return OnpolicyTrainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=sampling_config.num_epochs,
            step_per_epoch=sampling_config.step_per_epoch,
            repeat_per_collect=sampling_config.repeat_per_collect,
            episode_per_test=sampling_config.num_test_envs,
            batch_size=sampling_config.batch_size,
            step_per_collect=sampling_config.step_per_collect,
            save_best_fn=self._create_save_best_fn(envs, logger.log_path),
            logger=logger.logger,
            test_in_train=False,
            train_fn=train_fn,
            test_fn=test_fn,
            stop_fn=stop_fn,
        )


class OffpolicyAgentFactory(AgentFactory, ABC):
    def create_trainer(
        self,
        policy: BasePolicy,
        train_collector: Collector,
        test_collector: Collector,
        envs: Environments,
        logger: Logger,
    ) -> OffpolicyTrainer:
        sampling_config = self.sampling_config
        callbacks = self.trainer_callbacks
        context = TrainingContext(policy, envs, logger)
        train_fn = (
            callbacks.epoch_callback_train.get_trainer_fn(context)
            if callbacks.epoch_callback_train
            else None
        )
        test_fn = (
            callbacks.epoch_callback_test.get_trainer_fn(context)
            if callbacks.epoch_callback_test
            else None
        )
        stop_fn = (
            callbacks.stop_callback.get_trainer_fn(context) if callbacks.stop_callback else None
        )
        return OffpolicyTrainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=sampling_config.num_epochs,
            step_per_epoch=sampling_config.step_per_epoch,
            step_per_collect=sampling_config.step_per_collect,
            episode_per_test=sampling_config.num_test_envs,
            batch_size=sampling_config.batch_size,
            save_best_fn=self._create_save_best_fn(envs, logger.log_path),
            logger=logger.logger,
            update_per_step=sampling_config.update_per_step,
            test_in_train=False,
            train_fn=train_fn,
            test_fn=test_fn,
            stop_fn=stop_fn,
        )


class _ActorCriticMixin:  # TODO merge
    """Mixin for agents that use an ActorCritic module with a single optimizer."""

    def __init__(
        self,
        actor_factory: ActorFactory,
        critic_factory: CriticFactory,
        optim_factory: OptimizerFactory,
        critic_use_action: bool,
    ):
        self.actor_factory = actor_factory
        self.critic_factory = critic_factory
        self.optim_factory = optim_factory
        self.critic_use_action = critic_use_action

    def create_actor_critic_module_opt(
        self,
        envs: Environments,
        device: TDevice,
        lr: float,
    ) -> ActorCriticModuleOpt:
        actor = self.actor_factory.create_module(envs, device)
        critic = self.critic_factory.create_module(envs, device, use_action=self.critic_use_action)
        actor_critic = ActorCritic(actor, critic)
        optim = self.optim_factory.create_optimizer(actor_critic, lr)
        return ActorCriticModuleOpt(actor_critic, optim)


class PGAgentFactory(OnpolicyAgentFactory):
    def __init__(
        self,
        params: PGParams,
        sampling_config: SamplingConfig,
        actor_factory: ActorFactory,
        optim_factory: OptimizerFactory,
    ):
        super().__init__(sampling_config, optim_factory)
        self.params = params
        self.actor_factory = actor_factory
        self.optim_factory = optim_factory

    def _create_policy(self, envs: Environments, device: TDevice) -> PGPolicy:
        actor = self.actor_factory.create_module_opt(
            envs,
            device,
            self.optim_factory,
            self.params.lr,
        )
        kwargs = self.params.create_kwargs(
            ParamTransformerData(
                envs=envs,
                device=device,
                optim=actor.optim,
                optim_factory=self.optim_factory,
            ),
        )
        return PGPolicy(
            actor=actor.module,
            optim=actor.optim,
            action_space=envs.get_action_space(),
            observation_space=envs.get_observation_space(),
            **kwargs,
        )


class ActorCriticAgentFactory(
    Generic[TActorCriticParams, TPolicy],
    OnpolicyAgentFactory,
    ABC,
):
    def __init__(
        self,
        params: TActorCriticParams,
        sampling_config: SamplingConfig,
        actor_factory: ActorFactory,
        critic_factory: CriticFactory,
        optimizer_factory: OptimizerFactory,
    ):
        super().__init__(sampling_config, optim_factory=optimizer_factory)
        self.params = params
        self.actor_factory = actor_factory
        self.critic_factory = critic_factory
        self.optim_factory = optimizer_factory
        self.critic_use_action = False

    @abstractmethod
    def _get_policy_class(self) -> type[TPolicy]:
        pass

    def create_actor_critic_module_opt(
        self,
        envs: Environments,
        device: TDevice,
        lr: float,
    ) -> ActorCriticModuleOpt:
        actor = self.actor_factory.create_module(envs, device)
        critic = self.critic_factory.create_module(envs, device, use_action=self.critic_use_action)
        actor_critic = ActorCritic(actor, critic)
        optim = self.optim_factory.create_optimizer(actor_critic, lr)
        return ActorCriticModuleOpt(actor_critic, optim)

    def _create_kwargs(self, envs: Environments, device: TDevice) -> dict[str, Any]:
        actor_critic = self.create_actor_critic_module_opt(envs, device, self.params.lr)
        kwargs = self.params.create_kwargs(
            ParamTransformerData(
                envs=envs,
                device=device,
                optim_factory=self.optim_factory,
                optim=actor_critic.optim,
            ),
        )
        kwargs["actor"] = actor_critic.actor
        kwargs["critic"] = actor_critic.critic
        kwargs["optim"] = actor_critic.optim
        kwargs["action_space"] = envs.get_action_space()
        return kwargs

    def _create_policy(self, envs: Environments, device: TDevice) -> TPolicy:
        policy_class = self._get_policy_class()
        return policy_class(**self._create_kwargs(envs, device))


class A2CAgentFactory(ActorCriticAgentFactory[A2CParams, A2CPolicy]):
    def _get_policy_class(self) -> type[A2CPolicy]:
        return A2CPolicy


class PPOAgentFactory(ActorCriticAgentFactory[PPOParams, PPOPolicy]):
    def _get_policy_class(self) -> type[PPOPolicy]:
        return PPOPolicy


class NPGAgentFactory(ActorCriticAgentFactory[NPGParams, NPGPolicy]):
    def _get_policy_class(self) -> type[NPGPolicy]:
        return NPGPolicy


class TRPOAgentFactory(ActorCriticAgentFactory[TRPOParams, TRPOPolicy]):
    def _get_policy_class(self) -> type[TRPOPolicy]:
        return TRPOPolicy


class DQNAgentFactory(OffpolicyAgentFactory):
    def __init__(
        self,
        params: DQNParams,
        sampling_config: SamplingConfig,
        actor_factory: ActorFactory,
        optim_factory: OptimizerFactory,
    ):
        super().__init__(sampling_config, optim_factory)
        self.params = params
        self.actor_factory = actor_factory
        self.optim_factory = optim_factory

    def _create_policy(self, envs: Environments, device: TDevice) -> BasePolicy:
        model = self.actor_factory.create_module(envs, device)
        optim = self.optim_factory.create_optimizer(model, self.params.lr)
        kwargs = self.params.create_kwargs(
            ParamTransformerData(
                envs=envs,
                device=device,
                optim=optim,
                optim_factory=self.optim_factory,
            ),
        )
        envs.get_type().assert_discrete(self)
        action_space = cast(gymnasium.spaces.Discrete, envs.get_action_space())
        return DQNPolicy(
            model=model,
            optim=optim,
            action_space=action_space,
            observation_space=envs.get_observation_space(),
            **kwargs,
        )


class DDPGAgentFactory(OffpolicyAgentFactory):
    def __init__(
        self,
        params: DDPGParams,
        sampling_config: SamplingConfig,
        actor_factory: ActorFactory,
        critic_factory: CriticFactory,
        optim_factory: OptimizerFactory,
    ):
        super().__init__(sampling_config, optim_factory)
        self.critic_factory = critic_factory
        self.actor_factory = actor_factory
        self.params = params
        self.optim_factory = optim_factory

    def _create_policy(self, envs: Environments, device: TDevice) -> BasePolicy:
        actor = self.actor_factory.create_module_opt(
            envs,
            device,
            self.optim_factory,
            self.params.actor_lr,
        )
        critic = self.critic_factory.create_module_opt(
            envs,
            device,
            True,
            self.optim_factory,
            self.params.critic_lr,
        )
        kwargs = self.params.create_kwargs(
            ParamTransformerData(
                envs=envs,
                device=device,
                optim_factory=self.optim_factory,
                actor=actor,
                critic1=critic,
            ),
        )
        return DDPGPolicy(
            actor=actor.module,
            actor_optim=actor.optim,
            critic=critic.module,
            critic_optim=critic.optim,
            action_space=envs.get_action_space(),
            observation_space=envs.get_observation_space(),
            **kwargs,
        )


class REDQAgentFactory(OffpolicyAgentFactory):
    def __init__(
        self,
        params: REDQParams,
        sampling_config: SamplingConfig,
        actor_factory: ActorFactory,
        critic_ensemble_factory: CriticEnsembleFactory,
        optim_factory: OptimizerFactory,
    ):
        super().__init__(sampling_config, optim_factory)
        self.critic_ensemble_factory = critic_ensemble_factory
        self.actor_factory = actor_factory
        self.params = params
        self.optim_factory = optim_factory

    def _create_policy(self, envs: Environments, device: TDevice) -> BasePolicy:
        envs.get_type().assert_continuous(self)
        actor = self.actor_factory.create_module_opt(
            envs,
            device,
            self.optim_factory,
            self.params.actor_lr,
        )
        critic_ensemble = self.critic_ensemble_factory.create_module_opt(
            envs,
            device,
            self.params.ensemble_size,
            True,
            self.optim_factory,
            self.params.critic_lr,
        )
        kwargs = self.params.create_kwargs(
            ParamTransformerData(
                envs=envs,
                device=device,
                optim_factory=self.optim_factory,
                actor=actor,
                critic1=critic_ensemble,
            ),
        )
        action_space = cast(gymnasium.spaces.Box, envs.get_action_space())
        return REDQPolicy(
            actor=actor.module,
            actor_optim=actor.optim,
            critic=critic_ensemble.module,
            critic_optim=critic_ensemble.optim,
            action_space=action_space,
            observation_space=envs.get_observation_space(),
            **kwargs,
        )


class ActorDualCriticsAgentFactory(
    OffpolicyAgentFactory,
    Generic[TActorDualCriticsParams, TPolicy],
    ABC,
):
    def __init__(
        self,
        params: TActorDualCriticsParams,
        sampling_config: SamplingConfig,
        actor_factory: ActorFactory,
        critic1_factory: CriticFactory,
        critic2_factory: CriticFactory,
        optim_factory: OptimizerFactory,
    ):
        super().__init__(sampling_config, optim_factory)
        self.params = params
        self.actor_factory = actor_factory
        self.critic1_factory = critic1_factory
        self.critic2_factory = critic2_factory
        self.optim_factory = optim_factory

    @abstractmethod
    def _get_policy_class(self) -> type[TPolicy]:
        pass

    def _get_discrete_last_size_use_action_shape(self) -> bool:
        return True

    def _create_policy(self, envs: Environments, device: TDevice) -> TPolicy:
        actor = self.actor_factory.create_module_opt(
            envs,
            device,
            self.optim_factory,
            self.params.actor_lr,
        )
        use_action_shape = self._get_discrete_last_size_use_action_shape()
        critic1 = self.critic1_factory.create_module_opt(
            envs,
            device,
            True,
            self.optim_factory,
            self.params.critic1_lr,
            discrete_last_size_use_action_shape=use_action_shape,
        )
        critic2 = self.critic2_factory.create_module_opt(
            envs,
            device,
            True,
            self.optim_factory,
            self.params.critic2_lr,
            discrete_last_size_use_action_shape=use_action_shape,
        )
        kwargs = self.params.create_kwargs(
            ParamTransformerData(
                envs=envs,
                device=device,
                optim_factory=self.optim_factory,
                actor=actor,
                critic1=critic1,
                critic2=critic2,
            ),
        )
        policy_class = self._get_policy_class()
        return policy_class(
            actor=actor.module,
            actor_optim=actor.optim,
            critic=critic1.module,
            critic_optim=critic1.optim,
            critic2=critic2.module,
            critic2_optim=critic2.optim,
            action_space=envs.get_action_space(),
            observation_space=envs.get_observation_space(),
            **kwargs,
        )


class SACAgentFactory(ActorDualCriticsAgentFactory[SACParams, SACPolicy]):
    def _get_policy_class(self) -> type[SACPolicy]:
        return SACPolicy


class DiscreteSACAgentFactory(ActorDualCriticsAgentFactory[DiscreteSACParams, DiscreteSACPolicy]):
    def _get_policy_class(self) -> type[DiscreteSACPolicy]:
        return DiscreteSACPolicy


class TD3AgentFactory(ActorDualCriticsAgentFactory[TD3Params, TD3Policy]):
    def _get_policy_class(self) -> type[TD3Policy]:
        return TD3Policy
