from typing import Callable, Optional, Sequence, Union

import numpy as np
import torch
from torch import nn
from torch.distributions import Independent, Normal

from tianshou.env import VectorEnvNormObs
from tianshou.policy import BasePolicy
from tianshou.utils.net.common import ActorCritic, ModuleType, Net
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.utils.types import TDevice, TOptimFactory, TShape


def get_module_by_name(module: nn.Module, name: str) -> nn.Module:
    names = name.split(".")
    for n in names:
        try:
            module = getattr(module, n)
        except AttributeError:
            raise KeyError(f"Module {module} has no named module {n}")
        if not isinstance(module, nn.Module):
            raise KeyError(f"Expected to find a module named {n}, but found {module}")
    return module


def simple_nn_init(
    module: nn.Module,
    initializer: Callable[[torch.Tensor, ...], Optional[torch.Tensor]],
    layers_to_scale: Sequence[str] = (),
    weight_scale: float = 1.0,
    **initializer_kwargs,
):
    """Initializes the weights of a module with a given initializer.

    :param module: the module to initialize
    :param initializer: a function that takes a tensor and initializes it
    :param layers_to_scale: layer names to be scale by `weight_scale`. The bias
        parameters of these layers will be zeroed.
    :param weight_scale: the scale to apply to the weights of layers in `layers_to_scale`
    :param initializer_kwargs: keyword arguments to pass to the initializer
    """
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            initializer(m.weight, **initializer_kwargs)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    for name in layers_to_scale:
        layer = get_module_by_name(module, name)
        try:
            layer.weight.data *= weight_scale
            layer.bias.data *= 0.0
        except AttributeError:
            raise KeyError(f"Module {layer} has no attribute weight or bias")


def resume_from_checkpoint(
    path: str,
    policy: BasePolicy,
    train_envs: Optional[VectorEnvNormObs] = None,
    test_envs: Optional[VectorEnvNormObs] = None,
    device: TDevice = None,
):
    ckpt = torch.load(path, map_location=device)
    policy.load_state_dict(ckpt["model"])
    print("Loaded agent from: ", path)

    obs_rms = ckpt.get("obs_rms")
    if obs_rms is not None:
        print(f"Loaded observation running mean from {path}")
        if train_envs:
            train_envs.set_obs_rms(ckpt["obs_rms"])
        if test_envs:
            test_envs.set_obs_rms(ckpt["obs_rms"])


def get_actor_critic(
    state_shape: TShape,
    hidden_sizes: Sequence[int],
    action_shape: TShape,
    activation_a: Union[ModuleType, Sequence[ModuleType]] = nn.Tanh,
    activation_c: Union[ModuleType, Sequence[ModuleType]] = nn.Tanh,
    device: TDevice = "cpu",
):
    net_a = Net(state_shape, hidden_sizes=hidden_sizes, activation=activation_a, device=device)
    actor = ActorProb(net_a, action_shape, unbounded=True, device=device)
    net_c = Net(state_shape, hidden_sizes=hidden_sizes, activation=activation_c, device=device)
    critic = Critic(net_c)
    return actor, critic


def init_actor_critic(actor: nn.Module, critic: nn.Module):
    """Initializes layers of actor and critic and returns an actor_critic object.

    **Note**: this modifies the actor and critic in place.
    """

    actor_critic = ActorCritic(actor, critic)

    torch.nn.init.constant_(actor.sigma_param, -0.5)
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            # orthogonal initialization
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)
    if hasattr(actor, "mu"):
        # For continuous action spaces with Gaussian policies
        # do last policy layer scaling, this will make initial actions have (close to)
        # 0 mean and std, and will help boost performances,
        # see https://arxiv.org/abs/2006.05990, Fig.24 for details
        for m in actor.mu.modules():
            # TODO: seems like biases are initialized twice for the actor
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.zeros_(m.bias)
                m.weight.data.copy_(0.01 * m.weight.data)
    return actor_critic


def init_and_get_optim(
    actor: nn.Module,
    critic: nn.Module,
    lr: float,
    optim_class: TOptimFactory = torch.optim.Adam,
    optim_on_actor_only: bool = False,
):
    """Initializes layers of actor and critic and returns an optimizer.

    :param actor:
    :param critic:
    :param lr:
    :param optim_class: optimizer class or callable, should accept `lr` as kwarg
    :return: the optimizer instance
    """
    actor_critic = init_actor_critic(actor, critic)
    params = actor_critic.parameters() if optim_on_actor_only else actor.parameters()
    return optim_class(params, lr=lr)


def std_normal(*logits):
    return Independent(Normal(*logits), 1)


# todo make this work with passing the std multiplier only once, this value if for halfcheetah
def fixed_std_normal(*logits):
    logits, std = logits
    std = torch.ones_like(std) * 0.4
    return Independent(Normal(logits, std), 1)
