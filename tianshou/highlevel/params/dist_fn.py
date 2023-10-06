from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TypeAlias

import torch

from tianshou.highlevel.env import Environments, EnvType
from tianshou.policy.modelfree.pg import TDistParams
from tianshou.utils.string import ToStringMixin

TDistributionFunction: TypeAlias = Callable[[TDistParams], torch.distributions.Distribution]


class DistributionFunctionFactory(ToStringMixin, ABC):
    @abstractmethod
    def create_dist_fn(self, envs: Environments) -> TDistributionFunction:
        pass


class DistributionFunctionFactoryCategorical(DistributionFunctionFactory):
    def create_dist_fn(self, envs: Environments) -> TDistributionFunction:
        assert envs.get_type().assert_discrete(self)
        return self._dist_fn

    @staticmethod
    def _dist_fn(p):
        return torch.distributions.Categorical(logits=p)


class DistributionFunctionFactoryIndependentGaussians(DistributionFunctionFactory):
    def create_dist_fn(self, envs: Environments) -> TDistributionFunction:
        assert envs.get_type().assert_continuous(self)
        return self._dist_fn

    @staticmethod
    def _dist_fn(*p):
        return torch.distributions.Independent(torch.distributions.Normal(*p), 1)


class DistributionFunctionFactoryDefault(DistributionFunctionFactory):
    def create_dist_fn(self, envs: Environments) -> TDistributionFunction:
        match envs.get_type():
            case EnvType.DISCRETE:
                return DistributionFunctionFactoryCategorical().create_dist_fn(envs)
            case EnvType.CONTINUOUS:
                return DistributionFunctionFactoryIndependentGaussians().create_dist_fn(envs)
            case _:
                raise ValueError(envs.get_type())
