from test.highlevel.env_factory import ContinuousTestEnvFactory

import pytest

from tianshou.highlevel.config import RLSamplingConfig
from tianshou.highlevel.experiment import (
    A2CExperimentBuilder,
    DDPGExperimentBuilder,
    PPOExperimentBuilder,
    RLExperimentConfig,
    SACExperimentBuilder,
    TD3ExperimentBuilder,
)


@pytest.mark.parametrize(
    "builder_cls",
    [
        PPOExperimentBuilder,
        A2CExperimentBuilder,
        SACExperimentBuilder,
        DDPGExperimentBuilder,
        TD3ExperimentBuilder,
    ],
)
def test_experiment_builder_continuous_default_params(builder_cls):
    env_factory = ContinuousTestEnvFactory()
    sampling_config = RLSamplingConfig(num_epochs=1, step_per_epoch=100)
    experiment_config = RLExperimentConfig()
    builder = builder_cls(
        experiment_config=experiment_config,
        env_factory=env_factory,
        sampling_config=sampling_config,
    )
    experiment = builder.build()
    experiment.run("test")
    print(experiment)
