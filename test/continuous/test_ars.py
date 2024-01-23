import argparse
import os
import pprint
import datetime

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer, ReplayBuffer
from tianshou.env import DummyVectorEnv, VectorEnvNormObs
from tianshou.policy import ARSPolicy
from tianshou.trainer import PopulationBasedTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Linear


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Hopper-v4")
    parser.add_argument("--reward-threshold", type=float, default=1000)
    parser.add_argument("--seed", type=int, default=1331)
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--episode-per-collect", type=int, default=1)
    parser.add_argument("--training-num", type=int, default=1)
    parser.add_argument("--test-num", type=int, default=4)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.01)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-interval", type=int, default=4)
    # ars special
    parser.add_argument("--sigma", type=float, default=0.02)
    parser.add_argument("--n_delta", type=float, default=10)
    parser.add_argument("--n_top", type=float, default=5)
    return parser.parse_known_args()[0]


def test_ars(args=get_args()):
    env = gym.make(args.task, render_mode="human")
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]

    # We use VectorEnvNormObs to realize observation normalization for ARS V2 (see line 8) and set healthy_reward=0
    # to exclude the survival bonus of the task (if it exists).
    try:
        train_envs = VectorEnvNormObs(DummyVectorEnv([lambda: gym.make(args.task, healthy_reward=0) for _ in range(args.training_num)]))
    except TypeError:
        train_envs = VectorEnvNormObs(
            DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.training_num)]))

    test_envs = VectorEnvNormObs(DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.test_num)]), update_obs_rms=False)
    test_envs.set_obs_rms(train_envs.get_obs_rms())

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # model
    actor = Linear(args.state_shape[0],
                   output_dim=args.action_shape[0],
                   bias=False,
                   activation=None,
                   device=args.device)

    # zero initialization
    for p in list(actor.parameters()):
        torch.nn.init.zeros_(p)

    # within the ARS algorithm, the gradient is provided manually
    optim = torch.optim.SGD(actor.parameters(), lr=args.lr)

    if args.n_top:
        assert args.n_top < args.n_delta, "n_top should be less than n_delta"

    policy = ARSPolicy(actor=actor,
                       optim=optim,
                       n_top=args.n_top,
                       action_space=env.action_space,
                       observation_space=env.observation_space,
                       action_scaling=args.max_action,
                       action_bound_method="clip",
                       )

    # collector
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
    )
    test_collector = Collector(policy, test_envs)
    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.algo_name = "ars"
    log_name = os.path.join(args.task, args.algo_name, str(args.seed), now)
    log_path = os.path.join(args.logdir, log_name)
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer, save_interval=args.save_interval)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards):
        return mean_rewards >= args.reward_threshold

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        ckpt_path = os.path.join(log_path, "checkpoint.pth")
        # Example: saving by epoch num
        # ckpt_path = os.path.join(log_path, f"checkpoint_{epoch}.pth")
        torch.save(
            {
                "model": policy.state_dict(),
                "optim": optim.state_dict(),
            },
            ckpt_path,
        )
        return ckpt_path

    if args.resume:
        # load from existing checkpoint
        print(f"Loading agent under {log_path}")
        ckpt_path = os.path.join(log_path, "checkpoint.pth")
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=args.device)
            policy.load_state_dict(checkpoint["model"])
            optim.load_state_dict(checkpoint["optim"])
            print("Successfully restore policy and optim.")
        else:
            print("Fail to restore policy and optim.")

    trainer = PopulationBasedTrainer(
        policy,
        buffer=ReplayBuffer(20, len(train_envs)),
        n_delta=args.n_delta,
        sigma=args.sigma,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        train_collector=train_collector,
        test_collector=test_collector,
        episode_per_test=args.test_num,
        episode_per_collect=args.episode_per_collect,
        train_fn=None,
        test_fn=None,
        stop_fn=stop_fn,
        save_fn=save_best_fn,
        logger=logger,
        verbose=True,
    )

    for epoch_stat in trainer:
        print(f"Epoch: {epoch_stat.epoch}")
        print(epoch_stat)
        # print(info)

    assert stop_fn(epoch_stat.info_stat.best_reward)

    if __name__ == "__main__":
        pprint.pprint(epoch_stat)
        # Let's watch its performance!
        env = VectorEnvNormObs(DummyVectorEnv([lambda: gym.make(args.task, render_mode="human") for _ in range(1)]), update_obs_rms=False)
        env.set_obs_rms(train_envs.get_obs_rms())
        policy.eval()
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1, render=args.render)
        print(f"Final reward: {result.returns_stat.mean}, length: {result.lens_stat.mean}")


if __name__ == "__main__":
    test_ars()
