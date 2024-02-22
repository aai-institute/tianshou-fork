import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict

import numpy as np
from matplotlib import pyplot as plt

from tianshou.data import (
    Collector,
    CollectStats,
    InfoStats,
    SequenceSummaryStats,
    TimingStats,
)
from tianshou.policy import BasePolicy
from tianshou.utils import BaseLogger


def create_episode_return_plots(returns: np.array, ep_ret_per_env: dict[str, list[float]]):

    env_ids = list(ep_ret_per_env.keys())
    env_num = len(env_ids)
    exp_per_env = [len(ep_ret_per_env[i]) for i in env_ids]

    # Combine all entries and subsets for plotting
    data = [returns] + [sub_ret for sub_ret in ep_ret_per_env.values()]
    # print(data)
    # Create x-tick labels
    xtick_labels = ['All Entries'] + [f'Subset {env}' for env in env_ids]

    # Check if the dimensions are compatible
    if len(data) != len(xtick_labels):
        print(f"{len(data), {len(xtick_labels)} }")
        raise ValueError(
            "Dimensions of data and xtick_labels are not compatible.")

    # Create a box plot for all entries and subsets
    fig, ax = plt.subplots()
    ax.boxplot(data, labels=xtick_labels)
    ax.tick_params(axis='x', labelrotation=45)
    ax.set_xlabel('Data Subset')
    ax.set_ylabel('Values')
    ax.set_title(
        f'Box Plots of Eval Returns {env_num} tenv a {exp_per_env} exp/tenv')

    fig2, ax2 = plt.subplots()
    ax2.errorbar(range(len(data)), [np.mean(d) for d in data], [np.std(d) for d in data], fmt='o')
    ax2.set_xticks(range(len(data)), xtick_labels, rotation=45)
    ax2.set_xlabel('Data Subset')
    ax2.set_ylabel('Return Mean and Std')
    ax2.set_title(
        f'Mean and Std of Eval Returns {env_num} tenv a {exp_per_env}  exp/tenv')

    return fig, fig2


def test_episode(
    policy: BasePolicy,
    collector: Collector,
    test_fn: Callable[[int, int | None], None] | None,
    epoch: int,
    n_episode: int,
    logger: BaseLogger | None = None,
    global_step: int | None = None,
    reward_metric: Callable[[np.ndarray], np.ndarray] | None = None,
) -> CollectStats:
    """A simple wrapper of testing policy in collector."""
    collector.reset_env()
    collector.reset_buffer()
    policy.eval()
    if test_fn:
        test_fn(epoch, global_step)
    result = collector.collect(n_episode=n_episode)
    if reward_metric:  # TODO: move into collector
        rew = reward_metric(result.returns)
        result.returns = rew
        result.returns_stat = SequenceSummaryStats.from_sequence(rew)
    if logger and global_step is not None:
        assert result.n_collected_episodes > 0
        result_dict = asdict(result)
        if 11 < 3:
            fig1, fig2 = create_episode_return_plots(result_dict['returns'], result_dict['episode_returns_per_env'])
            result_dict.update({"episode_return_boxplot": fig1,
                                "episode_return_error_bar": fig2})
        logger.log_test_data(result_dict, global_step)
        if 11 < 3:
            plt.close(fig1)
            plt.close(fig2)
    return result


def gather_info(
    start_time: float,
    policy_update_time: float,
    gradient_step: int,
    best_reward: float,
    best_reward_std: float,
    train_collector: Collector | None = None,
    test_collector: Collector | None = None,
) -> InfoStats:
    """A simple wrapper of gathering information from collectors.

    :return: A dataclass object with the following members (depending on available collectors):

        * ``gradient_step`` the total number of gradient steps;
        * ``best_reward`` the best reward over the test results;
        * ``best_reward_std`` the standard deviation of best reward over the test results;
        * ``train_step`` the total collected step of training collector;
        * ``train_episode`` the total collected episode of training collector;
        * ``test_step`` the total collected step of test collector;
        * ``test_episode`` the total collected episode of test collector;
        * ``timing`` the timing statistics, with the following members:
        * ``total_time`` the total time elapsed;
        * ``train_time`` the total time elapsed for learning training (collecting samples plus model update);
        * ``train_time_collect`` the time for collecting transitions in the \
            training collector;
        * ``train_time_update`` the time for training models;
        * ``test_time`` the time for testing;
        * ``update_speed`` the speed of updating (env_step per second).
    """
    duration = max(0.0, time.time() - start_time)
    test_time = 0.0
    update_speed = 0.0
    train_time_collect = 0.0
    if test_collector is not None:
        test_time = test_collector.collect_time

    if train_collector is not None:
        train_time_collect = train_collector.collect_time
        update_speed = train_collector.collect_step / (duration - test_time)

    timing_stat = TimingStats(
        total_time=duration,
        train_time=duration - test_time,
        train_time_collect=train_time_collect,
        train_time_update=policy_update_time,
        test_time=test_time,
        update_speed=update_speed,
    )

    return InfoStats(
        gradient_step=gradient_step,
        best_reward=best_reward,
        best_reward_std=best_reward_std,
        train_step=train_collector.collect_step if train_collector is not None else 0,
        train_episode=train_collector.collect_episode if train_collector is not None else 0,
        test_step=test_collector.collect_step if test_collector is not None else 0,
        test_episode=test_collector.collect_episode if test_collector is not None else 0,
        timing=timing_stat,
    )
