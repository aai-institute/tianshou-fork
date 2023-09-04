import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Any, Callable, DefaultDict, Dict, Optional, Tuple, Union, \
    Sequence

import numpy as np
import tqdm

from tianshou.data import AsyncCollector, Collector, ReplayBuffer
from tianshou.policy import BasePolicy
from tianshou.trainer.utils import gather_info, test_episode
from tianshou.utils import (
    BaseLogger,
    DummyTqdm,
    LazyLogger,
    MovAvg,
    deprecation,
    tqdm_config,
)


class BaseTrainer(ABC):
    @staticmethod
    def gen_doc(learning_type: str) -> str:
        """Document string for subclass trainer."""
        step_means = f'The "step" in {learning_type} trainer means '
        if learning_type != "offline":
            step_means += "an environment step (a.k.a. transition)."
        else:  # offline
            step_means += "a gradient step."

        trainer_name = learning_type.capitalize() + "Trainer"

        return f"""An iterator class for {learning_type} trainer procedure.

        Returns an iterator that yields a 3-tuple (epoch, stats, info) of
        train results on every epoch.

        {step_means}

        Example usage:

        ::

            trainer = {trainer_name}(...)
            for epoch, epoch_stat, info in trainer:
                print("Epoch:", epoch)
                print(epoch_stat)
                print(info)
                do_something_with_policy()
                query_something_about_policy()
                make_a_plot_with(epoch_stat)
                display(info)

        - epoch int: the epoch number
        - epoch_stat dict: a large collection of metrics of the current epoch
        - info dict: result returned from :func:`~tianshou.trainer.gather_info`

        You can even iterate on several trainers at the same time:

        ::

            trainer1 = {trainer_name}(...)
            trainer2 = {trainer_name}(...)
            for result1, result2, ... in zip(trainer1, trainer2, ...):
                compare_results(result1, result2, ...)
        """

    def __init__(
        self,
        policy: BasePolicy,
        max_epoch: int,
        batch_size: int,
        train_collector: Optional[Collector] = None,
        test_collector: Optional[Collector] = None,
        buffer: Optional[ReplayBuffer] = None,
        step_per_epoch: Optional[int] = None,
        repeat_per_collect: Optional[int] = None,
        episode_per_test: Optional[int] = None,
        update_per_step: float = 1.0,
        step_per_collect: Optional[int] = None,
        episode_per_collect: Optional[int] = None,
        train_fn: Optional[Callable[[int, int], None]] = None,
        test_fn: Optional[Callable[[int, Optional[int]], None]] = None,
        stop_fn: Optional[Callable[[float], bool]] = None,
        save_best_fn: Optional[Callable[[BasePolicy], None]] = None,
        save_checkpoint_fn: Optional[Callable[[int, int, int], str]] = None,
        resume_from_log: bool = False,
        reward_metric: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        logger: BaseLogger = LazyLogger(),
        verbose: bool = True,
        show_progress: bool = True,
        test_in_train: bool = True,
        save_fn: Optional[Callable[[BasePolicy], None]] = None,
        on_step_callbacks: Sequence[Callable[[dict], None]] = (),
        final_step_callbacks: Sequence[Callable[[dict], None]] = (),

    ):
        """An iterator base class for trainers procedure.

        Returns an iterator that yields a 3-tuple (epoch, stats, info) of train results
        on every epoch.

        :param policy: an instance of the :class:`~tianshou.policy.BasePolicy` class.
        :param batch_size: the batch size of sample data, which is going to feed in
            the policy network.
        :param train_collector: the collector used for training.
        :param test_collector: the collector used for testing. If it's None,
            then no testing will be performed.
        :param max_epoch: the maximum number of epochs for training. The training
            process might be finished before reaching ``max_epoch`` if ``stop_fn``
            is set.
        :param step_per_epoch: the number of transitions collected per epoch.
        :param repeat_per_collect: the number of repeat time for policy learning,
            for example, set it to 2 means the policy needs to learn each given batch
            data twice. Only used in on-policy algorithms
        :param episode_per_test: the number of episodes for one policy evaluation.
        :param update_per_step: only used in off-policy algorithms.
            How many gradient steps to perform per step in the environment
            (i.e., per sample added to the buffer).
        :param step_per_collect: the number of transitions the collector would
            collect before the network update, i.e., trainer will collect
            "step_per_collect" transitions and do some policy network update repeatedly
            in each epoch.
        :param episode_per_collect: the number of episodes the collector would
            collect before the network update, i.e., trainer will collect
            "episode_per_collect" episodes and do some policy network update repeatedly
            in each epoch.
        :param train_fn: a hook called at the beginning of training in each
            epoch. It can be used to perform custom additional operations, with the
            signature ``f(num_epoch: int, step_idx: int) -> None``.
        :param test_fn: a hook called at the beginning of testing in each
            epoch. It can be used to perform custom additional operations, with the
            signature ``f(num_epoch: int, step_idx: int) -> None``.
        :param save_best_fn: a hook called when the undiscounted average mean
            reward in evaluation phase gets better, with the signature
            ``f(policy: BasePolicy) -> None``. It was ``save_fn`` previously.
        :param save_checkpoint_fn: a function to save training process and
            return the saved checkpoint path, with the signature ``f(epoch: int,
            env_step: int, gradient_step: int) -> str``; you can save whatever you want.
        :param resume_from_log: resume env_step/gradient_step and other metadata
            from existing tensorboard log. Default to False.
        :param stop_fn: a function with signature ``f(mean_rewards: float) ->
            bool``, receives the average undiscounted returns of the testing result,
            returns a boolean which indicates whether reaching the goal.
        :param reward_metric: a function with signature
            ``f(rewards: np.ndarray with shape (num_episode, agent_num)) -> np.ndarray
            with shape (num_episode,)``, used in multi-agent RL. We need to return a
            single scalar for each episode's result to monitor training in the
            multi-agent RL setting. This function specifies what is the desired metric,
            e.g., the reward of agent 1 or the average reward over all agents.
        :param logger: A logger that logs statistics during
            training/testing/updating. Default to a logger that doesn't log anything.
        :param verbose: whether to print the information. Default to True.
        :param show_progress: whether to display a progress bar when training.
            Default to True.
        :param test_in_train: whether to test in the training phase.
            Default to True.
        """
        if save_fn:
            deprecation(
                "save_fn in trainer is marked as deprecated and will be "
                "removed in the future. Please use save_best_fn instead."
            )
            assert save_best_fn is None
            save_best_fn = save_fn

        self.policy = policy
        self.buffer = buffer

        self.train_collector = train_collector
        self.test_collector = test_collector

        self.logger = logger
        self.start_time = time.time()
        self.stat: DefaultDict[str, MovAvg] = defaultdict(MovAvg)
        self.best_reward = 0.0
        self.best_reward_std = 0.0
        self.start_epoch = 0
        # This is only used for logging but creeps into the implementations
        # of the trainers. I believe it would be better to remove
        self.gradient_step = 0
        self.env_step = 0
        self.max_epoch = max_epoch
        self.step_per_epoch = step_per_epoch

        # either on of these two
        self.step_per_collect = step_per_collect
        self.episode_per_collect = episode_per_collect

        self.update_per_step = update_per_step
        self.repeat_per_collect = repeat_per_collect

        self.episode_per_test = episode_per_test

        self.batch_size = batch_size

        self.train_fn = train_fn
        self.test_fn = test_fn
        self.stop_fn = stop_fn
        self.save_best_fn = save_best_fn
        self.save_checkpoint_fn = save_checkpoint_fn

        self.reward_metric = reward_metric
        self.verbose = verbose
        self.show_progress = show_progress
        self.test_in_train = test_in_train
        self.resume_from_log = resume_from_log

        self.is_run = False
        self.last_rew, self.last_len = 0.0, 0

        self.epoch = self.start_epoch
        self.best_epoch = self.start_epoch
        self.stop_fn_flag = False
        self.iter_num = 0

        self.on_step_callbacks = on_step_callbacks
        self.final_step_callbacks = final_step_callbacks

    def reset(self) -> None:
        """Initialize or reset the instance to yield a new iterator from zero."""
        self.is_run = False
        self.env_step = 0
        if self.resume_from_log:
            (
                self.start_epoch,
                self.env_step,
                self.gradient_step,
            ) = self.logger.restore_data()

        self.last_rew, self.last_len = 0.0, 0
        self.start_time = time.time()
        if self.train_collector is not None:
            self.train_collector.reset_stat()

            if self.train_collector.policy != self.policy:
                self.test_in_train = False
            elif self.test_collector is None:
                self.test_in_train = False

        if self.test_collector is not None:
            assert self.episode_per_test is not None
            assert not isinstance(self.test_collector, AsyncCollector)  # Issue 700
            self.test_collector.reset_stat()
            test_result = test_episode(
                self.policy,
                self.test_collector,
                self.test_fn,
                self.start_epoch,
                self.episode_per_test,
                self.logger,
                self.env_step,
                self.reward_metric,
            )
            self.best_epoch = self.start_epoch
            self.best_reward, self.best_reward_std = (
                test_result["rew"],
                test_result["rew_std"],
            )
        if self.save_best_fn:
            self.save_best_fn(self.policy)

        self.epoch = self.start_epoch
        self.stop_fn_flag = False
        self.iter_num = 0

    def __iter__(self):  # type: ignore
        self.reset()
        return self

    def __next__(self) -> Union[None, Tuple[int, Dict[str, Any], Dict[str, Any]]]:
        """Perform one epoch (both train and eval)."""
        self.epoch += 1
        self.iter_num += 1

        if self.iter_num > 1:
            # iterator exhaustion check
            if self.epoch > self.max_epoch:
                raise StopIteration

            # exit flag 1, when stop_fn succeeds in train_step or test_step
            if self.stop_fn_flag:
                raise StopIteration

        # set policy in train mode
        self.policy.train()

        epoch_stat: Dict[str, Any] = dict()

        if self.show_progress:
            progress = tqdm.tqdm
        else:
            progress = DummyTqdm

        # perform n step_per_epoch
        with progress(
            total=self.step_per_epoch, desc=f"Epoch #{self.epoch}", **tqdm_config
        ) as t:
            while t.n < t.total and not self.stop_fn_flag:
                # TODO: data and result are very similar to each other, shouldn't return both.
                #  Maybe switch to dataclasses instead of dicts?
                data: Dict[str, Any] = dict()
                result: Dict[str, Any] = dict()
                if self.train_collector is not None:
                    # collect transitions from the training env and store them in the replay buffer
                    # policy is not updated here
                    data, result, self.stop_fn_flag = self.train_step()
                    t.update(result["n/st"])
                    if self.stop_fn_flag:
                        t.set_postfix(**data)
                        break
                else:
                    assert self.buffer, "No train_collector or buffer specified"
                    result["n/ep"] = len(self.buffer)
                    result["n/st"] = int(self.gradient_step)
                    t.update()
                for callback in self.on_step_callbacks:
                    callback({**result, **data})
                self.policy_update_fn(data, result)
                t.set_postfix(**data)

            if t.n <= t.total and not self.stop_fn_flag:
                t.update()

        # for offline RL
        if self.train_collector is None:
            self.env_step = self.gradient_step * self.batch_size

        if not self.stop_fn_flag:
            self.logger.save_data(
                self.epoch, self.env_step, self.gradient_step, self.save_checkpoint_fn
            )
            # test
            if self.test_collector is not None:
                test_stat, self.stop_fn_flag = self.test_step()
                if not self.is_run:
                    epoch_stat.update(test_stat)

        if not self.is_run:
            epoch_stat.update({k: v.get() for k, v in self.stat.items()})
            epoch_stat["gradient_step"] = self.gradient_step
            epoch_stat.update(
                {
                    "env_step": self.env_step,
                    "rew": self.last_rew,
                    "len": int(self.last_len),
                    "n/ep": int(result["n/ep"]),
                    "n/st": int(result["n/st"]),
                }
            )
            info = gather_info(
                self.start_time,
                self.train_collector,
                self.test_collector,
                self.best_reward,
                self.best_reward_std,
            )
            return self.epoch, epoch_stat, info
        else:
            return None

    def test_step(self) -> Tuple[Dict[str, Any], bool]:
        """Perform one testing step."""
        assert self.episode_per_test is not None
        assert self.test_collector is not None
        stop_fn_flag = False
        test_result = test_episode(
            self.policy,
            self.test_collector,
            self.test_fn,
            self.epoch,
            self.episode_per_test,
            self.logger,
            self.env_step,
            self.reward_metric,
        )
        rew, rew_std = test_result["rew"], test_result["rew_std"]
        if self.best_epoch < 0 or self.best_reward < rew:
            self.best_epoch = self.epoch
            self.best_reward = float(rew)
            self.best_reward_std = rew_std
            if self.save_best_fn:
                self.save_best_fn(self.policy)
        if self.verbose:
            print(
                f"Epoch #{self.epoch}: test_reward: {rew:.6f} ± {rew_std:.6f},"
                f" best_reward: {self.best_reward:.6f} ± "
                f"{self.best_reward_std:.6f} in #{self.best_epoch}",
                flush=True,
            )
        if not self.is_run:
            test_stat = {
                "test_reward": rew,
                "test_reward_std": rew_std,
                "best_reward": self.best_reward,
                "best_reward_std": self.best_reward_std,
                "best_epoch": self.best_epoch,
            }
        else:
            test_stat = {}
        if self.stop_fn and self.stop_fn(self.best_reward):
            stop_fn_flag = True

        return test_stat, stop_fn_flag

    # TODO: 1 a lot of duplication between the returned data_stats and result_stats
    # TODO: 2 rename? A bit confusing to name this "train" when policy is not updated
    def train_step(self) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
        """Perform one training-collection step.

        Collects transitions from the training environments and executes
        related hooks (like `train_fn` and `reward_metric`).
        Note: does NOT update the policy!

        :return: collected_data_stats, result_stats, stop_fn_flag
        """
        assert self.episode_per_test is not None
        assert self.train_collector is not None
        stop_fn_flag = False
        if self.train_fn:
            self.train_fn(self.epoch, self.env_step)
        result_stats = self.train_collector.collect(
            n_step=self.step_per_collect, n_episode=self.episode_per_collect
        )
        if result_stats["n/ep"] > 0 and self.reward_metric:
            rew = self.reward_metric(result_stats["rews"])
            result_stats.update(rews=rew, rew=rew.mean(), rew_std=rew.std())
        self.env_step += int(result_stats["n/st"])
        self.logger.log_train_data(result_stats, self.env_step)
        self.last_rew = (
            result_stats["rew"] if result_stats["n/ep"] > 0 else self.last_rew
        )
        self.last_len = (
            result_stats["len"] if result_stats["n/ep"] > 0 else self.last_len
        )
        collected_data_stats = {
            "env_step": str(self.env_step),
            "rew": f"{self.last_rew:.2f}",
            "len": str(int(self.last_len)),
            "n/ep": str(int(result_stats["n/ep"])),
            "n/st": str(int(result_stats["n/st"])),
        }
        if result_stats["n/ep"] > 0:
            if (
                self.test_in_train
                and self.stop_fn
                and self.stop_fn(result_stats["rew"])
            ):
                assert self.test_collector is not None
                test_result = test_episode(
                    self.policy,
                    self.test_collector,
                    self.test_fn,
                    self.epoch,
                    self.episode_per_test,
                    self.logger,
                    self.env_step,
                )
                if self.stop_fn(test_result["rew"]):
                    stop_fn_flag = True
                    self.best_reward = test_result["rew"]
                    self.best_reward_std = test_result["rew_std"]
                else:
                    self.policy.train()

        return collected_data_stats, result_stats, stop_fn_flag

    def log_update_data(self, data: Dict[str, Any], losses: Dict[str, Any]) -> None:
        """Log losses to current logger."""
        for k in losses.keys():
            self.stat[k].add(losses[k])
            losses[k] = self.stat[k].get()
            data[k] = f"{losses[k]:.3f}"
        self.logger.log_update_data(losses, self.gradient_step)

    @abstractmethod
    def policy_update_fn(self, data: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Policy update function for different trainer implementation.

        :param data: information in progress bar.
        :param result: collector's return value.
        """

    def run(self) -> Dict[str, Union[float, str]]:
        """Consume iterator.

        See itertools - recipes. Use functions that consume iterators at C speed
        (feed the entire iterator into a zero-length deque).
        """
        try:
            self.is_run = True
            deque(self, maxlen=0)  # feed the entire iterator into a zero-length deque
            info = gather_info(
                self.start_time,
                self.train_collector,
                self.test_collector,
                self.best_reward,
                self.best_reward_std,
            )
        finally:
            self.is_run = False
            for callback in self.final_step_callbacks:
                callback({**info})

        return info

    def _sample_and_update(self, buffer: ReplayBuffer, data: Dict[str, Any]) -> None:
        self.gradient_step += 1
        # Note: since sample_size=batch_size, this will perform
        # exactly one gradient step. This is why we don't need to calculate the
        # number of gradient steps, like in the on-policy case.
        losses = self.policy.update(sample_size=self.batch_size, buffer=buffer)
        data.update({"gradient_step": str(self.gradient_step)})
        self.log_update_data(data, losses)

    def _update_on_entire_buffer(
        self, buffer: ReplayBuffer, data: Dict[str, Any]
    ) -> None:
        losses = self.policy.update(
            sample_size=0,
            buffer=buffer,
            # Note: sample_size is 0, so the whole buffer is used for the update.
            # The kwargs are in the end passed to the .learn method, which uses
            # batch_size to iterate through the buffer in mini-batches
            # Off-policy algos typically don't use the batch_size kwarg at all
            batch_size=self.batch_size,
            repeat=self.repeat_per_collect,
        )

        # just for logging, no functional role
        self.gradient_step += (
            1 + (len(self.train_collector.buffer) - 0.1) // self.batch_size
        )

        # The step is the number of mini-batches used for the update, so essentially
        self.log_update_data(data, losses)


class OfflineTrainer(BaseTrainer):
    def policy_update_fn(
        self, data: Dict[str, Any], result: Optional[Dict[str, Any]] = None
    ) -> None:
        """Perform one off-line policy update."""
        assert self.buffer
        self._sample_and_update(self.buffer, data)


class OffpolicyTrainer(BaseTrainer):
    # TODO: this is very similar to the on-policy trainer, so maybe one could refactor
    #   to make the small differences more explicit?
    def policy_update_fn(self, data: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Perform off-policy updates.

        :param data:
        :param result: must contain `n/st` key, see documentation of
            `:meth:~tianshou.data.collector.Collector.collect` for the kind of
            data returned there. `n/st` stands for `step_count`
        """
        assert self.train_collector is not None
        n_collected_steps = result["n/st"]
        # Same as training intensity, right?
        num_updates = round(self.update_per_step * n_collected_steps)
        for _ in range(num_updates):
            self._sample_and_update(self.train_collector.buffer, data)


class OffpolicyFullBufferTrainer(BaseTrainer):
    """Contrary to the on-policy trainer, doesn't empty the buffer after an epoch.
    And contrary to the off-policy trainer, the updates are computed by
    applying the policy's `learn` on the full buffer instead of on samples
    from it"""

    def policy_update_fn(
        self, data: Dict[str, Any], result: Optional[Dict[str, Any]] = None
    ) -> None:
        """Perform one on-policy update."""
        assert self.train_collector is not None
        self._update_on_entire_buffer(self.train_collector.buffer, data)


class OnpolicyTrainer(BaseTrainer):
    def policy_update_fn(
        self, data: Dict[str, Any], result: Optional[Dict[str, Any]] = None
    ) -> None:
        """Perform one on-policy update."""
        assert self.train_collector is not None
        self._update_on_entire_buffer(self.train_collector.buffer, data)
        # Note: this is the main difference to the off-policy trainer!
        # The second difference is that batches of data are sampled without replacement
        # during training, whereas in off-policy or offline training, the batches are
        # sampled with replacement (and potentially custom prioritization).
        self.train_collector.reset_buffer(keep_statistics=True)


# TODO: I really don't see the point of the code below...
def offline_trainer(*args, **kwargs) -> Dict[str, Union[float, str]]:  # type: ignore
    """Wrapper for offline_trainer run method.

    It is identical to ``OfflineTrainer(...).run()``.

    :return: See :func:`~tianshou.trainer.gather_info`.
    """
    return OfflineTrainer(*args, **kwargs).run()


def offpolicy_trainer(*args, **kwargs) -> Dict[str, Union[float, str]]:  # type: ignore
    """Wrapper for OffPolicyTrainer run method.

    It is identical to ``OffpolicyTrainer(...).run()``.

    :return: See :func:`~tianshou.trainer.gather_info`.
    """
    return OffpolicyTrainer(*args, **kwargs).run()


def onpolicy_trainer(*args, **kwargs) -> Dict[str, Union[float, str]]:  # type: ignore
    """Wrapper for OnpolicyTrainer run method.

    It is identical to ``OnpolicyTrainer(...).run()``.

    :return: See :func:`~tianshou.trainer.gather_info`.
    """
    return OnpolicyTrainer(*args, **kwargs).run()


offline_trainer_iter = OfflineTrainer
offpolicy_trainer_iter = OffpolicyTrainer
onpolicy_trainer_iter = OnpolicyTrainer
