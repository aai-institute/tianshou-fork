import argparse
import os
import re

from tensorboard.backend.event_processing import event_accumulator
import tqdm

from examples.mujoco.tools import find_all_files


def plot_variance(root_dir, refresh=False):
    """Recursively convert test/reward from all tfevent file under root_dir to csv.

    This function assumes that there is at most one tfevents file in each directory
    and will add suffix to that directory.

    :param bool refresh: re-create csv file under any condition.
    """
    tfevent_files = find_all_files(root_dir, re.compile(r"^.*tfevents.*$"))
    print(f"Converting {len(tfevent_files)} tfevents files under {root_dir} ...")
    result = {}
    with tqdm.tqdm(tfevent_files) as t:
        for tfevent_file in t:
            t.set_postfix(file=tfevent_file)
            ea = event_accumulator.EventAccumulator(tfevent_file)
            ea.Reload()
            initial_time = ea._first_event_timestamp
            content = [["env_step", "reward", "time"]]
            for test_reward in ea.scalars.Items("test/reward"):
                pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--root-dir", type=str,
                        default="log/Ant-v4/sac/0/240201-142401")
    root_dir = "log/Ant-v4/sac/0/240201-142401"
    args = parser.parse_args()

    csv_files = plot_variance(args.root_dir)
