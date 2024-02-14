import os
from collections import defaultdict
from typing import Callable

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from tianshou.utils import BaseLogger, logging
from tianshou.utils.logger.base import VALID_LOG_VALS_TYPE


class PandasLogger(BaseLogger):
    def __init__(
        self,
        log_dir: str,
        train_interval: int = 1000,
        test_interval: int = 1,
        update_interval: int = 1000,
        info_interval: int = 1,
    ) -> None:
        super().__init__(train_interval, test_interval, update_interval, info_interval)
        self.log_path = log_dir
        self.csv_name = os.path.join(self.log_path, "log.csv")
        self.pkl_name = os.path.join(self.log_path, "log.pkl")
        self.data = defaultdict(list)

    def write(self, step_type: str, step: int, data: dict[str, VALID_LOG_VALS_TYPE]) -> None:
        data = self.filter(data)
        if step_type.startswith('test'):
            self.data['test'].append(data)
        elif step_type.startswith('train'):
            self.data['train'].append(data)
        elif step_type.startswith('update'):
            self.data['update'].append(data)

    def save_data(self, epoch: int, env_step: int, gradient_step: int,
                  save_checkpoint_fn: Callable[[int, int, int], str] | None = None) -> None:
        # create and dump a dataframe
        for k, v in self.data.items():
            df = pd.DataFrame(v)
            df.to_csv(os.path.join(self.log_path, k + "_log.csv"), index_label="index")
            df.to_pickle(os.path.join(self.log_path, k + "_log.pkl"))

    def restore_data(self) -> tuple[int, int, int]:
        pass

    def filter(self, data: dict[str, VALID_LOG_VALS_TYPE]) -> dict[str, VALID_LOG_VALS_TYPE]:
        """Filter out numpy arrays and matplotlib figures from the data."""
        filtered_data = {}
        for key, value in data.items():
            if isinstance(value, (np.ndarray, Figure)):
                continue
            filtered_data[key] = value
        return filtered_data
