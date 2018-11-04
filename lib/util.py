import os
import time
import pickle
from typing import Any

nesting_level = 0
is_start = None

def timeit(method):
    def timed(*args, **kw):
        global is_start
        global nesting_level

        if not is_start:
            print()

        is_start = True

        log("Start {}.".format(method.__name__))
        nesting_level += 1

        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()

        nesting_level -= 1
        log("End {}. Time: {:0.2f} sec.".format(method.__name__, end_time - start_time))
        is_start = False

        return result

    return timed


def log(entry: Any):
    global nesting_level
    space = "." * (4 * nesting_level)
    print("{}{}".format(space, entry))


class Config:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.tmp_dir = model_dir
        self.data = {
            "n_split_lgb": 4,
            "n_split_xgb": 4,
            "n_split_ctb": 5,
            "train_xgb": True,
            "xgb_min_time_allowance": 60,
            "ctb_min_time_allowance": 60,
            "lgb_min_time_allowance_pct": 0.6,
            "xgb_max_time_allowance_pct": 0.2,
            "ctb_max_time_allowance_pct": 0.2,
            "start_time": time.time(),
            "time_limit": int(os.environ.get("TIME_LIMIT", 5 * 60)),
            "train_h2o": False,
            "h2o_max_rows": 300000,
            "h2o_max_cols": 100,
            "h2o_min_time_allowance": 60,
            "h2o_time_coeff": 0.7,
            "h2o_max_time_allowance": 600,
            "other_time_allowance": 20,
            "lgbm_weight": 0.6,
            "h2o_weight": 0.15,
            "xgb_weight": 0.4,
            "h2o_trained": False,
            "iter_time_coeff": 1.15
        }

    def is_train(self) -> bool:
        return self["task"] == "train"

    def is_predict(self) -> bool:
        return self["task"] == "predict"

    def is_regression(self) -> bool:
        return self["mode"] == "regression"

    def is_classification(self) -> bool:
        return self["mode"] == "classification"

    def time_left(self):
        return self["time_limit"] - (time.time() - self["start_time"])

    def save(self):
        with open(os.path.join(self.model_dir, "config.pkl"), "wb") as f:
            pickle.dump(self.data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self):
        with open(os.path.join(self.model_dir, "config.pkl"), "rb") as f:
            data = pickle.load(f)

        self.data = {**data, **self.data}

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        del self.data[key]

    def __contains__(self, key):
        return key in self.data

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return repr(self.data)
