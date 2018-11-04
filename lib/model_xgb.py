import pandas as pd
import numpy as np
from lib.util import timeit, log, Config
import hyperopt
from hyperopt import hp, tpe, STATUS_OK, space_eval, Trials

#from .model import data_sample, data_split

from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split, KFold
from typing import List, Dict
import xgboost as xgb
import time


@timeit
def train_xgboost(X: pd.DataFrame, y: pd.Series, config: Config):
    
    params = {"objective" : "reg:linear" if config["mode"] == "regression" else "binary:logistic",
              "eval_metric": "rmse" if config["mode"] == "regression" else "auc",
              "learing_rate" : 0.1, "silent": 1,
            #nthread=4,
            "seed":2018
    }

    X_sample, y_sample = data_sample(X, y)
    hyperparams = hyperopt_xgboost(X_sample, y_sample, params, config)

    n_split = config["n_split_xgb"]
    kf = KFold(n_splits=n_split, random_state=2018, shuffle=True)
    config["model"] = []
    oofs = np.zeros((X.shape[0],))
    scores = []

    iter_time = 0
    iter_times = []
    config["xgb_models"] = []

    for i, (train_ind, test_ind) in enumerate(kf.split(X)):
        time_spent = (time.time() - config["start_time"])

        time_left = max(0, (config["time_limit"] - time_spent))

        # reserving time for h2o if needed
        #if config["train_h2o"] and (time_left > time_reserved): time_left = max(0, time_left - time_reserved)

        max_iter_time = max(iter_times) if len(iter_times) > 0 else 0
        #assume iterations take same time. if no time left, break
        if max_iter_time * config["iter_time_coeff"] > time_left:
            break



        iter_start  = time.time()

        X_train, X_val = X.iloc[train_ind, :], X.iloc[test_ind,:]
        y_train, y_val = y[train_ind], y[test_ind]

        train_data = xgb.DMatrix(X_train, label = y_train)
        test_data = xgb.DMatrix(X_val, label = y_val)

        watchlist = [(train_data, "train"), (test_data, "test")]

        params["learing_rate"] = 0.01

        mdl = xgb.train({**params, **hyperparams},
                        train_data,
                        evals = watchlist,
                        num_boost_round=3000,
                        early_stopping_rounds=100,
                        verbose_eval=100
                        )

        mdl_fname = (f"predictions/xgb_model_fold_{i}.xgb")
        config["model"].append(mdl_fname)

        mdl.save_model(mdl_fname)

        oof = mdl.predict(test_data)
        oofs[test_ind] = oof
        if config["mode"] == "regression":
            score = np.sqrt(mean_squared_error(y_val, oof ))
        else:
            score = roc_auc_score(y_val,oof )
        scores.append(score)
        iter_time = time.time() - iter_start
        iter_times.append(iter_time)
        log(f"FOLD: {i}, Score: {round(score,2)} , time: {iter_time:.2f}")

    log(f"Total score: {np.mean(scores)} , std: {np.std(scores)}")



@timeit
def hyperopt_xgboost(X: pd.DataFrame, y: pd.Series, params: Dict, config: Config):
    X_train, X_val, y_train, y_val = data_split(X, y, test_size=0.5)

    train_data = xgb.DMatrix(X_train, label=y_train)
    test_data = xgb.DMatrix(X_val, label=y_val)

    space = {
        "max_depth": hp.choice("max_depth", [4,5,6]),
        "min_child_weight": hp.choice("min_child_weight", [4, 8, 12, 16]),
        "gamma": hp.quniform("gamma", 0.1, 0.5, 0.1),
        "subsample": hp.choice("subsample", [i/10.0 for i in range(6,10)]),
        "colsample_bytree": hp.choice("colsample_bytree", [i/10.0 for i in range(6,10)]),
        "reg_alpha": hp.choice("reg_alpha", [0, 0.001, 0.005, 0.01, 0.05]),
    }

    def objective(hyperparams):
        watchlist = [(train_data, "train"), (test_data, "test")]

        mdl = xgb.train({**params, **hyperparams},
                        train_data,
                        evals = watchlist,
                        num_boost_round=300,
                        early_stopping_rounds=100,
                        verbose_eval=100
                        )

        score = mdl.best_score

        if config.is_classification():
            score = -score

        return {'loss': score, 'status': STATUS_OK}

    trials = Trials()
    best = hyperopt.fmin(fn=objective, space=space, trials=trials, algo=tpe.suggest, max_evals=50, verbose=1,
                         rstate=np.random.RandomState(1))

    hyperparams = space_eval(space, best)
    log("{:0.4f} {}".format(trials.best_trial['result']['loss'], hyperparams))
    return hyperparams

@timeit
def predict_xgboost(X: pd.DataFrame, config: Config) -> List:
    preds = np.zeros((config["n_split_xgb"], X.shape[0]))

    for i, mdl_fname in enumerate(config["xgb_models"]):
        mdl = xgb.Booster({'nthread': 4})
        mdl = xgb.load_model(mdl_fname)


        preds[i,:] = mdl.predict(xgb.DMatrix(X), ntree_limit = mdl.best_ntree_limit)
    return list(np.mean(preds, 0))


def data_split(X: pd.DataFrame, y: pd.Series, test_size: float=0.2) -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
    return train_test_split(X, y, test_size=test_size, random_state=1)


def data_sample(X: pd.DataFrame, y: pd.Series, nrows: int=5000) -> (pd.DataFrame, pd.Series):
    if len(X) > nrows:
        X_sample = X.sample(nrows, random_state=1)
        y_sample = y[X_sample.index]
    else:
        X_sample = X
        y_sample = y

    return X_sample, y_sample