import pandas as pd
import numpy as np
import lightgbm as lgb
import hyperopt
from hyperopt import hp, tpe, STATUS_OK, space_eval, Trials
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, roc_auc_score
from lib.util import timeit, log, Config
from typing import List, Dict


@timeit
def train(X: pd.DataFrame, y: pd.Series, config: Config):
    if "leak" in config:
        return

    train_lightgbm(X, y, config)


@timeit
def predict(X: pd.DataFrame, config: Config) -> List:
    if "leak" in config:
        preds = predict_leak(X, config)
    else:
        preds = predict_lightgbm(X, config)
        if config["non_negative_target"]:
            preds = [max(0, p) for p in preds]

    return preds


@timeit
def validate(preds: pd.DataFrame, target_csv: str, mode: str) -> np.float64:
    df = pd.merge(preds, pd.read_csv(target_csv), on="line_id", left_index=True)
    score = roc_auc_score(df.target.values, df.prediction.values) if mode == "classification" else \
        np.sqrt(mean_squared_error(df.target.values, df.prediction.values))
    log("Score: {:0.4f}".format(score))
    return score


@timeit
def train_lightgbm(X: pd.DataFrame, y: pd.Series, config: Config):
    params = {
        "objective": "regression" if config["mode"] == "regression" else "binary",
        "metric": "rmse" if config["mode"] == "regression" else "auc",
        "verbosity": -1,
        "seed": 1,
    }

    X_sample, y_sample = data_sample(X, y)
    hyperparams = hyperopt_lightgbm(X_sample, y_sample, params, config)

    n_split = 4
    config["n_split"] = n_split
    kf = KFold(n_splits=n_split, random_state=2018, shuffle=True)
    config["model"] = []
    oofs = np.zeros((X.shape[0],))
    scores = []
    for i, (train_ind, test_ind) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_ind, :], X.iloc[test_ind,:]
        y_train, y_val = y[train_ind], y[test_ind]
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val)
        mdl = lgb.train({**params, **hyperparams},
                                         train_data, 3000, valid_data,
                                         early_stopping_rounds=50, verbose_eval=100)
        config["model"].append(mdl)
        oof = mdl.predict(X_val)
        oofs[test_ind] = oof
        if config["mode"] == "regression":
            score = np.sqrt(mean_squared_error(y_val, oof ))
        else:
            score = roc_auc_score(y_val,oof )
        scores.append(score)
        log(f"FOLD: {i}, Score: {round(score,2)}")
    log(f"Total score: {np.mean(scores)} , std: {np.std(scores)}")

@timeit
def predict_lightgbm(X: pd.DataFrame, config: Config) -> List:
    preds = np.zeros((config["n_split"], X.shape[0]))
    for i, mdl in enumerate(config["model"]):
        preds[i,:] = mdl.predict(X)
    return list(np.mean(preds, 0))


@timeit
def hyperopt_lightgbm(X: pd.DataFrame, y: pd.Series, params: Dict, config: Config):
    X_train, X_val, y_train, y_val = data_split(X, y, test_size=0.5)
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val)

    space = {
        "learning_rate": hp.uniform("learning_rate", 0.01, 0.05),
        "max_depth": hp.choice("max_depth", [-1, 4,  6, 10, 16]),
        "num_leaves": hp.choice("num_leaves", np.linspace(10, 200, 50, dtype=int)),
        "feature_fraction": hp.quniform("feature_fraction", 0.5, 1.0, 0.1),
        "bagging_fraction": hp.quniform("bagging_fraction", 0.5, 1.0, 0.1),
        "bagging_freq": hp.choice("bagging_freq", np.linspace(0, 50, 10, dtype=int)),
        "reg_alpha": hp.uniform("reg_alpha", 0, 30),
        "reg_lambda": hp.uniform("reg_lambda", 0, 30),
        "min_child_weight": hp.uniform('min_child_weight', 0.5, 50),
    }

    def objective(hyperparams):
        model = lgb.train({**params, **hyperparams}, train_data, 300, valid_data,
                          early_stopping_rounds=100, verbose_eval=100)

        score = model.best_score["valid_0"][params["metric"]]
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
def predict_leak(X: pd.DataFrame, config: Config) -> List:
    preds = pd.Series(0, index=X.index)

    for name, group in X.groupby(by=config["leak"]["id_col"]):
        gr = group.sort_values(config["leak"]["dt_col"])
        preds.loc[gr.index] = gr[config["leak"]["num_col"]].shift(config["leak"]["lag"])

    return preds.fillna(0).tolist()


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
