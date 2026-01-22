#!/usr/bin/env python
# Version: 0.4.0
# Created: 2024-04-07
# Author: ["Hanyuan Zhang"]

from enum import Enum
from itertools import product

import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.model_selection import StratifiedKFold

from .estimator import Estimators
from .metrics import Metrics


class _DefaultGSParameterGrid(Enum):
    LR = {"C": np.arange(0.01, 1, 0.01)}
    XGB = {
        "n_estimators": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "max_depth": [3, 4],
        "max_leaves": [5, 7, 9, 11, 13, 15],
        "learning_rate": np.arange(0.01, 0.2, 0.01),
        "colsample_bytree": np.arange(0.7, 1, 0.01),
        "min_child_weight": np.arange(0.7, 1, 0.01),
        "reg_alpha": np.arange(0, 10, 1),
        "reg_lambda": np.arange(0, 10, 1),
        "scale_pos_weight": [1],
    }
    LGBM = {
        "n_estimators": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "max_depth": [3, 4],
        "max_leaves": [5, 7, 9, 11, 13, 15],
        "min_child_samples": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "learning_rate": np.arange(0.01, 0.2, 0.01),
        "reg_alpha": np.arange(0, 10, 1),
        "reg_lambda": np.arange(0, 10, 1),
    }


class _DefaultBOParameterGrid(Enum):
    LR = {
        "C": hp.choice("C", np.arange(0.01, 1, 0.01)),
    }
    XGB = {
        "n_estimators": hp.choice(
            "n_estimators", [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        ),
        "max_depth": hp.choice("max_depth", [3, 4]),
        "max_leaves": hp.choice("max_leaves", [5, 7, 9, 11, 13, 15]),
        "learning_rate": hp.uniform("learning_rate", 0.01, 0.2),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.7, 1),
        "min_child_weight": hp.uniform("min_child_weight", 0.7, 1),
        "reg_alpha": hp.uniform("reg_alpha", 0, 10),
        "reg_lambda": hp.uniform("reg_lambda", 0, 10),
        "scale_pos_weight": hp.choice("scale_pos_weight", [1]),
    }
    LGBM = {
        "n_estimators": hp.choice(
            "n_estimators", [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        ),
        "max_depth": hp.choice("max_depth", [3, 4]),
        "num_leaves": hp.choice("num_leaves", [5, 7, 9, 11, 13, 15]),
        "min_child_samples": hp.choice(
            "min_child_samples", [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        ),
        "learning_rate": hp.uniform("learning_rate", 0.01, 0.2),
        "reg_alpha": hp.uniform("reg_alpha", 0, 10),
        "reg_lambda": hp.uniform("reg_lambda", 0, 10),
    }


class GridSearch:
    def __init__(self, model_type, param_grid=None):
        self.model_type = model_type
        self.best_params = None
        self.best_estimator = None
        self.results = list()
        self.param_grid = param_grid

    def fit(self, X, y, Xtest, ytest):
        param_grid = self.param_grid or _DefaultGSParameterGrid[self.model_type].value
        lst_params = self.generate_param_grid(param_grid)

        for param_grid in lst_params:
            model = Estimators[self.model_type].value
            model.set_params(**param_grid)
            cv_res = self.get_cv_results(model, X, y)
            model, train_res = self.get_results(
                model, self.model_type, X, y, Xtest, ytest
            )

            res = param_grid
            res.update(cv_res)
            res.update(train_res)
            self.results.append(res)

            print("")
            print(
                "\t".join(
                    [
                        f"{k}: {v}"
                        for k, v in param_grid.items()
                        if "auc" not in k and "ks" not in k
                    ]
                )
            )
            print(
                f"AUC: Train: {train_res['train_auc'][0]:.2%} Test: {train_res['test_auc'][0]:.2%} OOT: {train_res['oot_auc'][0]:.2%}"
            )
            print(
                f"KS: Train: {train_res['train_ks'][0]:.2%} Test: {train_res['test_ks'][0]:.2%} OOT: {train_res['oot_ks'][0]:.2%}"
            )
            print(
                "==============================================================================="
            )

            if self.best_params is None:
                self.best_params = param_grid
                self.best_estimator = model
            elif res["cv_valid_avg_auc"] == max(
                [res["cv_valid_avg_auc"] for res in self.results]
            ):
                self.best_params = param_grid
                self.best_estimator = model
            else:
                continue

        self.results = pd.concat(
            [pd.DataFrame(res) for res in self.results], axis=0
        ).reset_index(drop=True)
        return self

    @classmethod
    def generate_param_grid(cls, param_grid):
        if isinstance(param_grid, dict):
            param_grid = [param_grid]

        lst_params = list()
        for p in param_grid:
            items = sorted(p.items())
            keys, values = zip(*items)

            for v in product(*values):
                params = dict(zip(keys, v))
                lst_params.append(params)
        return lst_params

    @classmethod
    def get_cv_results(cls, model, X, y):
        cv_res = dict()

        lst_train_auc, lst_train_ks, lst_valid_auc, lst_valid_ks = (
            list(),
            list(),
            list(),
            list(),
        )
        kf = StratifiedKFold(n_splits=5, random_state=1024, shuffle=True)
        for idx_train, idx_valid in kf.split(X, y):
            X_train = X.iloc[idx_train, :]
            y_train = y.iloc[idx_train]
            X_valid = X.iloc[idx_valid, :]
            y_valid = y.iloc[idx_valid]

            model.fit(X_train, y_train)
            df_ypred_train = model.predict_proba(X_train)[:, 1]
            df_ypred_valid = model.predict_proba(X_valid)[:, 1]

            lst_train_auc.append(Metrics.get_auc(y_train, df_ypred_train))
            lst_train_ks.append(Metrics.get_ks(y_train, df_ypred_train))
            lst_valid_auc.append(Metrics.get_auc(y_valid, df_ypred_valid))
            lst_valid_ks.append(Metrics.get_ks(y_valid, df_ypred_valid))

        cv_res["cv_train_auc"] = [[round(x, 4) for x in lst_train_auc]]
        cv_res["cv_valid_auc"] = [[round(x, 4) for x in lst_valid_auc]]
        cv_res["cv_train_ks"] = [[round(x, 4) for x in lst_train_ks]]
        cv_res["cv_valid_ks"] = [[round(x, 4) for x in lst_valid_ks]]

        cv_res["cv_train_avg_auc"] = [np.mean(lst_train_auc)]
        cv_res["cv_valid_avg_auc"] = [np.mean(lst_valid_auc)]
        cv_res["cv_train_avg_ks"] = [np.mean(lst_train_ks)]
        cv_res["cv_valid_avg_ks"] = [np.mean(lst_valid_ks)]
        cv_res["cv_ks_gap"] = [abs(np.mean(lst_valid_ks) - np.mean(lst_train_ks))]

        return cv_res

    @classmethod
    def get_results(cls, model, X, y, Xtest, ytest):
        res = {
            "train_auc": list(),
            "test_auc": list(),
            "train_ks": list(),
            "test_ks": list(),
            "ks_gap": list(),
        }

        model.fit(X, y)
        df_ypred_train = model.predict_proba(X)[:, 1]
        df_ypred_test = model.predict_proba(Xtest)[:, 1]

        res["train_auc"].append(Metrics.get_auc(y, df_ypred_train))
        res["test_auc"].append(Metrics.get_auc(ytest, df_ypred_test))
        res["train_ks"].append(Metrics.get_ks(y, df_ypred_train))
        res["test_ks"].append(Metrics.get_ks(ytest, df_ypred_test))
        res["ks_gap"].append(
            abs(
                Metrics.get_ks(ytest, df_ypred_test) - Metrics.get_ks(y, df_ypred_train)
            )
        )

        return model, res


class BO:
    def __init__(self, model_type, max_evals, param_grid=None):
        self.best_params = None
        self.best_estimator = None
        self.model_type = model_type
        self.results = list()
        self.iter = 0
        self.max_evals = max_evals
        self.param_grid = param_grid

    def fit(self, X, y, Xtest, ytest):
        self.X = X
        self.y = y
        self.Xtest = Xtest
        self.ytest = ytest
        param_grid = self.param_grid or _DefaultBOParameterGrid[self.model_type].value

        fmin(
            fn=self.objective,
            space=param_grid,
            algo=tpe.suggest,
            max_evals=self.max_evals,
            trials=Trials(),
            rstate=np.random.default_rng(1024),
        )
        self.results = pd.concat(
            [pd.DataFrame(res) for res in self.results], axis=0
        ).reset_index(drop=True)

        # Retrain the best model with best_params to ensure consistency
        if self.best_params is not None:
            model = Estimators[self.model_type].value
            model.set_params(**self.best_params)
            model.fit(X, y)
            self.best_estimator = model

        return self

    def objective(self, params):
        self.iter += 1
        model = Estimators[self.model_type].value
        model.set_params(**params)
        cv_res = self.get_cv_results(model, self.X, self.y)
        model, train_res = self.get_results(
            model, self.X, self.y, self.Xtest, self.ytest
        )

        search_info = {
            "iteration": self.iter,
            "loss": (-1) * cv_res["cv_valid_avg_auc"][0],
            "status": STATUS_OK,
        }
        search_info.update(params)
        search_info.update(cv_res)
        search_info.update(train_res)
        self.results.append(search_info)
        print(params, train_res)

        if self.best_params is None:
            self.best_params = params
            self.best_estimator = model
        elif search_info["cv_valid_avg_auc"] == max(
            [res["cv_valid_avg_auc"] for res in self.results]
        ):
            self.best_params = params
            self.best_estimator = model

        return search_info

    @classmethod
    def get_cv_results(cls, model, X, y):
        cv_res = dict()

        lst_train_auc, lst_train_ks, lst_valid_auc, lst_valid_ks = (
            list(),
            list(),
            list(),
            list(),
        )
        kf = StratifiedKFold(n_splits=5, random_state=1024, shuffle=True)
        for idx_train, idx_valid in kf.split(X, y):
            X_train = X.iloc[idx_train, :]
            y_train = y.iloc[idx_train]
            X_valid = X.iloc[idx_valid, :]
            y_valid = y.iloc[idx_valid]

            model.fit(X_train, y_train)
            df_ypred_train = model.predict_proba(X_train)[:, 1]
            df_ypred_valid = model.predict_proba(X_valid)[:, 1]

            lst_train_auc.append(Metrics.get_auc(y_train, df_ypred_train))
            lst_train_ks.append(Metrics.get_ks(y_train, df_ypred_train))
            lst_valid_auc.append(Metrics.get_auc(y_valid, df_ypred_valid))
            lst_valid_ks.append(Metrics.get_ks(y_valid, df_ypred_valid))

        cv_res["cv_train_auc"] = [[round(x, 4) for x in lst_train_auc]]
        cv_res["cv_train_ks"] = [[round(x, 4) for x in lst_train_ks]]
        cv_res["cv_valid_auc"] = [[round(x, 4) for x in lst_valid_auc]]
        cv_res["cv_valid_ks"] = [[round(x, 4) for x in lst_valid_ks]]

        cv_res["cv_train_avg_auc"] = [np.mean(lst_train_auc)]
        cv_res["cv_train_avg_ks"] = [np.mean(lst_train_ks)]
        cv_res["cv_valid_avg_auc"] = [np.mean(lst_valid_auc)]
        cv_res["cv_valid_avg_ks"] = [np.mean(lst_valid_ks)]

        return cv_res

    @classmethod
    def get_results(cls, model, X, y, Xtest, ytest):
        res = {
            "train_auc": list(),
            "test_auc": list(),
            "train_ks": list(),
            "test_ks": list(),
            "ks_gap": list(),
        }

        model.fit(X, y)
        df_ypred_train = model.predict_proba(X)[:, 1]
        df_ypred_test = model.predict_proba(Xtest)[:, 1]

        res["train_auc"].append(Metrics.get_auc(y, df_ypred_train))
        res["train_ks"].append(Metrics.get_ks(y, df_ypred_train))
        res["test_auc"].append(Metrics.get_auc(ytest, df_ypred_test))
        res["test_ks"].append(Metrics.get_ks(ytest, df_ypred_test))
        res["ks_gap"].append(
            abs(
                Metrics.get_ks(ytest, df_ypred_test) - Metrics.get_ks(y, df_ypred_train)
            )
        )

        return model, res


def main():
    pass


if __name__ == "__main__":
    main()
