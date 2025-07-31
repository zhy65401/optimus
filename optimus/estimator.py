#!/usr/bin/env python
# Version: 0.1.0
# Created: 2024-04-07
# Author: ["Hanyuan Zhang"]

import warnings
import numpy as np
import pandas as pd
from enum import Enum

import lightgbm as lgb
import xgboost as xgb
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.base import TransformerMixin, BaseEstimator, ClassifierMixin

from .metrics import Metrics

warnings.filterwarnings("ignore")


class Estimators(Enum):
        
        LR = LogisticRegression('l2')
        
        LGBM = lgb.LGBMClassifier(
            class_weight='balanced', 
            importance_type='gain',
            n_jobs=8, 
            random_state=1024, 
            verbosity=-1
        )
        XGB = xgb.XGBClassifier(
            n_jobs=8, 
            random_state=1024,
            use_label_encoder=False
        )


class Benchmark(BaseEstimator, ClassifierMixin):
    def __init__(self, positive_coef=False, remove_method="iv", pvalue_threshold=0.05, user_feature_list=None):
        self.version = "v1.0.2"
        self.model = None
        self.model_detail = None
        self.select_detail = dict()
        self.selected_features = None
        self.removed_features = dict()
        self.coef_selector = None
        self.pvalue_selector = None
        
        self.positive_coef = positive_coef
        self.remove_method = remove_method
        self.pvalue_threshold = pvalue_threshold
        self.user_feature_list = user_feature_list

    def fit(self, X, y):
        feature_list = sorted(self.user_feature_list or X.columns.tolist())
        self.selected_features = feature_list

        # remove variable with inconsistent trend between woe and coefficient
        print("[INFO] Removing inconsistent trend features between woe and coefficient...")
        coef_selector = CoefSelector(positive_coef=self.positive_coef, remove_method=self.remove_method)
        coef_selector.fit(
            X[self.selected_features],
            y,
        )
        self.selected_features = coef_selector.selected_features
        self.coef_selector = coef_selector
        self.removed_features["by_coef"] = coef_selector.removed_features
        self.select_detail["by_coef"] = coef_selector.detail
        # print(self.selected_features)

        # remove variable with insignificant p value
        print("[INFO] Removing features with insignificant p-value...")
        pvalue_selector = PValueSelector(pvalue_threshold=self.pvalue_threshold)
        pvalue_selector.fit(
            X[self.selected_features],
            y
        )
        self.selected_features = pvalue_selector.selected_features
        self.pvalue_selector = pvalue_selector
        self.removed_features["by_pvalue"] = pvalue_selector.removed_features
        self.select_detail["by_pvalue"] = pvalue_selector.detail
        # print(self.selected_features)

        # run logit model
        model = Logit()
        model.fit(X[self.selected_features], y)

        self.model = model
        self.model_detail = model.detail
        self.summary()
        return self
    
    def transform(self, X, y=None):
        return X
    
    def predict(self, df_xtest):
        # sm.add_constant won't add a constant if there exists a column with variance 0
        df_xtest = sm.add_constant(df_xtest)
        df_xtest["const"] = 1
        return self.model.predict(df_xtest[["const"] + self.selected_features])

    def predict_proba(self, df_xtest):
        # sm.add_constant won't add a constant if there exists a column with variance 0
        df_xtest = sm.add_constant(df_xtest)
        df_xtest["const"] = 1
        yprob = self.model.predict(df_xtest[["const"] + self.selected_features])
        res = np.zeros((len(df_xtest), 2))
        res[:, 1] = yprob
        res[:, 0] = 1 - yprob
        return res

    def summary(self):
        print(f"\nRemoved {len(self.removed_features['by_pvalue'])} features by pvalue  {len(self.removed_features['by_coef'])} features by inconsistent coef, {len(self.selected_features)} remaining.\n")
        print("\n===============================================================================")


class Logit(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.version = "v1.0.2"
        self.model = None
        self.detail = None
        self.selected_features = None

    def fit(self, df_xtrain, df_ytrain):
        self.selected_features = sorted(df_xtrain.columns.tolist())

        df_xtrain_const = sm.add_constant(df_xtrain)

        # model training. default using newton method, if fail use bfgs method
        try:
            self.model = sm.Logit(df_ytrain, df_xtrain_const).fit(
                method="newton", maxiter=100
            )
        except:
            print(
                "[WARNING]: exist strong correlated features, "
                "got singular matrix in linear model, retry bfgs method instead."
            )
            self.model = sm.Logit(df_ytrain, df_xtrain_const).fit(
                method="bfgs", maxiter=100
            )

        # prepare model result
        self.detail = pd.DataFrame(
            {
                "var": df_xtrain_const.columns.tolist(),
                "coef": self.model.params,
                "std_err": [round(v, 2) for v in self.model.bse],
                "z": [round(v, 2) for v in self.model.tvalues],
                "pvalue": [round(v, 2) for v in self.model.pvalues],
            }
        )
        self.detail["std_var"] = df_xtrain.std()
        self.detail["std_var"] = self.detail["std_var"].apply(lambda x: round(x, 2))
        self.detail["feature_importance"] = (
            abs(self.detail["coef"]) * self.detail["std_var"]
        )
        self.detail["feature_importance"] = (
            self.detail["feature_importance"] / self.detail["feature_importance"].sum()
        )
        self.detail["feature_importance"] = self.detail["feature_importance"].apply(
            lambda x: round(x, 2)
        )

        return self

    def predict(self, df_xtest):
        # sm.add_constant won't add a constant if there exists a column with variance 0
        df_xtest = sm.add_constant(df_xtest)
        df_xtest["const"] = 1
        return self.model.predict(df_xtest[["const"] + self.selected_features])

    def predict_proba(self, df_xtest):
        # sm.add_constant won't add a constant if there exists a column with variance 0
        df_xtest = sm.add_constant(df_xtest)
        df_xtest["const"] = 1
        yprob = self.model.predict(df_xtest[["const"] + self.selected_features])
        res = np.zeros((len(df_xtest), 2))
        res[:, 1] = yprob
        res[:, 0] = 1 - yprob
        return res

    def summary(self):
        print(self.detail)

    def get_importance(self):
        return self.detail.drop("const", axis=0)


class CoefSelector(TransformerMixin):
    def __init__(self, positive_coef=False, remove_method="iv", user_feature_list=None):
        self.detail = None
        self.selected_features = list()
        self.removed_features = list()
        self.user_feature_list = user_feature_list
        self.positive_coef = positive_coef
        self.remove_method = remove_method

    def fit(self, df_xtrain, df_ytrain):
        feature_list = sorted(self.user_feature_list or df_xtrain.columns.tolist())
        self.selected_features = feature_list
        self.detail = list()

        if self.remove_method == "iv":
            lst_iv = [
                Metrics.get_iv(df_ytrain, df_xtrain[c]) for c in feature_list
            ]
            df_iv = pd.DataFrame({"var": feature_list, "iv": lst_iv})
            df_iv = df_iv[["var", "iv"]]

        while True:
            model = Logit()
            model.fit(df_xtrain[self.selected_features], df_ytrain)

            if self.remove_method == "feature_importance":
                df_res = model.get_importance()[
                    ["var", "coef", "pvalue", "feature_importance"]
                ]
                df_res = df_res.reset_index(drop=True)
                self.detail.append(df_res)
            else:
                df_res = model.get_importance()[["var", "coef", "pvalue"]]
                df_res = df_res.reset_index(drop=True)
                df_res = df_res.merge(df_iv, on=["var"], how="left")
                self.detail.append(df_res)

            if df_res["pvalue"].isnull().sum() != 0:
                df_remove = df_res.loc[(df_res["pvalue"].isnull()), :]
                df_remove = df_remove.sort_values(by=f"{self.remove_method}", ascending=True)
                df_remove = df_remove.reset_index(drop=True)
                remove_var = df_remove.loc[0, "var"]
                self.selected_features.remove(remove_var)
                self.removed_features.append(remove_var)
            else:
                if self.positive_coef is True:
                    df_res["coef"] = -df_res["coef"]

                df_remove = df_res.loc[(df_res["coef"] >= 0), :]
                if len(df_remove) != 0:
                    df_remove = df_remove.sort_values(
                        by=f"{self.remove_method}", ascending=True
                    )
                    df_remove = df_remove.reset_index(drop=True)
                    remove_var = df_remove.loc[0, "var"]
                    self.selected_features.remove(remove_var)
                    self.removed_features.append(remove_var)
                else:
                    break

            if len(self.selected_features) == 0:
                break

        return self

    def transform(self, df_xtest,):
        return df_xtest[self.selected_features]

    def summary(self):
        print(f"\nRemoved {len(self.removed_features)} features, {len(self.selected_features)} remaining.\n")
        print("\n===============================================================================")


class PValueSelector(TransformerMixin):
    def __init__(self, pvalue_threshold=0.05, user_feature_list=None):
        self.detail = None
        self.selected_features = list()
        self.removed_features = list()
        self.user_feature_list = user_feature_list
        self.pvalue_threshold = pvalue_threshold

    def fit(self, df_xtrain, df_ytrain):
        feature_list = sorted(self.user_feature_list or df_xtrain.columns.tolist())
        self.selected_features = feature_list
        self.detail = list()

        while True:
            model = Logit()
            model.fit(df_xtrain[self.selected_features], df_ytrain)

            df_res = model.get_importance()[["var", "coef", "pvalue"]]
            df_res = df_res.reset_index(drop=True)
            self.detail.append(df_res)

            df_remove = df_res.loc[(df_res["pvalue"] > self.pvalue_threshold), :]
            if len(df_remove) != 0:
                df_remove = df_remove.sort_values(by="pvalue", ascending=False)
                df_remove = df_remove.reset_index()
                remove_var = df_remove.loc[0, "var"]
                self.selected_features.remove(remove_var)
                self.removed_features.append(remove_var)
            else:
                break

            if len(self.selected_features) == 0:
                break
        return self

    def transform(self, df_xtest):
        return df_xtest[self.selected_features]

    def summary(self):
        print(f"\nRemoved {len(self.removed_features)} features, {len(self.selected_features)} remaining.\n")
        print("\n===============================================================================")
