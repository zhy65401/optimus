#!/usr/bin/env python
# Version: 0.1.0
# Created: 2024-04-07
# Author: ["Hanyuan Zhang"]

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


class Metrics:
    @classmethod
    def get_auc(cls, ytrue, yprob, **kwargs):
        auc = roc_auc_score(ytrue, yprob)

        if kwargs.get("symmetry", False) is True:
            if auc < 0.5:
                auc = 1 - auc
        return auc

    @classmethod
    def get_ks(cls, ytrue, yprob):
        fpr, tpr, thr = roc_curve(ytrue, yprob)
        ks = max(abs(tpr - fpr))
        return ks

    @classmethod
    def get_gini(cls, ytrue, yprob, **kwargs):
        auc = cls.get_auc(ytrue, yprob, **kwargs)
        gini = 2 * auc - 1

        return gini

    @classmethod
    def get_stat(cls, df_label, df_feature):
        var = df_feature.name
        df_data = pd.DataFrame({"val": df_feature, "label": df_label})

        # statistics of total count, total ratio, bad count, bad rate
        df_stat = df_data.groupby("val").agg(
            total=("label", "count"), bad=("label", "sum"), bad_rate=("label", "mean")
        )
        df_stat["var"] = var
        df_stat["good"] = df_stat["total"] - df_stat["bad"]
        df_stat["total_ratio"] = df_stat["total"] / df_stat["total"].sum()
        df_stat["good_density"] = df_stat["good"] / df_stat["good"].sum()
        df_stat["bad_density"] = df_stat["bad"] / df_stat["bad"].sum()

        eps = np.finfo(np.float32).eps
        df_stat.loc[:, "iv"] = (
            df_stat["bad_density"] - df_stat["good_density"]
        ) * np.log((df_stat["bad_density"] + eps) / (df_stat["good_density"] + eps))

        cols = ["var", "total", "total_ratio", "bad", "bad_rate", "iv", "val"]
        df_stat = df_stat.reset_index()[cols].set_index("var")
        return df_stat

    @classmethod
    def get_iv(cls, df_label, df_feature):
        df_stat = cls.get_stat(df_label, df_feature)
        return df_stat["iv"].sum()

    @classmethod
    def get_psi(cls, df_train, df_test):
        base_prop = pd.Series(df_train).value_counts(
            normalize=True, dropna=False
        )  # 基准数据集
        test_prop = pd.Series(df_test).value_counts(
            normalize=True, dropna=False
        )  # 测试数据集

        return np.sum((test_prop - base_prop) * np.log(test_prop / base_prop))
