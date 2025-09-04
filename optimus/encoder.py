#!/usr/bin/env python
# Version: 0.2.0
# Created: 2024-04-07
# Last Modified: 2025-08-19
# Author: ["Hanyuan Zhang"]

import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from termcolor import cprint

from .binner import BestKSCut, ChiMergeCut, OptimalCut, QCut, SimpleCut, WOEMerge


class Encoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        spec: dict,
        missing_values: list = None,
        treat_missing: str = "mean",
        keep_dtypes=False,
        copy=True,
    ):
        """
        Args:
            spec (dict): A dictionary for features and binning strategy. The strategy should be a list or
                one of ('auto', 'qcut', 'chiMerge', 'bestKS', 'woeMerge', 'simple'), default 'auto'.
            copy (bool, optional)
        """
        self.spec = spec
        self.bin_info = {}
        self.missing_values = missing_values or [-990000, "__N.A__"]
        self.treat_missing = treat_missing
        self.keep_dtypes = keep_dtypes
        self.copy = copy
        self._bin_strategy = {}
        self._feat_types = {}
        self._bin_array = {}
        self._cat_others = {}
        self._split_symbol = "||"
        self._eps = np.finfo(float).eps

    def _split_dataset(self, X, y=None):
        """
        This function will split features into normal and missing values.
        The missing values include user-defined ones and NaN (which will be assign -990000) and None (which will be assign "__N.A__")
        """
        missing_idc = X.apply(lambda x: pd.isna(x) or (x in self.missing_values))
        normal_idc = X.apply(
            lambda x: (not pd.isna(x)) and (x not in self.missing_values)
        )
        X_missing = X.loc[missing_idc].copy()
        X_normal = X.loc[normal_idc].copy()
        y_missing = None
        y_normal = None
        # at least fill to -990000 for numerical features and __N.A__ for categorical features.
        if is_numeric_dtype(X.dtype):
            X_missing = X_missing.fillna(-990000)
        else:
            X_missing = X_missing.fillna("__N.A__")
        if y is not None:
            y_missing = y.loc[missing_idc].copy()
            y_normal = y.loc[normal_idc].copy()

        return X_missing, X_normal, y_missing, y_normal

    def _odds_ratio(self, df):
        """
        Use for risk rules setting
        neg_odds > pos_odds -> Feature can saperate bad and good better before current bin (a,b] inclusive -> initiate rule feature<=b
        """

        neg_odds = []
        pos_odds = []
        for idx, _ in df.iterrows():
            with np.errstate(divide="ignore", invalid="ignore"):
                bad_bef = df.iloc[: idx + 1]["bad"].sum()
                good_bef = df.iloc[: idx + 1]["good"].sum()
                odds_bef = np.float64(bad_bef) / good_bef
                bad_aft = df.iloc[idx + 1 :]["bad"].sum()
                good_aft = df.iloc[idx + 1 :]["good"].sum()
                odds_aft = np.float64(bad_aft) / good_aft
                neg_odds.append(np.float64(odds_bef) / odds_aft)
                pos_odds.append(np.float64(odds_aft) / odds_bef)

        return neg_odds, pos_odds

    def _stat_feat(self, X_normal, y_normal, X_missing, y_missing):
        original_cols = [
            "bin_total",
            "bin_rate",
            "bin_dist",
            "bad",
            "bad_rate",
            "bad_dist",
            "cum_bad_dist",
            "good",
            "good_rate",
            "good_dist",
            "cum_good_dist",
            "odds_neg",
            "odds_pos",
            "ks",
            "lift",
            "woe",
            "iv_bin",
            "iv_total",
        ]
        renamed_cols = [
            [
                "Obs",
                "Obs",
                "Obs",
                "Bad",
                "Bad",
                "Bad",
                "Bad",
                "Good",
                "Good",
                "Good",
                "Good",
                "OR",
                "OR",
                "KS",
                "Lift",
                "WOE",
                "IV",
                "IV",
            ],
            [
                "#Obs",
                "%Obs",
                "dist",
                "#Bad",
                "%Bad",
                "dist",
                "cum_dist",
                "#Good",
                "%Good",
                "dist",
                "cum_dist",
                "neg",
                "pos",
                "",
                "",
                "",
                "bin",
                "feature",
            ],
        ]
        df_y = pd.concat([y_normal, y_missing])
        total = len(df_y)
        total_bad = sum(df_y)
        total_good = total - total_bad
        df_normal = pd.DataFrame({"bins": X_normal, "label": y_normal})
        df_missing = pd.DataFrame({"bins": X_missing, "label": y_missing})
        gp_normal = (
            df_normal.groupby("bins", observed=True)
            .agg(
                bin_total=("label", "count"),
                bad=("label", lambda x: sum(x)),
                good=("label", lambda x: len(x) - sum(x)),
            )
            .sort_index()
            .reset_index()
        )
        gp_normal["cum_good"] = np.cumsum(gp_normal["good"])
        gp_normal["cum_bad"] = np.cumsum(gp_normal["bad"])
        gp_normal["bin_type"] = "bin_normal"
        gp_normal["odds_neg"], gp_normal["odds_pos"] = self._odds_ratio(gp_normal)
        gp_missing = (
            df_missing.groupby("bins", observed=True)
            .agg(
                bin_total=("label", "count"),
                bad=("label", lambda x: sum(x)),
                good=("label", lambda x: len(x) - sum(x)),
            )
            .sort_index()
            .reset_index()
        )
        gp_missing["bin_type"] = "bin_missing"
        res = pd.concat([gp_normal, gp_missing])
        res["bin_rate"] = res["bin_total"] / total
        res["bin_dist"] = np.cumsum(res["bin_rate"])
        res["bad_rate"] = res["bad"] / res["bin_total"]
        res["bad_dist"] = res["bad"] / total_bad
        res["cum_bad_dist"] = np.cumsum(res.bad_dist)
        res["good_rate"] = res["good"] / res["bin_total"]
        res["good_dist"] = res["good"] / total_good
        res["cum_good_dist"] = np.cumsum(res.good_dist)
        res["ks"] = np.abs(res["cum_bad_dist"] - res["cum_good_dist"])
        res["lift"] = res["bad_rate"] / (total_bad / total)
        # should not compute a new woe value in transform. If the value is not in training data, use the mean woe value of all bins.
        res["woe"] = (
            res["bins"]
            .map(
                lambda x: self.bin_info[X_normal.name].get(
                    x, np.mean(list(self.bin_info[X_normal.name].values()))
                )
            )
            .astype(float)
        )
        res["iv_bin"] = res["woe"] * (res["good_dist"] - res["bad_dist"])
        res["iv_total"] = res["iv_bin"].sum()
        res.index = pd.MultiIndex.from_tuples(
            [(X_normal.name, b) for b in res.bins], names=("feature_name", "bins")
        )
        res = res[original_cols]
        res.columns = renamed_cols
        return res

    def _get_woe(self, X_normal, y_normal, X_missing, y_missing):
        # need to compute distribution and woe after concating missing and normal parts.
        df_y = pd.concat([y_normal, y_missing])
        total_bad = sum(df_y)
        total_good = len(df_y) - total_bad

        feat_bin_info = {}
        feat_bin_info.update(
            pd.concat([X_normal, y_normal], axis=1)
            .rename({X_normal.name: "bin", y_normal.name: "label"}, axis=1)
            .groupby("bin", observed=True)
            .agg(
                woe=(
                    "label",
                    lambda x: np.log(
                        ((len(x) - sum(x)) / total_good + self._eps)
                        / (sum(x) / total_bad + self._eps)
                    ),
                )
            )
            .sort_values("woe", ascending=False)
            .to_dict()["woe"]
        )
        feat_bin_info.update(
            pd.concat([X_missing, y_missing], axis=1)
            .rename({X_missing.name: "bin", y_missing.name: "label"}, axis=1)
            .groupby("bin", observed=True)
            .agg(
                woe=(
                    "label",
                    lambda x: np.log(
                        ((len(x) - sum(x)) / total_good + self._eps)
                        / (sum(x) / total_bad + self._eps)
                    ),
                )
            )
            .sort_values("woe", ascending=False)
            .to_dict()["woe"]
        )

        return feat_bin_info

    def _cat_bin_mapping(self, x, bin_list, cat_others):
        if pd.isna(x) or not x:
            return "__N.A__"
        if x in cat_others:
            return "__OTHERS__"
        for val in bin_list:
            if str(x) in val.split(self._split_symbol):
                return val
        return "__N.A__"

    def _treat_missing(self):
        """
        Every bin should have at least either -990000 or __N.A__ bins for handle unexpected values and empty values.
        This is used in the situation that, there is no null value in training data or there is new value in the test/online data.
        """
        for feat, bins_woe in self.bin_info.items():
            if ("__N.A__" not in bins_woe) and (-990000 not in bins_woe):
                normal_woes = [
                    v for k, v in bins_woe.items() if k not in self.missing_values
                ]
                if not normal_woes:
                    res = np.nan
                elif self.treat_missing == "mean":
                    res = np.mean(normal_woes)
                elif self.treat_missing == "min":
                    res = np.min(normal_woes)
                elif self.treat_missing == "max":
                    res = np.max(normal_woes)
                elif self.treat_missing == "zero":
                    res = 0
                else:
                    raise ValueError(
                        f"Unknown treat_missing strategy `{self.treat_missing}`, expected one of (`mean`, `min`, `max`, `zero`)"
                    )
                at_least_bin = (
                    -990000 if self._feat_types[feat] == "numerical" else "__N.A__"
                )
                self.bin_info[feat][at_least_bin] = res

    def fit(self, X, y, **kwargs):
        """Fitting the encoder.

        Returns:
            self: Encoder
        """
        cprint("[INFO] Fitting...", "white")
        if self.copy:
            inX = X.copy(deep=True)
        else:
            inX = X
        # check whether y is binary
        if len(np.unique(y)) != 2:
            raise ValueError("Label should be binary!")
        # check whether all column binning strategy are defined
        diff_cols = set(inX.columns) - set(self.spec.keys())
        if diff_cols:
            raise KeyError(
                f"{len(diff_cols)} columns haven't decided the binning strategy yet!"
            )
        idx = 0
        for feat, bin_strategy in self.spec.items():
            idx += 1
            cprint(f"[INFO] {idx}/{len(self.spec)} Process {feat}", "white")
            target_bin_cnt = kwargs.get("target_bin_cnt", 5)
            min_bin_rate = kwargs.get("min_bin_rate", 0.05)
            monotonic_trend = kwargs.get("monotonic_trend", "auto_asc_desc")
            if feat not in X:
                raise KeyError(f"{feat} in Binning Category is not in X columns.")

            X_missing, X_normal, y_missing, y_normal = self._split_dataset(X[feat], y)
            # inX.loc[X_missing.index, feat] = X_missing

            self._feat_types[feat] = (
                "numerical" if is_numeric_dtype(X_normal.dtype) else "categorical"
            )
            if bin_strategy == "auto":
                if self._feat_types[feat] == "numerical":
                    bin_strategy = "bestKS"
                else:
                    bin_strategy = "woeMerge"
            if X_normal.empty:
                cprint(f"[WARN] {feat} has no normal values!", "yellow")
                binned_data = pd.Series([], name=feat)
                self._bin_array[feat] = []
            elif isinstance(bin_strategy, list):
                assert (
                    len(bin_strategy) > 1
                ), "Custom binning strategy should have at least two elements!"
                binned_data = pd.cut(X_normal, bin_strategy, include_lowest=True)
                self._bin_array[feat] = bin_strategy
            elif bin_strategy == "qcut":
                qcut = QCut(target_bin_cnt)
                binned_data = qcut.fit_transform(X_normal, y_normal)
                self._bin_array[feat] = qcut.bins.tolist()
            elif bin_strategy == "simple":
                simple = SimpleCut()
                binned_data = simple.fit_transform(X_normal, y_normal)
                self._bin_array[feat] = simple.bins.tolist()
            elif bin_strategy == "chiMerge":
                nu = X_normal.nunique()
                chimerge = ChiMergeCut(
                    target_bin_cnt, initial_intervals=100 if nu > 100 else False
                )
                binned_data = chimerge.fit_transform(X_normal, y_normal)
                self._bin_array[feat] = chimerge.bins.tolist()
            elif bin_strategy == "bestKS":
                best_ks = BestKSCut(target_bin_cnt, min_bin_rate)
                binned_data = best_ks.fit_transform(X_normal, y_normal)
                self._bin_array[feat] = best_ks.bins.tolist()
            elif bin_strategy == "optimal":
                optimal = OptimalCut(
                    feat,
                    self._feat_types[feat],
                    max_n_bins=target_bin_cnt,
                    min_prebin_size=min_bin_rate,
                    monotonic_trend=monotonic_trend,
                )
                binned_data = optimal.fit_transform(X_normal, y_normal)
                self._bin_array[feat] = optimal.bins
            elif bin_strategy == "woeMerge":
                woemerge = WOEMerge(target_bin_cnt, min_bin_rate)
                binned_data = woemerge.fit_transform(X_normal, y_normal)
                self._bin_array[feat] = woemerge.bins
                self._cat_others[feat] = woemerge.cat_others
            elif bin_strategy == False:
                bins = pd.unique(X_normal).tolist()
                if len(bins) > 10:
                    cprint(
                        f"[WARN] {feat} have too many categories, suggest to bin before!",
                        "yellow",
                    )
                binned_data = X_normal
                self._bin_array[feat] = bins
            else:
                raise ValueError(f"Unknown binning_method strategy `{bin_strategy}`.")

            # inX.loc[binned_data.index, feat] = binned_data

            self._bin_strategy[feat] = (
                bin_strategy
                if isinstance(bin_strategy, str) or isinstance(bin_strategy, bool)
                else "custom"
            )
            # compute and save woe
            self.bin_info[feat] = self._get_woe(
                binned_data, y_normal, X_missing, y_missing
            )

        self._treat_missing()
        return self

    def transform(self, X, y=None):
        """
        Transforming the X with fitted encoder.

        Returns:
            outX: The transformed X
        """
        cprint("[INFO] Transforming...", "white")
        if self.copy:
            outX = X.copy(deep=True)
        else:
            outX = X
        diff_cols = set(outX.columns) - set(self.bin_info.keys())
        if diff_cols:
            raise KeyError(f"{len(diff_cols)} columns haven't fitted yet!")

        original_dtypes = outX.dtypes
        idx = 0
        for feat in X.columns:
            idx += 1
            cprint(f"[INFO] {idx}/{outX.shape[1]} Process {feat}", "white")
            X_missing, X_normal, _, _ = self._split_dataset(outX[feat])
            outX.loc[X_missing.index, feat] = X_missing
            if not X_normal.empty:
                if self._feat_types[feat] == "numerical":
                    bins_series = pd.cut(
                        X_normal, self._bin_array[feat], include_lowest=True
                    )
                    binned_data = bins_series.map(self.bin_info[feat]).astype(float)
                else:
                    bin_labels = outX[feat].map(
                        lambda x: self._cat_bin_mapping(
                            x, self._bin_array[feat], self._cat_others.get(feat, [])
                        )
                    )
                    binned_data = bin_labels.map(self.bin_info[feat]).astype(
                        outX[feat].dtype
                    )
                outX.loc[X_normal.index, feat] = binned_data
                outX.loc[X_missing.index, feat] = outX.loc[
                    X_missing.index, feat
                ].replace(self.bin_info[feat])
                # outX[feat] = outX[feat].replace(self.bin_info[feat])

        if self.keep_dtypes:
            outX = outX.astype(original_dtypes)
        else:
            outX = outX.astype("float64")
        return outX

    def get_woe_df(self, X, y):
        """
        Generate WOE DataFrame for the given X and y.
        """
        cprint("[INFO] Generating WOE DataFrame...", "white")
        idx = 0
        woe_info = []
        for feat in X.columns:
            idx += 1
            cprint(f"[INFO] {idx}/{X.shape[1]} Process {feat}", "white")
            X_missing, X_normal, y_missing, y_normal = self._split_dataset(X[feat], y)
            X.loc[X_missing.index, feat] = X_missing
            if X_normal.empty:
                binned_data = pd.Series([], name=feat)
            elif self._feat_types.get(feat) == "numerical":
                binned_data = pd.cut(
                    X_normal, self._bin_array[feat], include_lowest=True
                )
            else:
                binned_data = X_normal.map(
                    lambda x: self._cat_bin_mapping(
                        x, self._bin_array[feat], self._cat_others.get(feat, [])
                    )
                )

            feat_woe_df = self._stat_feat(binned_data, y_normal, X_missing, y_missing)
            feat_woe_df["bin_strategy"] = self._bin_strategy[feat]
            feat_woe_df["feat_type"] = self._feat_types[feat]
            woe_info.append(feat_woe_df)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            woe_info = pd.concat(woe_info)
        # Trying to order the woe df with IV value
        ordered_ft = (
            woe_info["IV"]["feature"]
            .droplevel(1)
            .reset_index()
            .drop_duplicates()
            .sort_values("feature", ascending=False)["feature_name"]
            .tolist()
        )
        woe_info = woe_info.reindex(ordered_ft, level=0)
        return woe_info

    def fit_transform(self, X, y=None, **kwargs):
        return self.fit(X, y, **kwargs).transform(X, y)
