#!/usr/bin/env python
# Version: 0.3.0
# Created: 2024-04-07
# Last Modified: 2025-08-19
# Author: ["Hanyuan Zhang"]

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from termcolor import cprint

from .binner import BestKSCut, ChiMergeCut, OptimalCut, QCut, SimpleCut, WOEMerge


class Encoder(BaseEstimator, TransformerMixin):
    """
    Weight of Evidence (WOE) encoder for feature transformation and binning.

    This encoder transforms features using Weight of Evidence encoding, which is
    commonly used in credit scoring and risk modeling. It supports multiple
    binning strategies for both numerical and categorical features.

    Attributes:
        spec: Dictionary mapping feature names to binning strategies
        bin_info: Dictionary containing WOE mappings for each feature
        woe_df: DataFrame containing detailed WOE analysis (generated when y is provided to transform)
        missing_values: List of values to treat as missing
        treat_missing: Strategy for handling missing values ('mean', 'min', 'max', 'zero')
        keep_dtypes: Whether to preserve original data types after transformation
        copy: Whether to make a copy of input data

    Examples:
        >>> from optimus.encoder import Encoder
        >>> import pandas as pd
        >>>
        >>> # Sample data
        >>> X = pd.DataFrame({
        ...     'age': [25, 35, 45, 55, 65],
        ...     'income': [30000, 50000, 70000, 90000, 110000],
        ...     'education': ['high_school', 'bachelor', 'master', 'phd', 'bachelor']
        ... })
        >>> y = pd.Series([0, 0, 1, 1, 1])
        >>>
        >>> # Define binning strategies
        >>> spec = {
        ...     'age': 'bestKS',           # Best KS binning for numerical
        ...     'income': 'chiMerge',      # Chi-merge binning for numerical
        ...     'education': 'woeMerge'    # WOE merge for categorical
        ... }
        >>>
        >>> # Fit and transform
        >>> encoder = Encoder(spec=spec)
        >>> encoder.fit(X, y)
        >>> X_woe = encoder.transform(X, y)  # y provided to generate woe_df
        >>>
        >>> # Access WOE analysis
        >>> print(encoder.woe_df)
    """

    def __init__(
        self,
        spec: Dict[str, Union[str, List[float]]],
        missing_values: Optional[List[Union[int, str]]] = None,
        treat_missing: str = "mean",
        keep_dtypes: bool = False,
        copy: bool = True,
    ) -> None:
        """
        Initialize the WOE Encoder.

        Args:
            spec: Dictionary mapping feature names to binning strategies.
                Strategies can be:
                - 'auto': Automatically choose best strategy
                - 'qcut': Equal frequency binning
                - 'chiMerge': Chi-square based binning
                - 'bestKS': KS statistic based binning
                - 'woeMerge': WOE based categorical merging
                - 'optimal': Optimal binning using OptimalBinning
                - 'simple': Simple binning
                - List[float]: Custom bin edges
                - False: No binning (use original categories)
            missing_values: List of values to treat as missing (default: [-990000, "__N.A__"])
            treat_missing: Strategy for missing value treatment:
                - 'mean': Use mean of normal WOE values
                - 'min': Use minimum WOE value
                - 'max': Use maximum WOE value
                - 'zero': Use 0
            keep_dtypes: Whether to preserve original data types after transformation
            copy: Whether to make a copy of input data during processing

        Raises:
            ValueError: If invalid treat_missing strategy is provided
        """
        self.spec = spec
        self.bin_info: Dict[str, Dict] = {}
        self.woe_df: Optional[pd.DataFrame] = None
        self.missing_values = missing_values or [-990000, "__N.A__"]
        self.treat_missing = treat_missing
        self.keep_dtypes = keep_dtypes
        self.copy = copy
        self._bin_strategy: Dict[str, str] = {}
        self._feat_types: Dict[str, str] = {}
        self._bin_array: Dict[str, List] = {}
        self._cat_others: Dict[str, List] = {}
        self._split_symbol = "||"
        self._eps = np.finfo(float).eps

    def _split_dataset(
        self, X: pd.Series, y: Optional[pd.Series] = None
    ) -> Tuple[pd.Series, pd.Series, Optional[pd.Series], Optional[pd.Series]]:
        """
        Split feature data into normal and missing value subsets.

        This method identifies missing values based on NaN, None, and user-defined
        missing values, then splits both feature and target data accordingly.

        Args:
            X: Feature series to split
            y: Optional target series to split correspondingly

        Returns:
            Tuple containing:
            - X_missing: Feature values identified as missing
            - X_normal: Feature values not identified as missing
            - y_missing: Target values corresponding to missing features (if y provided)
            - y_normal: Target values corresponding to normal features (if y provided)

        Examples:
            >>> encoder = Encoder(spec={})
            >>> X = pd.Series([1, 2, np.nan, 4, -990000])
            >>> y = pd.Series([0, 1, 0, 1, 0])
            >>> X_miss, X_norm, y_miss, y_norm = encoder._split_dataset(X, y)
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
        res["ks"] = res.apply(
            lambda row: (
                np.abs(row["cum_bad_dist"] - row["cum_good_dist"])
                if row["bin_type"] == "bin_normal"
                else np.nan
            ),
            axis=1,
        )
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
        # Direct match for values in bin_list (for bin_strategy=False)
        if x in bin_list:
            return x
        # Split-based match for merged categories (for bin_strategy='woeMerge')
        for val in bin_list:
            if str(x) in str(val).split(self._split_symbol):
                return val
        return "__N.A__"

    def _treat_missing(self):
        """
        Every bin should have at least either -990000 or __N.A__ bins for handle unexpected values and empty values.
        This is used in the situation that, there is no null value in training data or there is new value in the test/online data.
        Also ensures all values in missing_values list are present in bin_info with appropriate WOE values.
        """
        for feat, bins_woe in self.bin_info.items():
            # Calculate default missing WOE value based on normal values
            normal_woes = [
                v for k, v in bins_woe.items() if k not in self.missing_values
            ]
            if not normal_woes:
                default_woe = np.nan
            elif self.treat_missing == "mean":
                default_woe = np.mean(normal_woes)
            elif self.treat_missing == "min":
                default_woe = np.min(normal_woes)
            elif self.treat_missing == "max":
                default_woe = np.max(normal_woes)
            elif self.treat_missing == "zero":
                default_woe = 0
            else:
                raise ValueError(
                    f"Unknown treat_missing strategy `{self.treat_missing}`, expected one of (`mean`, `min`, `max`, `zero`)"
                )

            # Ensure standard missing bin exists
            if ("__N.A__" not in bins_woe) and (-990000 not in bins_woe):
                at_least_bin = (
                    -990000 if self._feat_types[feat] == "numerical" else "__N.A__"
                )
                self.bin_info[feat][at_least_bin] = default_woe

            # Ensure all user-defined missing values are in bin_info
            for missing_val in self.missing_values:
                if missing_val not in bins_woe:
                    self.bin_info[feat][missing_val] = default_woe

    def fit(
        self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray], **kwargs: Any
    ) -> "Encoder":
        """
        Fit the WOE encoder to the training data.

        This method learns the WOE mappings for each feature based on the target variable.
        It applies the specified binning strategy for each feature and calculates
        corresponding WOE values.

        Args:
            X: Training feature data
            y: Target binary labels (must be binary: 0 and 1)
            **kwargs: Additional keyword arguments:
                target_bin_cnt (int): Target number of bins (default: 5)
                min_bin_rate (float): Minimum bin size ratio (default: 0.05)
                monotonic_trend (str): Monotonic trend for optimal binning (default: 'auto_asc_desc')

        Returns:
            self: Fitted encoder instance

        Raises:
            ValueError: If target variable is not binary
            KeyError: If features in spec are not found in X

        Examples:
            >>> encoder = Encoder(spec={'age': 'bestKS', 'income': 'chiMerge'})
            >>> encoder.fit(X_train, y_train)
            >>> print(f"Fitted {len(encoder.bin_info)} features")
        """
        cprint("[INFO] Fitting...", "white")
        if self.copy:
            inX = X.copy(deep=True)
        else:
            inX = X
        # check whether y is binary
        if len(np.unique(y)) != 2:
            raise ValueError("Label should be binary!")
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
                if self._feat_types[feat] == "numerical":
                    X_missing = X_missing.astype(str)
                    X_normal = X_normal.astype(str)
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

    def transform(
        self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> pd.DataFrame:
        """
        Transform features using fitted WOE encodings.

        This method applies the learned WOE mappings to transform the input features.
        If target variable y is provided, it also generates detailed WOE analysis
        and stores it in self.woe_df.

        Args:
            X: Input feature data to transform
            y: Optional target variable. If provided:
                - WOE DataFrame will be generated and stored in self.woe_df
                - Enables access to detailed WOE statistics and analysis

        Returns:
            pd.DataFrame: Transformed features with WOE values

        Raises:
            KeyError: If features haven't been fitted yet

        Examples:
            >>> # Transform without WOE analysis
            >>> X_woe = encoder.transform(X_test)
            >>>
            >>> # Transform with WOE analysis generation
            >>> X_woe = encoder.transform(X_test, y_test)
            >>> woe_analysis = encoder.woe_df  # Access detailed analysis
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
        woe_info = [] if y is not None else None

        for feat in X.columns:
            idx += 1
            cprint(f"[INFO] {idx}/{outX.shape[1]} Process {feat}", "white")

            X_missing, X_normal, y_missing, y_normal = self._split_dataset(
                outX[feat], y
            )
            if (
                self._bin_strategy[feat] == False
                and self._feat_types[feat] == "numerical"
            ):
                X_missing = X_missing.astype(str)
                X_normal = X_normal.astype(str)
            outX.loc[X_missing.index, feat] = X_missing
            bins_categrories = pd.Series([], name=feat)

            if not X_normal.empty:
                if (
                    self._bin_strategy[feat] == False
                    or self._feat_types[feat] == "categorical"
                ):
                    bins_categrories = X_normal.map(
                        lambda x: self._cat_bin_mapping(
                            x, self._bin_array[feat], self._cat_others.get(feat, [])
                        )
                    )
                    binned_data = bins_categrories.map(self.bin_info[feat]).astype(
                        float
                    )
                elif self._feat_types[feat] == "numerical":
                    bins_categrories = pd.cut(
                        X_normal, self._bin_array[feat], include_lowest=True
                    )
                    binned_data = bins_categrories.map(self.bin_info[feat]).astype(
                        float
                    )
                outX.loc[X_normal.index, feat] = binned_data
            outX.loc[X_missing.index, feat] = outX.loc[X_missing.index, feat].replace(
                self.bin_info[feat]
            )

            if y is not None:
                feat_woe_df = self._stat_feat(
                    bins_categrories, y_normal, X_missing, y_missing
                )
                feat_woe_df["bin_strategy"] = self._bin_strategy[feat]
                feat_woe_df["feat_type"] = self._feat_types[feat]
                woe_info.append(feat_woe_df)

        if y is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                self.woe_df = pd.concat(woe_info)
            # Order the woe df with IV value
            ordered_ft = (
                self.woe_df["IV"]["feature"]
                .droplevel(1)
                .reset_index()
                .drop_duplicates()
                .sort_values("feature", ascending=False)["feature_name"]
                .tolist()
            )
            self.woe_df = self.woe_df.reindex(ordered_ft, level=0)

        if self.keep_dtypes:
            outX = outX.astype(original_dtypes)
        else:
            outX = outX.astype("float64")
        return outX

    def fit_transform(self, X, y=None, **kwargs):
        return self.fit(X, y, **kwargs).transform(X)
