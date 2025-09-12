#!/usr/bin/env python
# Version: 0.3.0
# Created: 2024-04-07
# Author: ["Hanyuan Zhang"]

import warnings
from typing import Any, Dict, List, Optional, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .metrics import Metrics

warnings.filterwarnings("ignore")


class Filter(TransformerMixin):
    """
    A basic filter transformer for excluding specified columns from feature selection.

    This class allows you to specify certain columns that should be preserved
    during the transformation process but not included in feature selection.

    Attributes:
        extra_cols_: List of column names to exclude from feature selection
        df_extra: DataFrame containing the excluded columns
        feature_names_: List of feature names after filtering

    Examples:
        >>> filter_trans = Filter(extra_cols=['id', 'timestamp'])
        >>> filter_trans.fit(X)
        >>> X_filtered = filter_trans.transform(X)
    """

    def __init__(self, extra_cols: Optional[List[str]] = None) -> None:
        """
        Initialize the Filter transformer.

        Args:
            extra_cols: List of column names to exclude from feature selection.
                These columns will be preserved but not used in downstream processing.
        """
        self.extra_cols_ = extra_cols or []
        self.df_extra: Optional[pd.DataFrame] = None
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "Filter":
        """
        Fit the filter to identify feature columns.

        Args:
            X: Input feature matrix
            y: Target variable (not used, for sklearn compatibility)

        Returns:
            self: Fitted transformer

        Raises:
            AssertionError: If extra_cols contains columns not in X
        """
        assert (
            n := set(self.extra_cols_) - set(X.columns.tolist())
        ) == set(), f"Extra columns {n} not in X"
        self.feature_names_ = [
            c for c in X.columns.tolist() if c not in self.extra_cols_
        ]
        self.df_extra = X[self.extra_cols_]
        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Transform by selecting only the feature columns.

        Args:
            X: Input feature matrix
            y: Target variable (not used, for sklearn compatibility)

        Returns:
            pd.DataFrame: Filtered feature matrix
        """
        return X[self.feature_names_]


class CorrSelector(TransformerMixin):
    """
    Feature selector based on correlation analysis.

    This selector removes highly correlated features using various strategies
    to reduce multicollinearity while preserving predictive power.

    Attributes:
        detail: Dictionary containing correlation analysis details
        selected_features: List of selected feature names
        removed_features: List of removed feature names
        corr_threshold: Correlation threshold for feature removal
        user_feature_list: Optional list of specific features to consider
        method: Method for handling correlated features

    Examples:
        >>> selector = CorrSelector(corr_threshold=0.95, method='iv_descending')
        >>> selector.fit(X, y)
        >>> X_selected = selector.transform(X)
        >>> print(f"Removed {len(selector.removed_features)} correlated features")
    """

    def __init__(
        self,
        corr_threshold: float = 0.95,
        user_feature_list: Optional[List[str]] = None,
        method: str = "iv_descending",
    ) -> None:
        """
        Initialize the correlation selector.

        Args:
            corr_threshold: Correlation threshold above which features are considered
                highly correlated (default: 0.95)
            user_feature_list: Optional list of specific features to consider.
                If None, all features are considered.
            method: Method for selecting which feature to keep from correlated pairs:
                - 'iv_descending': Keep feature with higher Information Value
                - 'random': Random selection

        Examples:
            >>> # Remove features with correlation > 0.9, prioritize by IV
            >>> selector = CorrSelector(corr_threshold=0.9, method='iv_descending')

            >>> # Only consider specific features
            >>> selector = CorrSelector(
            ...     corr_threshold=0.95,
            ...     user_feature_list=['age', 'income', 'score'],
            ...     method='iv_descending'
            ... )
        """
        self.detail: Dict[str, Any] = {}
        self.selected_features: List[str] = []
        self.removed_features: List[str] = []
        self.corr_threshold = corr_threshold
        self.user_feature_list = user_feature_list
        self.method = method

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> "CorrSelector":
        """
        Fit the correlation selector to identify correlated features.

        This method analyzes correlation between features and selects which
        features to keep based on the specified method and threshold.

        Args:
            X: Input feature matrix
            y: Target variable for calculating Information Value

        Returns:
            self: Fitted selector

        Examples:
            >>> selector = CorrSelector(corr_threshold=0.95)
            >>> selector.fit(X_train, y_train)
            >>> print(f"Selected {len(selector.selected_features)} features")
        """
        print("[INFO]: Processing Correlation Selector...")
        feature_list = sorted(self.user_feature_list or X.columns.tolist())
        lst_iv = [Metrics.get_iv(y, X[c]) for c in feature_list]
        df_iv = pd.DataFrame({"var": feature_list, "iv": lst_iv})

        if self.method == "iv_ascending":
            df_iv = df_iv.sort_values(by="iv", ascending=True).set_index("var")
            feature_list = df_iv.index.tolist()
            df_corr = X[feature_list].corr()
            df_corr = abs(
                df_corr
                - pd.DataFrame(
                    np.identity(len(df_corr)),
                    index=df_corr.index,
                    columns=df_corr.columns,
                )
            )  # remove self
            df_res = pd.concat([df_iv, df_corr], axis=1)
            self.detail["before"] = df_res

            for var in feature_list:
                if df_res[var].max() >= self.corr_threshold:
                    df_res = df_res.drop(var, axis=0)
                    df_res = df_res.drop(var, axis=1)
                else:
                    continue
            self.detail["after"] = df_res

        elif self.method == "iv_descending":
            df_iv = df_iv.sort_values(by="iv", ascending=False).set_index("var")
            feature_list = df_iv.index.tolist()
            df_corr = X[feature_list].corr()
            df_corr = abs(
                df_corr
                - pd.DataFrame(
                    np.identity(len(df_corr)),
                    index=df_corr.index,
                    columns=df_corr.columns,
                )
            )
            df_res = pd.concat([df_iv, df_corr], axis=1)
            self.detail["before"] = df_res

            for var in feature_list:
                if var not in df_res.index:
                    continue
                else:
                    lst_remove = df_res[
                        df_res[var] >= self.corr_threshold
                    ].index.tolist()
                    df_res = df_res.drop(lst_remove, axis=0)
                    df_res = df_res.drop(lst_remove, axis=1)
            self.detail["after"] = df_res

        else:
            raise Exception(
                f"Unknown method {self.method}, please specify iv_descending or iv_ascending"
            )

        self.selected_features = df_res.index.tolist()
        self.removed_features = sorted(set(feature_list) - set(self.selected_features))
        self.summary()
        return self

    def transform(self, X, y=None):
        return X[self.selected_features]

    def summary(self):
        print(
            f"\nRemoved {len(self.removed_features)} features, {len(self.selected_features)} remaining.\n"
        )
        print(
            "\n==============================================================================="
        )


class GINISelector(TransformerMixin):
    def __init__(self, user_feature_list=None):
        self.detail = None
        self.selected_features = list()
        self.removed_features = list()
        self.user_feature_list = user_feature_list

    def fit(self, X, y, refX=None, refy=None):
        print("[INFO]: Processing GINI Selector...")
        if refX is None or refy is None:
            print(
                "[WARNING]: need to provide train and test data to do GINI based feature selection"
            )
            refX = X.copy()
            refy = y.copy()
        feature_list = sorted(self.user_feature_list or X.columns.tolist())
        # compute gini on train data
        lst_gini_train = [Metrics.get_gini(y, X[c]) for c in feature_list]

        # compute gini on test data
        lst_gini_test = [Metrics.get_gini(refy, refX[c]) for c in feature_list]

        # select feature with same gini sign in train and test data
        df_res = pd.DataFrame(
            {
                "var": feature_list,
                "gini_train": lst_gini_train,
                "gini_test": lst_gini_test,
            }
        )
        df_res.loc[:, "selected"] = df_res["gini_train"] * df_res["gini_test"] > 0

        self.detail = df_res
        self.selected_features = df_res.loc[
            (df_res["selected"] == True), "var"
        ].tolist()
        self.removed_features = df_res.loc[
            (df_res["selected"] == False), "var"
        ].tolist()
        self.summary()
        return self

    def transform(self, X, y=None):
        return X[self.selected_features]

    def summary(self):
        print(
            f"\nRemoved {len(self.removed_features)} features, {len(self.selected_features)} remaining.\n"
        )
        print(
            "\n==============================================================================="
        )


class PSISelector(TransformerMixin):
    def __init__(self, psi_threshold=0.1, user_feature_list=None):
        self.detail = None
        self.selected_features = list()
        self.removed_features = list()
        self.psi_threshold = psi_threshold
        self.user_feature_list = user_feature_list

    def fit(self, X, y, refX=None, refy=None):
        print("[INFO]: Processing PSI Selector...")
        feature_list = sorted(self.user_feature_list or X.columns.tolist())
        if refX is None:
            print(
                "[WARNING]: need to provide train and test data to do PSI based feature selection"
            )
            refX = X
        # compute PSI
        psi = []
        for col in refX[feature_list]:
            psi.append(Metrics.get_psi(X[col], refX[col]))

        sr_psi = pd.Series(psi, index=refX[feature_list].columns)
        sr_psi.name = "psi"
        sr_psi.index.name = "var"
        df_psi = sr_psi.reset_index()

        # select feature with PSI < psi_threshold
        df_psi.loc[:, "selected"] = df_psi["psi"] < self.psi_threshold

        self.detail = df_psi
        self.selected_features = df_psi.loc[
            (df_psi["selected"] == True), "var"
        ].tolist()
        self.removed_features = df_psi.loc[
            (df_psi["selected"] == False), "var"
        ].tolist()
        self.summary()
        return self

    def transform(self, X, y=None):
        return X[self.selected_features]

    def summary(self):
        print(
            f"\nRemoved {len(self.removed_features)} features, {len(self.selected_features)} remaining.\n"
        )
        print(
            "\n==============================================================================="
        )


class IVSelector(TransformerMixin):
    def __init__(self, iv_threshold=0.02, user_feature_list=None):
        self.detail = None
        self.selected_features = list()
        self.removed_features = list()
        self.iv_threshold = iv_threshold
        self.user_feature_list = user_feature_list

    def fit(self, X, y, sample_idx=None):
        print("[INFO]: Processing IV Selector...")
        feature_list = sorted(self.user_feature_list or X.columns.tolist())

        lst_iv = [Metrics.get_iv(y, X[c]) for c in feature_list]
        df_iv = pd.DataFrame({"var": feature_list, "iv": lst_iv})

        # select feature with IV >= iv_threshold
        df_iv.loc[:, "selected"] = df_iv["iv"] >= self.iv_threshold

        self.detail = df_iv
        self.selected_features = df_iv.loc[(df_iv["selected"] == True), "var"].tolist()
        self.removed_features = df_iv.loc[(df_iv["selected"] == False), "var"].tolist()
        self.summary()
        return self

    def transform(self, X, y=None):
        return X[self.selected_features]

    def summary(self):
        print(
            f"\nRemoved {len(self.removed_features)} features, {len(self.selected_features)} remaining.\n"
        )
        print(
            "\n==============================================================================="
        )


class VIFSelector(TransformerMixin):
    def __init__(self, vif_threshold=10, user_feature_list=None):
        self.detail = None
        self.selected_features = list()
        self.removed_features = list()
        self.vif_threshold = vif_threshold
        self.user_feature_list = user_feature_list

    def fit(self, X, y, sample_idx=None):
        print("[INFO]: Processing VIF Selector...")
        feature_list = sorted(self.user_feature_list or X.columns.tolist())

        # compute VIF
        X_subset = X[feature_list]
        vif = [
            variance_inflation_factor(X_subset.values, i)
            for i in range(X_subset.shape[1])
        ]

        # select features with VIF < vif_threshold
        df_res = pd.DataFrame({"var": feature_list, "vif": [round(v, 3) for v in vif]})
        df_res.loc[:, "selected"] = df_res["vif"] < self.vif_threshold

        self.detail = df_res
        self.selected_features = df_res.loc[
            (df_res["selected"] == True), "var"
        ].tolist()
        self.removed_features = df_res.loc[
            (df_res["selected"] == False), "var"
        ].tolist()
        self.summary()
        return self

    def transform(self, X, y=None):
        return X[self.selected_features]

    def summary(self):
        print(
            f"\nRemoved {len(self.removed_features)} features, {len(self.selected_features)} remaining.\n"
        )
        print(
            "\n==============================================================================="
        )


class BoostingTreeSelector(TransformerMixin):
    def __init__(self, select_frac=0.9, user_feature_list=None):
        self.detail = None
        self.selected_features = list()
        self.removed_features = list()
        self.select_frac = select_frac
        self.user_feature_list = user_feature_list

    def fit(self, X, y, refX=None, refy=None):
        print("[INFO]: Processing Boosting Tree Selector...")
        feature_list = sorted(self.user_feature_list or X.columns.tolist())
        if refX is None or refy is None:
            refX = X.copy()
            refy = y.copy()

        model = lgb.LGBMClassifier(
            boosting_type="gbdt",
            num_leaves=31,
            max_depth=5,
            learning_rate=0.02,
            n_estimators=200,
            min_split_gain=0,
            min_child_weight=1e-3,
            min_child_samples=20,
            subsample=1.0,
            subsample_freq=0,
            colsample_bytree=1.0,
            reg_alpha=0.0,
            reg_lambda=0.0,
            random_state=0,
            importance_type="split",
            verbosity=-1,
        )

        model.fit(
            X,
            y,
            eval_metric="auc",
            eval_set=[(X, y), (refX[feature_list], refy)],
            callbacks=[lgb.early_stopping(stopping_rounds=10)],
        )

        df_res = (
            pd.DataFrame(
                {
                    "feature": feature_list,
                    "feature_importance": model.feature_importances_,
                }
            )
            .sort_values(by="feature_importance", ascending=False)
            .reset_index(drop=True)
        )

        select_num = int(
            self.select_frac * len(df_res[df_res["feature_importance"] != 0])
        )
        df_res["selected"] = False
        df_res.loc[0:select_num, "selected"] = True

        self.detail = df_res
        self.selected_features = df_res.loc[
            (df_res["selected"] == True), "feature"
        ].tolist()
        self.removed_features = df_res.loc[
            (df_res["selected"] == False), "feature"
        ].tolist()
        self.summary()
        return self

    def transform(self, X, y=None):
        return X[self.selected_features]

    def summary(self):
        print(
            f"\nRemoved {len(self.removed_features)} features, {len(self.selected_features)} remaining.\n"
        )
        print(
            "\n==============================================================================="
        )


class ManualSelector(TransformerMixin):
    def __init__(self, drop_features=None):
        self.detail = None
        self.selected_features = list()
        self.removed_features = list()
        self.drop_features = drop_features

    def fit(self, X, y):
        print("[INFO]: Processing Manual Selector...")
        feature_list = sorted(X.columns.tolist())
        self.selected_features = feature_list
        if self.drop_features is not None:
            self.selected_features = sorted(set(feature_list) - set(self.drop_features))
            self.removed_features = sorted(set(feature_list) & set(self.drop_features))
        df_res = pd.DataFrame(
            {
                "feature": feature_list,
                "selected": [
                    True if c in self.selected_features else False for c in feature_list
                ],
            }
        )
        self.detail = df_res
        self.summary()
        return self

    def transform(self, X, y=None):
        return X[self.selected_features]

    def summary(self):
        print(
            f"\nRemoved {len(self.removed_features)} features, {len(self.selected_features)} remaining.\n"
        )
        print(
            "\n==============================================================================="
        )
