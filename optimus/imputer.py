#!/usr/bin/env python
# Version: 0.4.1
# Created: 2026-01-14
# Last Modified: 2026-01-14
# Author: ["Hanyuan Zhang"]

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from termcolor import cprint


class Imputer(BaseEstimator, TransformerMixin):
    """
    Missing value imputer with feature-level strategy control.

    This imputer fills missing values based on statistical measures computed from
    non-missing values. It supports multiple strategies for both numerical and
    categorical features, with feature-level granular control.

    Attributes:
        impute_strategy: Dictionary mapping feature names to imputation strategies
        missing_values: List of values to treat as missing
        fill_values_: Dictionary storing computed fill values for each feature

    Examples:
        >>> from optimus.imputer import Imputer
        >>> import pandas as pd
        >>>
        >>> # Sample data
        >>> X = pd.DataFrame({
        ...     'age': [25, np.nan, 45, -999999, 65],
        ...     'income': [30000, 50000, np.nan, 90000, 110000],
        ...     'education': ['high_school', 'bachelor', '__N.A__', 'phd', 'bachelor']
        ... })
        >>>
        >>> # Define imputation strategies
        >>> impute_strategy = {
        ...     'age': 'mean',
        ...     'income': 'median',
        ...     'education': 'mode'
        ... }
        >>>
        >>> # Fit and transform
        >>> imputer = Imputer(
        ...     impute_strategy=impute_strategy,
        ...     missing_values=[-999999, '__N.A__']
        ... )
        >>> imputer.fit(X)
        >>> X_imputed = imputer.transform(X)
    """

    def __init__(
        self,
        impute_strategy: Optional[Dict[str, str]] = None,
        missing_values: Optional[List[Union[int, str]]] = None,
        copy: bool = True,
    ) -> None:
        """
        Initialize the Imputer.

        Args:
            impute_strategy: Dictionary mapping feature names to strategies.
                Strategies can be:
                - 'mean': Fill with mean of non-missing values (numerical only)
                - 'median': Fill with median of non-missing values (numerical only)
                - 'min': Fill with minimum of non-missing values (numerical only)
                - 'max': Fill with maximum of non-missing values (numerical only)
                - 'mode': Fill with most frequent value (numerical and categorical)
                - 'separate': Do not impute, keep missing values as-is
                If None or empty, no imputation will be performed.
            missing_values: List of values to treat as missing (default: [-990000, "__N.A__"])
            copy: Whether to make a copy of input data during processing

        Raises:
            ValueError: If invalid imputation strategy is provided
        """
        self.impute_strategy = impute_strategy or {}
        self.missing_values = missing_values or [-990000, "__N.A__"]
        self.copy = copy
        self.fill_values_: Dict[str, Union[float, str, int]] = {}

    def _identify_missing(self, X: pd.Series) -> pd.Series:
        """
        Identify missing values in a feature series.

        Args:
            X: Feature series to check

        Returns:
            Boolean series indicating missing positions
        """
        return X.apply(lambda x: pd.isna(x) or (x in self.missing_values))

    def _compute_fill_value(
        self, X: pd.Series, strategy: str
    ) -> Union[float, str, int]:
        """
        Compute the fill value for a feature based on strategy.

        Args:
            X: Non-missing feature values
            strategy: Imputation strategy

        Returns:
            Computed fill value

        Raises:
            ValueError: If strategy is not applicable to feature type
        """
        valid_strategies = ("mean", "median", "min", "max", "mode", "separate")
        if strategy not in valid_strategies:
            raise ValueError(
                f"Unknown strategy `{strategy}`, expected one of {valid_strategies}"
            )

        if strategy == "separate":
            return None

        is_numerical = is_numeric_dtype(X.dtype)

        # Numerical strategies
        if strategy == "mean":
            if not is_numerical:
                raise ValueError(
                    f"Strategy 'mean' is only applicable to numerical features"
                )
            return X.mean()
        elif strategy == "median":
            if not is_numerical:
                raise ValueError(
                    f"Strategy 'median' is only applicable to numerical features"
                )
            return X.median()
        elif strategy == "min":
            if not is_numerical:
                raise ValueError(
                    f"Strategy 'min' is only applicable to numerical features"
                )
            return X.min()
        elif strategy == "max":
            if not is_numerical:
                raise ValueError(
                    f"Strategy 'max' is only applicable to numerical features"
                )
            return X.max()
        elif strategy == "mode":
            # Mode works for both numerical and categorical
            mode_values = X.mode()
            if len(mode_values) == 0:
                if len(X) > 0:
                    cprint(
                        f"[WARN] No mode found, using first value as fallback",
                        "yellow",
                    )
                    return X.iloc[0]
                else:
                    raise ValueError(
                        "Cannot compute mode for empty feature after removing missing values. "
                        "All values are missing for this feature."
                    )
            return mode_values[0]

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "Imputer":
        """
        Fit the imputer by computing fill values for each feature.

        This method calculates the statistical measures (mean, median, mode, etc.)
        from non-missing values for each feature based on its imputation strategy.

        Args:
            X: Training feature data
            y: Optional target variable (not used, kept for sklearn compatibility)

        Returns:
            self: Fitted imputer instance

        Raises:
            KeyError: If features in impute_strategy are not found in X
            ValueError: If invalid strategy is specified

        Examples:
            >>> imputer = Imputer(impute_strategy={'age': 'mean', 'income': 'median'})
            >>> imputer.fit(X_train)
            >>> print(imputer.fill_values_)
        """
        if not self.impute_strategy:
            cprint(
                "[INFO] No impute_strategy specified, Imputer will pass through data",
                "white",
            )
            return self

        cprint("[INFO] Fitting Imputer...", "white")

        # Check if all features in strategy exist in X
        missing_features = set(self.impute_strategy.keys()) - set(X.columns)
        if missing_features:
            raise KeyError(
                f"Features in impute_strategy not found in X: {missing_features}"
            )

        for feat, strategy in self.impute_strategy.items():
            if feat not in X.columns:
                continue

            cprint(
                f"[INFO] Computing fill value for {feat} with strategy '{strategy}'",
                "white",
            )

            # Split into missing and non-missing
            missing_mask = self._identify_missing(X[feat])
            X_normal = X[feat][~missing_mask]

            if X_normal.empty:
                cprint(
                    f"[WARN] Feature {feat} has no non-missing values, cannot compute fill value",
                    "yellow",
                )
                self.fill_values_[feat] = None
                continue

            # Compute fill value based on strategy
            try:
                fill_value = self._compute_fill_value(X_normal, strategy)
                self.fill_values_[feat] = fill_value
                if fill_value is not None:
                    cprint(
                        f"[INFO] {feat}: fill_value = {fill_value} (strategy: {strategy})",
                        "white",
                    )
            except ValueError as e:
                cprint(f"[ERROR] {feat}: {str(e)}", "red")
                raise

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features by filling missing values with computed fill values.

        This method applies the learned imputation strategy to replace missing
        values in the input data.

        Args:
            X: Input feature data to transform

        Returns:
            pd.DataFrame: Transformed features with imputed values

        Raises:
            KeyError: If features haven't been fitted yet

        Examples:
            >>> imputer.fit(X_train)
            >>> X_train_imputed = imputer.transform(X_train)
            >>> X_test_imputed = imputer.transform(X_test)
        """
        if not self.impute_strategy:
            return X

        cprint("[INFO] Transforming with Imputer...", "white")

        if self.copy:
            outX = X.copy(deep=True)
        else:
            outX = X

        for feat, strategy in self.impute_strategy.items():
            if feat not in outX.columns:
                cprint(
                    f"[WARN] Feature {feat} not found in data, skipping",
                    "yellow",
                )
                continue

            if strategy == "separate":
                cprint(
                    f"[INFO] {feat}: strategy='separate', keeping original values",
                    "white",
                )
                continue

            fill_value = self.fill_values_.get(feat)
            if fill_value is None:
                cprint(
                    f"[WARN] No fill value for {feat}, skipping imputation",
                    "yellow",
                )
                continue

            # Identify missing positions
            missing_mask = self._identify_missing(outX[feat])
            n_missing = missing_mask.sum()

            if n_missing > 0:
                # Fill missing values
                outX.loc[missing_mask, feat] = fill_value
                cprint(
                    f"[INFO] {feat}: filled {n_missing} missing values with {fill_value}",
                    "white",
                )

        return outX

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Fit the imputer and transform the data in one step.

        Args:
            X: Feature data
            y: Optional target variable

        Returns:
            pd.DataFrame: Transformed data with imputed values
        """
        return self.fit(X, y).transform(X)
