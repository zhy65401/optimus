#!/usr/bin/env python
# Version: 0.3.0
# Created: 2024-04-07
# Author: ["Hanyuan Zhang"]

from typing import Any, Union

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


class Metrics:
    """
    A collection of statistical metrics for model evaluation and feature analysis.

    This class provides various metrics commonly used in risk modeling and credit scoring,
    including AUC, KS, Gini, IV (Information Value), and PSI (Population Stability Index).
    """

    @classmethod
    def get_auc(
        cls,
        ytrue: Union[pd.Series, np.ndarray],
        yprob: Union[pd.Series, np.ndarray],
        **kwargs: Any
    ) -> float:
        """
        Calculate Area Under the ROC Curve (AUC).

        Args:
            ytrue: True binary labels (0 or 1)
            yprob: Predicted probabilities for the positive class
            **kwargs: Additional keyword arguments
                symmetry (bool): If True and AUC < 0.5, return 1 - AUC

        Returns:
            float: AUC score between 0 and 1

        Examples:
            >>> metrics = Metrics()
            >>> y_true = [0, 0, 1, 1]
            >>> y_prob = [0.1, 0.4, 0.35, 0.8]
            >>> auc = metrics.get_auc(y_true, y_prob)
            >>> print(f"AUC: {auc:.3f}")
        """
        auc = roc_auc_score(ytrue, yprob)

        if kwargs.get("symmetry", False) is True:
            if auc < 0.5:
                auc = 1 - auc
        return auc

    @classmethod
    def get_ks(
        cls, ytrue: Union[pd.Series, np.ndarray], yprob: Union[pd.Series, np.ndarray]
    ) -> float:
        """
        Calculate Kolmogorov-Smirnov (KS) statistic.

        The KS statistic measures the maximum separation between the
        cumulative distribution functions of positive and negative classes.

        Args:
            ytrue: True binary labels (0 or 1)
            yprob: Predicted probabilities for the positive class

        Returns:
            float: KS statistic between 0 and 1, higher values indicate better separation

        Examples:
            >>> metrics = Metrics()
            >>> y_true = [0, 0, 1, 1]
            >>> y_prob = [0.1, 0.4, 0.35, 0.8]
            >>> ks = metrics.get_ks(y_true, y_prob)
            >>> print(f"KS: {ks:.3f}")
        """
        fpr, tpr, thr = roc_curve(ytrue, yprob)
        ks = max(abs(tpr - fpr))
        return ks

    @classmethod
    def get_gini(
        cls,
        ytrue: Union[pd.Series, np.ndarray],
        yprob: Union[pd.Series, np.ndarray],
        **kwargs: Any
    ) -> float:
        """
        Calculate Gini coefficient.

        The Gini coefficient is calculated as 2 * AUC - 1 and measures
        the discriminative power of the model.

        Args:
            ytrue: True binary labels (0 or 1)
            yprob: Predicted probabilities for the positive class
            **kwargs: Additional keyword arguments passed to get_auc

        Returns:
            float: Gini coefficient between -1 and 1, higher values indicate better discrimination

        Examples:
            >>> metrics = Metrics()
            >>> y_true = [0, 0, 1, 1]
            >>> y_prob = [0.1, 0.4, 0.35, 0.8]
            >>> gini = metrics.get_gini(y_true, y_prob)
            >>> print(f"Gini: {gini:.3f}")
        """
        auc = cls.get_auc(ytrue, yprob, **kwargs)
        gini = 2 * auc - 1

        return gini

    @classmethod
    def get_stat(
        cls,
        df_label: Union[pd.Series, np.ndarray],
        df_feature: Union[pd.Series, np.ndarray],
    ) -> pd.DataFrame:
        """
        Calculate comprehensive statistics for a feature against target labels.

        This method computes various statistics including total counts, bad rates,
        Information Value (IV) for each bin/category of the feature.

        Args:
            df_label: Target binary labels (0 or 1)
            df_feature: Feature values to analyze

        Returns:
            pd.DataFrame: Statistics DataFrame with columns:
                - var: Variable name
                - total: Total count per bin
                - total_ratio: Proportion of total samples per bin
                - bad: Bad count per bin
                - bad_rate: Bad rate per bin
                - iv: Information Value per bin
                - val: Bin/category value

        Examples:
            >>> metrics = Metrics()
            >>> labels = pd.Series([0, 1, 0, 1, 1])
            >>> feature = pd.Series([1, 2, 1, 3, 2], name='test_feature')
            >>> stats = metrics.get_stat(labels, feature)
            >>> print(stats)
        """
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
    def get_iv(
        cls,
        df_label: Union[pd.Series, np.ndarray],
        df_feature: Union[pd.Series, np.ndarray],
    ) -> float:
        """
        Calculate Information Value (IV) for a feature.

        Information Value measures the strength of a feature in distinguishing
        between good and bad customers. Higher IV indicates stronger predictive power.

        IV interpretation:
        - < 0.02: Not useful for prediction
        - 0.02 to 0.1: Weak predictive power
        - 0.1 to 0.3: Medium predictive power
        - 0.3 to 0.5: Strong predictive power
        - > 0.5: Suspicious, likely overfitting

        Args:
            df_label: Target binary labels (0 or 1)
            df_feature: Feature values to analyze

        Returns:
            float: Information Value

        Examples:
            >>> metrics = Metrics()
            >>> labels = pd.Series([0, 1, 0, 1, 1])
            >>> feature = pd.Series([1, 2, 1, 3, 2])
            >>> iv = metrics.get_iv(labels, feature)
            >>> print(f"IV: {iv:.4f}")
        """
        df_stat = cls.get_stat(df_label, df_feature)
        return df_stat["iv"].sum()

    @classmethod
    def get_psi(
        cls,
        df_train: Union[pd.Series, np.ndarray],
        df_test: Union[pd.Series, np.ndarray],
    ) -> float:
        """
        Calculate Population Stability Index (PSI) between training and test datasets.

        PSI measures the difference in distribution between training and test data.
        It's used to detect data drift and ensure model stability.

        PSI interpretation:
        - < 0.1: No significant population change
        - 0.1 to 0.2: Moderate population change
        - > 0.2: Significant population change, model may need retraining

        Args:
            df_train: Training dataset feature values
            df_test: Test dataset feature values

        Returns:
            float: PSI value, lower values indicate more stable populations

        Examples:
            >>> metrics = Metrics()
            >>> train_data = [1, 1, 2, 2, 3, 3]
            >>> test_data = [1, 2, 2, 3, 3, 3]
            >>> psi = metrics.get_psi(train_data, test_data)
            >>> print(f"PSI: {psi:.4f}")
        """
        base_prop = pd.Series(df_train).value_counts(normalize=True, dropna=False)
        test_prop = pd.Series(df_test).value_counts(normalize=True, dropna=False)

        eps = 1e-6
        all_categories = base_prop.index.union(test_prop.index)
        base_prop = base_prop.reindex(all_categories, fill_value=eps)
        test_prop = test_prop.reindex(all_categories, fill_value=eps)

        psi = np.sum((test_prop - base_prop) * np.log(test_prop / base_prop))

        return psi
