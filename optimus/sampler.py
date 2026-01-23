#!/usr/bin/env python
# Version: 0.4.1
# Created: 2025-01-23
# Author: ["Hanyuan Zhang"]

from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from termcolor import cprint


class ImbalanceSampler(BaseEstimator, TransformerMixin):
    """
    Sampling strategies for imbalanced datasets.

    Supports multiple sampling approaches to address class imbalance:
    - Random under-sampling of majority class
    - Random over-sampling of minority class
    - SMOTE-like synthetic sampling (simplified version)
    - Combined under-sampling + over-sampling

    Attributes:
        strategy: Sampling strategy ('under', 'over', 'smote', 'combined')
        target_ratio: Desired ratio of minority to majority class (0-1)
        random_state: Random seed for reproducibility

    Examples:
        >>> # Under-sampling majority class to 1:3 ratio
        >>> sampler = ImbalanceSampler(strategy='under', target_ratio=0.33)
        >>> X_resampled, y_resampled = sampler.fit_resample(X, y)
        >>>
        >>> # Over-sampling minority class to 1:2 ratio
        >>> sampler = ImbalanceSampler(strategy='over', target_ratio=0.5)
        >>> X_resampled, y_resampled = sampler.fit_resample(X, y)
        >>>
        >>> # Combined approach
        >>> sampler = ImbalanceSampler(strategy='combined', target_ratio=0.4)
        >>> X_resampled, y_resampled = sampler.fit_resample(X, y)
    """

    def __init__(
        self,
        strategy: str = "under",
        target_ratio: float = 0.3,
        random_state: Optional[int] = None,
    ) -> None:
        """
        Initialize the sampler.

        Args:
            strategy: Sampling strategy
                - 'under': Random under-sampling of majority class
                - 'over': Random over-sampling of minority class
                - 'smote': SMOTE-like synthetic sampling
                - 'combined': Under-sample majority + over-sample minority
            target_ratio: Target ratio of minority/majority class (0-1)
                E.g., 0.3 means final dataset will have 30% minority, 70% majority
            random_state: Random seed for reproducibility

        Raises:
            ValueError: If invalid strategy or target_ratio
        """
        valid_strategies = ["under", "over", "smote", "combined"]
        if strategy not in valid_strategies:
            raise ValueError(
                f"Invalid strategy '{strategy}'. Must be one of {valid_strategies}"
            )

        if not 0 < target_ratio < 1:
            raise ValueError(
                f"target_ratio must be between 0 and 1, got {target_ratio}"
            )

        self.strategy = strategy
        self.target_ratio = target_ratio
        self.random_state = random_state

    def fit(
        self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]
    ) -> "ImbalanceSampler":
        return self

    def fit_resample(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> tuple:
        unique_vals = np.unique(y)
        if len(unique_vals) != 2:
            raise ValueError(f"Target must be binary, got {unique_vals}")

        class_counts = pd.Series(y).value_counts()
        minority_class = class_counts.idxmin()
        majority_class = class_counts.idxmax()

        minority_mask = y == minority_class
        majority_mask = y == majority_class

        X_minority = X[minority_mask]
        y_minority = y[minority_mask]
        X_majority = X[majority_mask]
        y_majority = y[majority_mask]

        n_minority = len(X_minority)
        n_majority = len(X_majority)

        cprint(
            f"[INFO] Original class distribution: minority={n_minority}, majority={n_majority} "
            f"(ratio={n_minority / n_majority:.3f})",
            "cyan",
        )

        if self.strategy == "under":
            X_resampled, y_resampled = self._undersample(
                X_minority, y_minority, X_majority, y_majority
            )
        elif self.strategy == "over":
            X_resampled, y_resampled = self._oversample(
                X_minority, y_minority, X_majority, y_majority
            )
        elif self.strategy == "smote":
            X_resampled, y_resampled = self._smote(
                X, y, X_minority, y_minority, X_majority, y_majority
            )
        elif self.strategy == "combined":
            X_resampled, y_resampled = self._combined(
                X_minority, y_minority, X_majority, y_majority
            )

        new_minority = (y_resampled == minority_class).sum()
        new_majority = (y_resampled == majority_class).sum()
        cprint(
            f"[INFO] Resampled class distribution: minority={new_minority}, majority={new_majority} "
            f"(ratio={new_minority / new_majority:.3f})",
            "green",
        )

        return X_resampled, y_resampled

    def _undersample(self, X_min, y_min, X_maj, y_maj):
        n_minority = len(X_min)
        n_majority_target = int(n_minority / self.target_ratio) - n_minority

        np.random.seed(self.random_state)
        majority_indices = np.random.choice(
            len(X_maj), size=n_majority_target, replace=False
        )

        X_maj_sampled = X_maj.iloc[majority_indices]
        y_maj_sampled = y_maj.iloc[majority_indices]

        X_resampled = pd.concat([X_min, X_maj_sampled], ignore_index=True)
        y_resampled = pd.concat([y_min, y_maj_sampled], ignore_index=True)

        return X_resampled, y_resampled

    def _oversample(self, X_min, y_min, X_maj, y_maj):
        n_majority = len(X_maj)
        n_minority_target = int(n_majority * self.target_ratio)

        np.random.seed(self.random_state)
        minority_indices = np.random.choice(
            len(X_min), size=n_minority_target, replace=True
        )

        X_min_sampled = X_min.iloc[minority_indices]
        y_min_sampled = y_min.iloc[minority_indices]

        X_resampled = pd.concat([X_min_sampled, X_maj], ignore_index=True)
        y_resampled = pd.concat([y_min_sampled, y_maj], ignore_index=True)

        return X_resampled, y_resampled

    def _smote(self, X, y, X_min, y_min, X_maj, y_maj):
        n_majority = len(X_maj)
        n_minority_target = int(n_majority * self.target_ratio)
        n_synthetic = n_minority_target - len(X_min)

        if n_synthetic <= 0:
            return pd.concat([X_min, X_maj], ignore_index=True), pd.concat(
                [y_min, y_maj], ignore_index=True
            )

        np.random.seed(self.random_state)

        synthetic_samples = []
        for _ in range(n_synthetic):
            idx1, idx2 = np.random.choice(len(X_min), size=2, replace=True)
            sample1 = X_min.iloc[idx1].values
            sample2 = X_min.iloc[idx2].values

            alpha = np.random.random()
            synthetic_sample = sample1 + alpha * (sample2 - sample1)
            synthetic_samples.append(synthetic_sample)

        X_synthetic = pd.DataFrame(
            synthetic_samples,
            columns=X_min.columns,
            index=range(len(synthetic_samples)),
        )
        y_synthetic = pd.Series(
            [y_min.iloc[0]] * len(synthetic_samples),
            index=range(len(synthetic_samples)),
        )

        X_resampled = pd.concat([X_min, X_synthetic, X_maj], ignore_index=True)
        y_resampled = pd.concat([y_min, y_synthetic, y_maj], ignore_index=True)

        return X_resampled, y_resampled

    def _combined(self, X_min, y_min, X_maj, y_maj):
        """Combined under-sampling and over-sampling."""
        n_minority = len(X_min)
        intermediate_ratio = self.target_ratio * 0.5
        n_majority_intermediate = int(n_minority / intermediate_ratio) - n_minority

        np.random.seed(self.random_state)
        majority_indices = np.random.choice(
            len(X_maj), size=n_majority_intermediate, replace=False
        )
        X_maj_sampled = X_maj.iloc[majority_indices]
        y_maj_sampled = y_maj.iloc[majority_indices]

        n_minority_target = int(n_majority_intermediate * self.target_ratio)
        minority_indices = np.random.choice(
            len(X_min), size=n_minority_target, replace=True
        )
        X_min_sampled = X_min.iloc[minority_indices]
        y_min_sampled = y_min.iloc[minority_indices]

        X_resampled = pd.concat([X_min_sampled, X_maj_sampled], ignore_index=True)
        y_resampled = pd.concat([y_min_sampled, y_maj_sampled], ignore_index=True)

        return X_resampled, y_resampled
