#!/usr/bin/env python
# Version: 0.3.0
# Created: 2024-04-07
# Author: ["Hanyuan Zhang"]

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.calibration import CalibrationDisplay
from sklearn.isotonic import IsotonicRegression


class PlattCalibrator(TransformerMixin):
    """
    A comprehensive score calibration system for credit scoring and risk modeling.

    This class provides two calibration strategies to transform predicted
    probabilities into interpretable credit scores or calibrated probabilities.

    Calibration Modes:
    -----------------
    Score Mapping Mode (mapping_base provided):
        Custom score scale with user-defined mapping.
        Transforms probabilities into credit scores based on score-to-probability mapping.
        Requires mapping_base, score_cap, and score_floor parameters.
        Useful for business requirements with specific score ranges.

    Probability Mode (mapping_base is None):
        Probability calibration only (0-1 scale).
        Ensures predicted probabilities match observed frequencies.
        Output is directly calibrated probabilities without score mapping.
        Useful when interpretable probabilities are preferred over scores.

    Attributes:
        n_bins: Number of bins for calibration binning.
        n_degree: Polynomial degree for probability calibration.
        mapping_base: Custom mapping dictionary {score: probability}.
        score_cap: Maximum score value (used with mapping_base).
        score_floor: Minimum score value (used with mapping_base).
        calibrate_detail: Detailed calibration results and diagnostics.
        calibrate_plot: Calibration plot figure (generated in probability mode).

    Examples:
        >>> # Probability calibration (default)
        >>> calibrator = PlattCalibrator()
        >>> calibrator.fit(y_prob, y_true)
        >>> calibrated_probs = calibrator.transform(y_prob_test)

        >>> # Custom score mapping
        >>> calibrator = Calibration(
        ...     mapping_base={600: 0.05, 700: 0.02, 800: 0.01},
        ...     score_cap=850,
        ...     score_floor=300
        ... )
        >>> calibrator.fit(y_prob, y_true)
        >>> scores = calibrator.transform(y_prob_test)

    Notes:
    -----
    Always validate calibration quality using:
    - calibrate_detail: Check calibration statistics and reliability
    - get_calibrate_plot(): Visualize calibration curve (always generated)
    - compare_calibrate_result(): Analyze score distribution and performance
    """

    def __init__(
        self,
        n_bins: int = 25,
        n_degree: int = 1,
        mapping_base: Optional[Dict[int, float]] = None,
        score_cap: Optional[float] = None,
        score_floor: Optional[float] = None,
    ) -> None:
        """
        Initialize the Calibration transformer.

        Args:
            n_bins: Number of bins for calibration analysis and probability binning.
            n_degree: Polynomial degree for probability calibration curve fitting.
            mapping_base: Custom score-to-probability mapping. If provided, transforms
                probabilities to credit scores. If None (default), outputs calibrated
                probabilities directly. Format: {score: probability}, e.g., {600: 0.05}.
            score_cap: Maximum score value (required when mapping_base is provided).
            score_floor: Minimum score value (required when mapping_base is provided).

        Examples:
            >>> # Probability calibration (default)
            >>> calibrator = PlattCalibrator()

            >>> # Custom score mapping
            >>> calibrator = Calibration(
            ...     mapping_base={500: 0.1, 600: 0.05, 700: 0.02},
            ...     score_cap=800,
            ...     score_floor=400
            ... )
        """
        self.mapping_base = mapping_base
        self.score_cap = score_cap
        self.score_floor = score_floor
        self.n_bins = n_bins
        self.n_degree = n_degree
        self._use_score_mapping = mapping_base is not None

        if self._use_score_mapping:
            if score_cap is None or score_floor is None:
                raise ValueError(
                    "score_cap and score_floor are required when mapping_base is provided."
                )
            if score_floor >= score_cap:
                raise ValueError(
                    f"score_floor ({score_floor}) must be less than score_cap ({score_cap})."
                )
        else:
            self.score_floor = 0
            self.score_cap = 1

        # Results and fitted parameters
        self.calibrate_detail: Optional[pd.DataFrame] = None
        self.calibrate_coef: Optional[np.ndarray] = None
        self.mapping_intercept: Optional[float] = None
        self.mapping_slope: Optional[float] = None

        # Calibration plot figure (generated during fit)
        self.calibrate_plot: Optional[plt.Figure] = None

    def fit(
        self,
        df_prob: Union[pd.Series, np.ndarray],
        df_label: Union[pd.Series, np.ndarray],
    ) -> "PlattCalibrator":
        """
        Fit the calibration model using predicted probabilities and true labels.

        This method learns the calibration parameters by analyzing the relationship
        between predicted probabilities and observed frequencies.

        Args:
            df_prob: Predicted probabilities from the model
            df_label: True binary labels (0 or 1)

        Returns:
            self: Fitted calibration instance

        Examples:
            >>> calibrator = PlattCalibrator()
            >>> calibrator.fit(y_prob_train, y_true_train)
            >>> print("Calibration fitted successfully")
        """
        if self.mapping_base is not None:
            # Score mapping mode: use provided mapping_base
            self._use_score_mapping = True
            logging.info(
                "Score mapping mode: using provided mapping_base, score_cap, and score_floor"
            )
            self.mapping_slope, self.mapping_intercept = self.__set_mapping_base(
                self.mapping_base
            )
        else:
            # Probability mode: output calibrated probabilities directly
            self._use_score_mapping = False
            logging.info("Probability mode: output calibrated probabilities directly")

        lst_prob = self.__check_type(df_prob)
        lst_label = self.__check_type(df_label)

        df_data = pd.DataFrame(
            {
                "yprob": lst_prob,
                "label": lst_label,
                "lnodds_prob": [self.prob2lnodds(x) for x in lst_prob],
            }
        )
        df_data["lnodds_prob_bin"] = pd.qcut(
            df_data["lnodds_prob"], self.n_bins, duplicates="drop"
        )

        df_cal = df_data.groupby("lnodds_prob_bin").agg(
            total=("label", "count"),
            bad_rate=("label", "mean"),
            lnodds_prob_mean_x=("lnodds_prob", "mean"),
        )
        df_cal["adj_bad_rate"] = df_cal.apply(
            lambda x: max(x["bad_rate"], 1 / x["total"], 0.0001), axis=1
        )
        df_cal["lnodds_bad_rate_y"] = df_cal["adj_bad_rate"].apply(
            lambda x: self.prob2lnodds(x)
        )

        lst_col = [
            "total",
            "bad_rate",
            "adj_bad_rate",
            "lnodds_prob_mean_x",
            "lnodds_bad_rate_y",
        ]
        self.calibrate_detail = df_cal[lst_col]

        # Fit polynomial with error handling
        try:
            self.calibrate_coef = np.polyfit(
                df_cal["lnodds_prob_mean_x"],
                df_cal["lnodds_bad_rate_y"],
                self.n_degree,
            )
        except (np.linalg.LinAlgError, ValueError) as e:
            logging.warning(
                f"Polynomial fitting failed with degree {self.n_degree}: {e}. "
                "Falling back to linear calibration (degree=1)."
            )
            try:
                self.calibrate_coef = np.polyfit(
                    df_cal["lnodds_prob_mean_x"],
                    df_cal["lnodds_bad_rate_y"],
                    1,  # Fallback to linear
                )
                self.n_degree = 1  # Update degree to reflect actual fit
            except (np.linalg.LinAlgError, ValueError) as e2:
                logging.error(
                    f"Linear fitting also failed: {e2}. "
                    "Using identity calibration (no adjustment)."
                )
                # Identity calibration: y = x (coefficients [1, 0])
                self.calibrate_coef = np.array([1.0, 0.0])
                self.n_degree = 1

        # Generate and store calibration plot (always generate regardless of mode)
        # Use probability outputs for the plot even in score mapping mode
        lst_lnodds_prob = [self.prob2lnodds(x) for x in lst_prob]
        lst_lnodds_cal_prob = [
            np.poly1d(self.calibrate_coef)(x) for x in lst_lnodds_prob
        ]
        y_prob_after = np.array([self.lnodds2prob(x) for x in lst_lnodds_cal_prob])
        self.calibrate_plot = self._generate_calibrate_plot(
            y_true=np.array(lst_label),
            y_prob_before=np.array(lst_prob),
            y_prob_after=y_prob_after,
        )

        return self

    def transform(self, df_prob):
        """
        Transform predicted probabilities to calibrated scores or probabilities.

        Applies the fitted calibration model to convert raw probabilities
        into either calibrated probabilities or mapped credit scores.

        Args:
            df_prob: Predicted probabilities from the classifier.
                Can be pd.Series, np.ndarray, list, or pd.DataFrame.

        Returns:
            np.ndarray: Calibrated output:
                - If mapping_base provided: Credit scores (bounded by score_cap/score_floor)
                - If mapping_base is None: Calibrated probabilities (0-1 scale)

        Example:
            >>> scores = calibrator.transform(y_prob_test)
        """
        lst_prob = self.__check_type(df_prob)
        lst_lnodds_prob = [self.prob2lnodds(x) for x in lst_prob]
        lst_lnodds_cal_prob = [
            np.poly1d(self.calibrate_coef)(x) for x in lst_lnodds_prob
        ]

        if self._use_score_mapping:
            # Score mapping mode: convert to credit scores
            lst_score = [
                self.mapping_intercept + self.mapping_slope * x
                for x in lst_lnodds_cal_prob
            ]
            lst_score = [max(x, self.score_floor) for x in lst_score]
            lst_score = [min(x, self.score_cap) for x in lst_score]
            return np.array(lst_score)
        else:
            # Probability mode: return calibrated probabilities
            lst_cal_prob = [self.lnodds2prob(x) for x in lst_lnodds_cal_prob]
            return np.array(lst_cal_prob)

    def compare_calibrate_result(self, df_score, df_label, bins=None):
        """
        Compare calibration results by generating a detailed scorecard analysis.

        Analyzes score distribution, bad rates, lift metrics, and KS/IV statistics
        across score bins to evaluate calibration quality.

        Args:
            df_score: Calibrated scores from transform().
            df_label: True binary labels (0 or 1).
            bins: Score bin boundaries (optional).
                - For probability mode: use bins like [0, 0.1, 0.2, ..., 1.0]
                - For score mapping mode: use bins like [300, 400, 500, ..., 1000]
                - If None, will create n_bins equal-frequency (quantile) bins

        Returns:
            pd.DataFrame: Scorecard with columns including:
                - score_bin: Score range intervals
                - total, total_pct: Sample counts and percentages
                - bad_rate, good_rate: Observed rates per bin
                - approval_rate, bad_aft_rate: Cumulative rates
                - odds_ratio, inv_odds_ratio: Odds comparison metrics
                - ks, iv: KS statistic and Information Value

        Example:
            >>> # Use automatic quantile binning
            >>> scorecard = calibrator.compare_calibrate_result(scores, y_test)
            >>>
            >>> # Or provide custom bins
            >>> scorecard = calibrator.compare_calibrate_result(
            ...     scores, y_test, bins=[0, 500, 600, 700, 800, 1000]
            ... )
        """
        lst_score = self.__check_type(df_score)
        lst_label = self.__check_type(df_label)
        df_data = pd.DataFrame({"score": lst_score, "label": lst_label})

        if bins is None:
            # Use n_bins to create equal-frequency (quantile) bins
            df_data["score_bin"] = pd.qcut(
                df_data["score"], self.n_bins, duplicates="drop"
            )
        else:
            df_data["score_bin"] = pd.cut(df_data["score"], bins)

        eps = np.finfo(np.float32).eps
        df_res = df_data.groupby("score_bin").agg(
            total=("label", "count"),
            total_pct=("label", lambda x: len(x) / (len(df_data) + eps)),
            total_good=("label", lambda x: len(x) - sum(x)),
            good_rate=("label", lambda x: (len(x) - sum(x)) / (len(x) + eps)),
            total_bad=("label", "sum"),
            bad_rate=("label", "mean"),
        )
        df_res = df_res.reset_index()

        bad_bef_rate = []
        good_bef_rate = []
        bad_aft_rate = []
        good_aft_rate = []
        approval_rate = []
        for idx, _ in df_res.iterrows():
            with np.errstate(divide="ignore", invalid="ignore"):
                approval_rate.append(df_res.iloc[idx:]["total_pct"].sum())
                # Before rates: cumulative from start to current bin (inclusive)
                bad_bef_rate.append(
                    df_res.iloc[: idx + 1]["total_bad"].sum()
                    / (df_res.iloc[: idx + 1]["total"].sum() + eps)
                )
                good_bef_rate.append(
                    df_res.iloc[: idx + 1]["total_good"].sum()
                    / (df_res.iloc[: idx + 1]["total"].sum() + eps)
                )
                # After rates: cumulative from current bin to end
                bad_aft_rate.append(
                    df_res.iloc[idx:]["total_bad"].sum()
                    / (df_res.iloc[idx:]["total"].sum() + eps)
                )
                good_aft_rate.append(
                    df_res.iloc[idx:]["total_good"].sum()
                    / (df_res.iloc[idx:]["total"].sum() + eps)
                )
        df_res["approval_rate"] = approval_rate
        df_res["bad_bef_rate"] = bad_bef_rate
        df_res["good_bef_rate"] = good_bef_rate
        df_res["bad_aft_rate"] = bad_aft_rate
        df_res["good_aft_rate"] = good_aft_rate

        df_res["score_max"] = df_res["score_bin"].apply(lambda x: x.right)
        df_res["total_cum_pct"] = df_res["total_pct"].cumsum()
        df_res["good_dist"] = df_res["total_good"] / (df_res["total_good"].sum() + eps)
        df_res["bad_dist"] = df_res["total_bad"] / (df_res["total_bad"].sum() + eps)

        if self._use_score_mapping:
            df_res["exp_bad_rate"] = df_res["score_max"].apply(
                lambda x: self.lnodds2prob(
                    (x - self.mapping_intercept) / self.mapping_slope
                )
            )
        else:
            # Probability mode: score_max is the probability itself
            df_res["exp_bad_rate"] = df_res["score_max"]

        df_res["iv"] = (
            (df_res["bad_dist"] - df_res["good_dist"])
            * np.log((df_res["bad_dist"] + eps) / (df_res["good_dist"] + eps))
        ).sum()
        df_res["cum_good_dist"] = df_res["good_dist"].cumsum()
        df_res["cum_bad_dist"] = df_res["bad_dist"].cumsum()
        df_res["ks"] = (df_res["cum_bad_dist"] - df_res["cum_good_dist"]).abs()
        # Odds ratio between the odds before (inclusive) and after the bin to decide score cut-off
        # if max(odds_ratio) >= max(inv_odds_ratio), take the bin (a,b] that achieve the max odds_ratio and set score <= b
        # if max(odds_ratio) < max(inv_odds_ratio), take the bin (a,b] that achieve the max inv_odds_ratio and set score > a
        df_res["odds_bef"] = df_res["total_bad"].cumsum() / (
            df_res["total_good"].cumsum() + eps
        )
        df_res["odds_aft"] = (df_res["total_bad"] - df_res["total_bad"].cumsum()) / (
            df_res["total_good"] - df_res["total_good"].cumsum() + eps
        )
        df_res["odds_ratio"] = df_res["odds_bef"] / df_res["odds_aft"]
        df_res["inv_odds_ratio"] = df_res["odds_aft"] / df_res["odds_bef"]

        lst_col = [
            "score_bin",
            "total",
            "total_pct",
            "total_cum_pct",
            "total_good",
            "good_rate",
            "good_dist",
            "good_bef_rate",
            "good_aft_rate",
            "total_bad",
            "bad_rate",
            "bad_dist",
            "bad_bef_rate",
            "bad_aft_rate",
            "approval_rate",
            "exp_bad_rate",
            "odds_ratio",
            "inv_odds_ratio",
            "ks",
            "iv",
        ]
        df_res = df_res[lst_col]
        return df_res

    def get_bad_rate(self, score_min, score_max, step):
        """
        Generate expected bad rate table for a score range.

        Calculates the theoretical bad rate at each score point based on
        the fitted calibration mapping. Useful for score interpretation
        and setting risk thresholds.

        Args:
            score_min: Minimum score value for the range.
            score_max: Maximum score value for the range.
            step: Step size between score points.

        Returns:
            pd.DataFrame: Table with columns:
                - score: Score values from score_min to score_max
                - bad_rate: Expected bad rate at each score
                - lnodds: Log-odds value at each score

        Raises:
            ValueError: If in probability mode (no score mapping available).

        Example:
            >>> bad_rate_table = calibrator.get_bad_rate(300, 850, 50)
            >>> print(bad_rate_table)
        """
        if not self._use_score_mapping:
            raise ValueError(
                "get_bad_rate() is not available in probability mode. "
                "Provide mapping_base to use score mapping mode."
            )

        ary_score = np.arange(score_min, score_max, step)
        ary_lnodds = (ary_score - self.mapping_intercept) / self.mapping_slope
        ary_bad_rate = self.lnodds2prob(ary_lnodds)
        return pd.DataFrame(
            {"score": ary_score, "bad_rate": ary_bad_rate, "lnodds": ary_lnodds}
        )

    def _generate_calibrate_plot(
        self,
        y_true: np.ndarray,
        y_prob_before: np.ndarray,
        y_prob_after: np.ndarray,
        n_bins: int = 10,
        strategy: str = "uniform",
        title: str = "Calibration Curve (Before vs After)",
    ) -> plt.Figure:
        """
        Generate a calibration plot showing before and after calibration curves.

        Uses sklearn's CalibrationDisplay to plot calibration curves. The plot shows
        two curves: one for the original predictions (blue) and one for the calibrated
        predictions (orange), along with the perfect calibration diagonal.

        Args:
            y_true: True binary labels.
            y_prob_before: Predicted probabilities before calibration.
            y_prob_after: Predicted probabilities after calibration.
            n_bins: Number of bins for the calibration curve (default: 10).
            strategy: Strategy for binning - 'uniform' or 'quantile' (default: 'uniform').
            title: Title for the plot (default: 'Calibration Curve (Before vs After)').

        Returns:
            matplotlib.figure.Figure: The calibration plot figure.
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot before calibration curve (blue)
        CalibrationDisplay.from_predictions(
            y_true,
            y_prob_before,
            n_bins=n_bins,
            strategy=strategy,
            ax=ax,
            name="Before Calibration",
            color="blue",
        )

        # Plot after calibration curve (orange)
        CalibrationDisplay.from_predictions(
            y_true,
            y_prob_after,
            n_bins=n_bins,
            strategy=strategy,
            ax=ax,
            name="After Calibration",
            color="orange",
        )

        ax.set_title(title)
        ax.legend(loc="lower right")
        fig.tight_layout()
        plt.close(fig)

        return fig

    def get_calibrate_plot(self) -> Optional[plt.Figure]:
        """
        Get the calibration plot figure generated during fit().

        Returns the stored calibration plot showing before and after calibration curves.
        This plot is automatically generated during fit() for all calibration modes.

        Returns:
            matplotlib.figure.Figure: The calibration plot figure, or None if fit() not called.

        Example:
            >>> calibrator = PlattCalibrator()
            >>> calibrator.fit(y_prob_train, y_train)
            >>> fig = calibrator.get_calibrate_plot()
            >>> fig.savefig('calibration.png')
        """
        return self.calibrate_plot

    def get_lnodds_calibrate_plot(self) -> plt.Axes:
        """
        Plot the log-odds calibration result (legacy method).

        The x-axis is the ln(odds(y_hat)) and the y-axis is the bad rate.
        Ideally, the points should be close to the fitted line.

        Returns:
            matplotlib.axes.Axes: The plot axes.
        """
        x = self.calibrate_detail["lnodds_prob_mean_x"]
        y_actual = self.calibrate_detail["lnodds_bad_rate_y"]

        y_pred = np.poly1d(self.calibrate_coef)(x)
        f, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x, y_actual, "o", x, y_pred, "-", label="1d")
        ax.set_xlabel("lnodds_prob_mean"), ax.set_ylabel("lnodds_bad_rate")
        return ax

    @classmethod
    def __set_mapping_base(cls, dict_base):
        # Linear regression for mapping base
        # Sort by score, keeping score-bad_rate pairs together
        items = sorted(dict_base.items(), key=lambda x: x[0])
        lst_score = [item[0] for item in items]
        lst_bad_rate = [item[1] for item in items]
        lst_lnodds_bad_rate = [cls.prob2lnodds(x) for x in lst_bad_rate]

        score_max, score_min = lst_score[-1], lst_score[0]
        lnodds_max, lnodds_min = lst_lnodds_bad_rate[-1], lst_lnodds_bad_rate[0]

        slope = (score_max - score_min) / (lnodds_max - lnodds_min)
        intercept = score_max - slope * lnodds_max
        return slope, intercept

    @classmethod
    def __check_type(cls, data):
        if isinstance(data, (list, pd.Series, np.ndarray)):
            lst_data = list(data)
        elif isinstance(data, pd.DataFrame):
            lst_data = data[data.columns.item()].tolist()
        else:
            raise TypeError("Expected data type: DataFrame, List, Series or Array")
        return lst_data

    @classmethod
    def prob2lnodds(cls, prob):
        if prob == 0:
            lnodds = np.log(np.finfo(float).eps)
        elif prob == 1:
            lnodds = np.log(prob / (1 - prob + np.finfo(float).eps))
        else:
            lnodds = np.log(prob / (1 - prob))
        return lnodds

    @classmethod
    def lnodds2prob(cls, lnodds):
        prob = 1 - 1 / (np.exp(lnodds) + 1)
        return prob


class IsotonicCalibrator(TransformerMixin):
    """
    Isotonic regression based calibration with linear scaling.

    This calibrator uses isotonic regression to calibrate probabilities,
    ensuring monotonicity while respecting risk direction (higher score = higher risk).
    After isotonic regression, it applies linear scaling to spread probabilities
    across the [0, 1] range.

    Workflow:
    1. Fit:
       - Apply isotonic regression directly on all data to calibrate probabilities
       - Apply linear scaling to spread calibrated probabilities:
         * If scale_threshold is None: scale all probabilities to [0, 1]
         * If scale_threshold is set: only scale probabilities above threshold to [threshold, 1]
           This enhances separation in the high-risk region.
       - Store slope and intercept for transform
    2. Transform:
       - Apply isotonic regression transform
       - Apply linear scaling with stored slope/intercept (segmented or full)
       - Clip results to [score_floor, score_cap]

    Important Note on scale_threshold:
        scale_threshold applies to the OUTPUT of isotonic regression, not the raw probabilities.
        When set (e.g., 0.3), it creates a two-region calibration:
        - Low risk region (isotonic output < 0.3): Keep isotonic values as-is
        - High risk region (isotonic output >= 0.3): Linearly scale to [0.3, 1.0]

        This is useful for enhancing discrimination in high-risk populations where
        decision-making is more sensitive to score differences.

    Attributes:
        n_bins: Number of quantile bins for scorecard analysis in compare_calibrate_result().
            Not used during fitting (isotonic regression determines optimal binning automatically).
            When bins parameter is not provided to compare_calibrate_result(), data will be
            divided into n_bins equal-frequency (quantile) bins to ensure no empty bins.
        score_floor: Minimum score value (for clipping).
        score_cap: Maximum score value (for clipping).
        scale_threshold: Optional threshold for segmented scaling (applied to isotonic output).
            If set, only isotonic-calibrated probabilities above this threshold are
            linearly scaled to enhance high-risk separation. If None, all probabilities
            are scaled uniformly to [0, 1].
        isotonic_regressor_: Fitted IsotonicRegression model.
        mapping_slope: Slope for linear scaling after isotonic regression.
        mapping_intercept: Intercept for linear scaling after isotonic regression.
        calibrate_detail: Detailed calibration fitting results (binned statistics).
        calibrate_plot: Calibration plot figure (generated during fit).

    Examples:
        >>> # Standard calibration - scale all probabilities to [0, 1]
        >>> calibrator = IsotonicCalibrator(n_bins=20, score_floor=0.0, score_cap=1.0)
        >>> calibrator.fit(y_prob, y_true)
        >>> calibrated_probs = calibrator.transform(y_prob_test)

        >>> # Segmented calibration - only scale high-risk probabilities
        >>> calibrator = IsotonicCalibrator(
        ...     n_bins=20,
        ...     score_floor=0.0,
        ...     score_cap=1.0,
        ...     scale_threshold=0.3
        ... )
        >>> calibrator.fit(y_prob, y_true)
        >>> calibrated_probs = calibrator.transform(y_prob_test)
    """

    def __init__(
        self,
        n_bins: int = 25,
        score_floor: float = 0.0,
        score_cap: float = 1.0,
        scale_threshold: Optional[float] = None,
    ) -> None:
        """
        Initialize the IsotonicCalibrator.

        Args:
            n_bins: Number of quantile bins for scorecard analysis in compare_calibrate_result().
                Not used during fitting (isotonic regression auto-determines optimal binning).
                When bins parameter is not provided, data will be divided into n_bins
                equal-frequency bins to ensure no empty bins.
            score_floor: Minimum score value (for clipping output).
            score_cap: Maximum score value (for clipping output).
            scale_threshold: Optional threshold for segmented scaling. If provided, only
                probabilities above this threshold (after isotonic regression) will be
                linearly scaled to spread across [threshold_value, 1]. Probabilities below
                the threshold remain unchanged from isotonic regression output.
                If None (default), all probabilities are scaled to [0, 1].

        Examples:
            >>> # Standard calibration - scale all probabilities to [0, 1]
            >>> calibrator = IsotonicCalibrator(n_bins=20, score_floor=0.0, score_cap=1.0)

            >>> # Segmented calibration - only scale probabilities above 0.3
            >>> calibrator = IsotonicCalibrator(
            ...     n_bins=20,
            ...     score_floor=0.0,
            ...     score_cap=1.0,
            ...     scale_threshold=0.3
            ... )
        """
        self.n_bins = n_bins
        self.score_floor = score_floor
        self.score_cap = score_cap
        self.scale_threshold = scale_threshold

        # Fitted parameters
        self.isotonic_regressor_: Optional[IsotonicRegression] = None
        self.mapping_slope: Optional[float] = None
        self.mapping_intercept: Optional[float] = None

        # Results and diagnostics
        self.calibrate_detail: Optional[pd.DataFrame] = None
        self.calibrate_plot: Optional[plt.Figure] = None

        # Validation
        if score_floor >= score_cap:
            raise ValueError(
                f"score_floor ({score_floor}) must be less than score_cap ({score_cap})."
            )

        if scale_threshold is not None:
            if not (0.0 <= scale_threshold <= 1.0):
                raise ValueError(
                    f"scale_threshold ({scale_threshold}) must be between 0 and 1."
                )

    def fit(
        self,
        df_prob: Union[pd.Series, np.ndarray],
        df_label: Union[pd.Series, np.ndarray],
    ) -> "IsotonicCalibrator":
        """
        Fit the isotonic calibration model with linear scaling.

        Process:
        1. Fit isotonic regression directly on all data points (increasing=True for higher score = higher risk)
        2. Transform training probabilities using isotonic regression
        3. Fit linear scaling to spread calibrated probabilities across [0, 1]
        4. Store slope and intercept for later transform

        Note: Isotonic regression automatically handles binary labels and finds the optimal
        monotonic step function. The n_bins parameter is not used during fitting, but is
        used in compare_calibrate_result() for scorecard analysis.

        Args:
            df_prob: Predicted probabilities from the model.
            df_label: True binary labels (0 or 1).

        Returns:
            self: Fitted calibrator instance.

        Examples:
            >>> calibrator = IsotonicCalibrator()
            >>> calibrator.fit(y_prob_train, y_true_train)
        """
        lst_prob = self.__check_type(df_prob)
        lst_label = self.__check_type(df_label)

        # Fit isotonic regression directly on all data points
        # IsotonicRegression can handle binary labels and will find optimal step function
        # increasing=True means higher score -> higher risk
        self.isotonic_regressor_ = IsotonicRegression(
            y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip"
        )
        self.isotonic_regressor_.fit(np.array(lst_prob), np.array(lst_label))

        # Transform training probabilities using isotonic regression
        y_prob_isotonic = self.isotonic_regressor_.transform(np.array(lst_prob))

        # Linear scaling to spread probabilities
        if self.scale_threshold is None:
            # Standard scaling: map [min, max] to [0, 1]
            min_prob = y_prob_isotonic.min()
            max_prob = y_prob_isotonic.max()

            if max_prob - min_prob < 1e-10:
                # Edge case: all probabilities are the same
                logging.warning(
                    "Isotonic regression produced constant probabilities. "
                    "Using identity scaling."
                )
                self.mapping_slope = 1.0
                self.mapping_intercept = 0.0
            else:
                # slope * x + intercept maps [min_prob, max_prob] to [0, 1]
                self.mapping_slope = 1.0 / (max_prob - min_prob)
                self.mapping_intercept = -min_prob * self.mapping_slope

            # Apply scaling to all probabilities
            y_prob_after = y_prob_isotonic * self.mapping_slope + self.mapping_intercept
        else:
            # Segmented scaling: only scale probabilities above threshold
            # Find max probability above threshold
            high_prob_mask = y_prob_isotonic >= self.scale_threshold

            if not np.any(high_prob_mask):
                # No probabilities above threshold
                logging.warning(
                    f"No probabilities above threshold {self.scale_threshold}. "
                    "Using identity scaling."
                )
                self.mapping_slope = 1.0
                self.mapping_intercept = 0.0
                y_prob_after = y_prob_isotonic.copy()
            else:
                high_probs = y_prob_isotonic[high_prob_mask]
                min_high_prob = high_probs.min()
                max_high_prob = high_probs.max()

                if max_high_prob - min_high_prob < 1e-10:
                    # All high probabilities are the same
                    logging.warning(
                        f"All probabilities above threshold {self.scale_threshold} are constant. "
                        "Using identity scaling for high probabilities."
                    )
                    self.mapping_slope = 1.0
                    self.mapping_intercept = 0.0
                else:
                    # Scale [min_high_prob, max_high_prob] to [scale_threshold, 1]
                    self.mapping_slope = (1.0 - self.scale_threshold) / (
                        max_high_prob - min_high_prob
                    )
                    self.mapping_intercept = (
                        self.scale_threshold - min_high_prob * self.mapping_slope
                    )

                # Apply segmented scaling
                y_prob_after = y_prob_isotonic.copy()
                y_prob_after[high_prob_mask] = (
                    y_prob_isotonic[high_prob_mask] * self.mapping_slope
                    + self.mapping_intercept
                )

        # Clip to bounds
        y_prob_after = np.clip(y_prob_after, self.score_floor, self.score_cap)

        # Generate calibrate_detail DataFrame for diagnostics
        df_data = pd.DataFrame(
            {
                "yprob": lst_prob,
                "label": lst_label,
                "prob_isotonic": y_prob_isotonic,
                "prob_scaled": y_prob_after,
            }
        )

        # Bin the data using quantile binning for analysis
        df_data["prob_bin"] = pd.qcut(df_data["yprob"], self.n_bins, duplicates="drop")

        df_cal = df_data.groupby("prob_bin").agg(
            total=("label", "count"),
            bad_rate=("label", "mean"),
            prob_mean=("yprob", "mean"),
            isotonic_mean=("prob_isotonic", "mean"),
            scaled_mean=("prob_scaled", "mean"),
        )

        self.calibrate_detail = df_cal[
            [
                "total",
                "bad_rate",
                "prob_mean",
                "isotonic_mean",
                "scaled_mean",
            ]
        ]

        # Generate calibration plot
        self.calibrate_plot = self._generate_calibrate_plot(
            y_true=np.array(lst_label),
            y_prob_before=np.array(lst_prob),
            y_prob_after=y_prob_after,
        )

        if self.scale_threshold is None:
            logging.info(
                f"Isotonic calibration fitted (full scaling). "
                f"Slope={self.mapping_slope:.4f}, intercept={self.mapping_intercept:.4f}"
            )
        else:
            logging.info(
                f"Isotonic calibration fitted (segmented scaling above {self.scale_threshold}). "
                f"Slope={self.mapping_slope:.4f}, intercept={self.mapping_intercept:.4f}"
            )

        return self

    def transform(self, df_prob: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """
        Transform probabilities using fitted isotonic regression and linear scaling.

        Process:
        1. Apply isotonic regression transform
        2. Apply linear scaling:
           - If scale_threshold is None: scale all probabilities
           - If scale_threshold is set: only scale probabilities above threshold
        3. Clip to [score_floor, score_cap]

        Args:
            df_prob: Predicted probabilities from the classifier.

        Returns:
            np.ndarray: Calibrated probabilities (clipped to [score_floor, score_cap]).

        Example:
            >>> calibrated_probs = calibrator.transform(y_prob_test)
        """
        if self.isotonic_regressor_ is None:
            raise ValueError("Calibrator must be fitted before transform()")

        lst_prob = self.__check_type(df_prob)

        # Apply isotonic regression
        y_prob_isotonic = self.isotonic_regressor_.transform(np.array(lst_prob))

        # Apply linear scaling
        if self.scale_threshold is None:
            # Standard scaling: apply to all probabilities
            y_prob_scaled = (
                y_prob_isotonic * self.mapping_slope + self.mapping_intercept
            )
        else:
            # Segmented scaling: only apply to probabilities above threshold
            y_prob_scaled = y_prob_isotonic.copy()
            high_prob_mask = y_prob_isotonic >= self.scale_threshold
            y_prob_scaled[high_prob_mask] = (
                y_prob_isotonic[high_prob_mask] * self.mapping_slope
                + self.mapping_intercept
            )

        # Clip to bounds
        y_prob_scaled = np.clip(y_prob_scaled, self.score_floor, self.score_cap)

        return y_prob_scaled

    def _generate_calibrate_plot(
        self,
        y_true: np.ndarray,
        y_prob_before: np.ndarray,
        y_prob_after: np.ndarray,
        n_bins: int = 10,
        strategy: str = "uniform",
        title: str = "Isotonic Calibration Curve (Before vs After)",
    ) -> plt.Figure:
        """
        Generate a calibration plot showing before and after calibration curves.

        Uses sklearn's CalibrationDisplay to plot calibration curves. The plot shows
        two curves: one for the original predictions (blue) and one for the calibrated
        predictions (orange), along with the perfect calibration diagonal.

        Args:
            y_true: True binary labels.
            y_prob_before: Predicted probabilities before calibration.
            y_prob_after: Predicted probabilities after calibration.
            n_bins: Number of bins for the calibration curve (default: 10).
            strategy: Strategy for binning - 'uniform' or 'quantile' (default: 'uniform').
            title: Title for the plot.

        Returns:
            matplotlib.figure.Figure: The calibration plot figure.
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot before calibration curve (blue)
        CalibrationDisplay.from_predictions(
            y_true,
            y_prob_before,
            n_bins=n_bins,
            strategy=strategy,
            ax=ax,
            name="Before Calibration",
            color="blue",
        )

        # Plot after calibration curve (orange)
        CalibrationDisplay.from_predictions(
            y_true,
            y_prob_after,
            n_bins=n_bins,
            strategy=strategy,
            ax=ax,
            name="After Calibration",
            color="orange",
        )

        ax.set_title(title)
        ax.legend(loc="lower right")
        fig.tight_layout()
        plt.close(fig)

        return fig

    def get_calibrate_plot(self) -> Optional[plt.Figure]:
        """
        Get the calibration plot figure generated during fit().

        Returns:
            matplotlib.figure.Figure: The calibration plot figure, or None if fit() not called.

        Example:
            >>> calibrator = IsotonicCalibrator()
            >>> calibrator.fit(y_prob_train, y_train)
            >>> fig = calibrator.get_calibrate_plot()
            >>> fig.savefig('isotonic_calibration.png')
        """
        return self.calibrate_plot

    def compare_calibrate_result(
        self,
        df_score: Union[pd.Series, np.ndarray],
        df_label: Union[pd.Series, np.ndarray],
        bins: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """
        Compare calibration results by generating a detailed scorecard analysis.

        Analyzes score distribution, bad rates, lift metrics, and KS/IV statistics
        across score bins to evaluate calibration quality.

        Args:
            df_score: Calibrated scores from transform().
            df_label: True binary labels (0 or 1).
            bins: Score bin boundaries (e.g., [0, 0.1, 0.2, ..., 1.0]).
                If None, will create n_bins equal-frequency (quantile) bins to ensure
                each bin has roughly equal sample size and no empty bins.

        Returns:
            pd.DataFrame: Scorecard with columns including:
                - score_bin: Score range intervals
                - total, total_pct: Sample counts and percentages
                - bad_rate, good_rate: Observed rates per bin
                - approval_rate, bad_aft_rate: Cumulative rates
                - odds_ratio, inv_odds_ratio: Odds comparison metrics
                - ks, iv: KS statistic and Information Value

        Example:
            >>> # Use automatic quantile binning (default)
            >>> scorecard = calibrator.compare_calibrate_result(scores, y_test)

            >>> # Or provide custom bin boundaries
            >>> scorecard = calibrator.compare_calibrate_result(
            ...     scores, y_test, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            ... )
        """
        lst_score = self.__check_type(df_score)
        lst_label = self.__check_type(df_label)
        df_data = pd.DataFrame({"score": lst_score, "label": lst_label})

        if bins is None:
            # Use n_bins to create equal-frequency (quantile) bins
            # duplicates='drop' handles cases where there are duplicate values
            df_data["score_bin"] = pd.qcut(
                df_data["score"], self.n_bins, duplicates="drop"
            )
        else:
            # Use user-provided bin boundaries
            df_data["score_bin"] = pd.cut(df_data["score"], bins)

        eps = np.finfo(np.float32).eps
        df_res = df_data.groupby("score_bin").agg(
            total=("label", "count"),
            total_pct=("label", lambda x: len(x) / (len(df_data) + eps)),
            total_good=("label", lambda x: len(x) - sum(x)),
            good_rate=("label", lambda x: (len(x) - sum(x)) / (len(x) + eps)),
            total_bad=("label", "sum"),
            bad_rate=("label", "mean"),
        )
        df_res = df_res.reset_index()

        bad_bef_rate = []
        good_bef_rate = []
        bad_aft_rate = []
        good_aft_rate = []
        approval_rate = []
        for idx, _ in df_res.iterrows():
            with np.errstate(divide="ignore", invalid="ignore"):
                approval_rate.append(df_res.iloc[idx:]["total_pct"].sum())
                # Before rates: cumulative from start to current bin (inclusive)
                bad_bef_rate.append(
                    df_res.iloc[: idx + 1]["total_bad"].sum()
                    / (df_res.iloc[: idx + 1]["total"].sum() + eps)
                )
                good_bef_rate.append(
                    df_res.iloc[: idx + 1]["total_good"].sum()
                    / (df_res.iloc[: idx + 1]["total"].sum() + eps)
                )
                # After rates: cumulative from current bin to end
                bad_aft_rate.append(
                    df_res.iloc[idx:]["total_bad"].sum()
                    / (df_res.iloc[idx:]["total"].sum() + eps)
                )
                good_aft_rate.append(
                    df_res.iloc[idx:]["total_good"].sum()
                    / (df_res.iloc[idx:]["total"].sum() + eps)
                )
        df_res["approval_rate"] = approval_rate
        df_res["bad_bef_rate"] = bad_bef_rate
        df_res["good_bef_rate"] = good_bef_rate
        df_res["bad_aft_rate"] = bad_aft_rate
        df_res["good_aft_rate"] = good_aft_rate

        df_res["score_max"] = df_res["score_bin"].apply(lambda x: x.right)
        df_res["total_cum_pct"] = df_res["total_pct"].cumsum()
        df_res["good_dist"] = df_res["total_good"] / (df_res["total_good"].sum() + eps)
        df_res["bad_dist"] = df_res["total_bad"] / (df_res["total_bad"].sum() + eps)

        # For isotonic calibrator, score_max is the probability itself
        df_res["exp_bad_rate"] = df_res["score_max"]

        df_res["iv"] = (
            (df_res["bad_dist"] - df_res["good_dist"])
            * np.log((df_res["bad_dist"] + eps) / (df_res["good_dist"] + eps))
        ).sum()
        df_res["cum_good_dist"] = df_res["good_dist"].cumsum()
        df_res["cum_bad_dist"] = df_res["bad_dist"].cumsum()
        df_res["ks"] = (df_res["cum_bad_dist"] - df_res["cum_good_dist"]).abs()

        # Odds ratio calculations
        df_res["odds_bef"] = df_res["total_bad"].cumsum() / (
            df_res["total_good"].cumsum() + eps
        )
        df_res["odds_aft"] = (df_res["total_bad"] - df_res["total_bad"].cumsum()) / (
            df_res["total_good"] - df_res["total_good"].cumsum() + eps
        )
        df_res["odds_ratio"] = df_res["odds_bef"] / df_res["odds_aft"]
        df_res["inv_odds_ratio"] = df_res["odds_aft"] / df_res["odds_bef"]

        lst_col = [
            "score_bin",
            "total",
            "total_pct",
            "total_cum_pct",
            "total_good",
            "good_rate",
            "good_dist",
            "good_bef_rate",
            "good_aft_rate",
            "total_bad",
            "bad_rate",
            "bad_dist",
            "bad_bef_rate",
            "bad_aft_rate",
            "approval_rate",
            "exp_bad_rate",
            "odds_ratio",
            "inv_odds_ratio",
            "ks",
            "iv",
        ]
        df_res = df_res[lst_col]
        return df_res

    def plot_score_vs_calibrated_prob(
        self,
        n_points: int = 100,
    ) -> plt.Figure:
        """
        Plot the linear scaling mapping from calibrated probability to scaled score.

        This visualization shows the linear scaling step that maps isotonic regression
        output (calibrated probability) to the final scaled score. The relationship
        is linear for standard scaling, or piecewise linear for segmented scaling.

        Args:
            n_points: Number of points to plot (default: 100).

        Returns:
            matplotlib.figure.Figure: The plot figure showing:
                - X-axis: Calibrated probability (after isotonic regression)
                - Y-axis: Scaled score (after linear scaling)

        Raises:
            ValueError: If calibrator has not been fitted.

        Example:
            >>> calibrator = IsotonicCalibrator()
            >>> calibrator.fit(y_prob_train, y_train)
            >>> fig = calibrator.plot_score_vs_calibrated_prob()
            >>> fig.savefig('linear_scaling_mapping.png')
        """
        if self.isotonic_regressor_ is None:
            raise ValueError("Calibrator must be fitted before plotting")

        # Generate range of calibrated probabilities (isotonic regression output)
        calibrated_probs = np.linspace(0, 1, n_points)

        # Apply linear scaling to get scaled scores
        if self.scale_threshold is None:
            # Standard scaling: apply to all
            scaled_scores = (
                calibrated_probs * self.mapping_slope + self.mapping_intercept
            )
        else:
            # Segmented scaling: only scale above threshold
            scaled_scores = calibrated_probs.copy()
            high_prob_mask = calibrated_probs >= self.scale_threshold
            scaled_scores[high_prob_mask] = (
                calibrated_probs[high_prob_mask] * self.mapping_slope
                + self.mapping_intercept
            )

        scaled_scores = np.clip(scaled_scores, self.score_floor, self.score_cap)

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the scaling mapping
        ax.plot(
            calibrated_probs,
            scaled_scores,
            "b-",
            linewidth=2,
            label="Linear Scaling Mapping",
        )

        # Plot diagonal for reference (identity mapping)
        ax.plot(
            [0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Identity (no scaling)"
        )

        # Add scale threshold line if applicable
        if self.scale_threshold is not None:
            ax.axvline(
                x=self.scale_threshold,
                color="g",
                linestyle="-.",
                alpha=0.5,
                label=f"Scale Threshold ({self.scale_threshold})",
            )
            ax.axhline(y=self.scale_threshold, color="g", linestyle="-.", alpha=0.5)

        # Add score floor and cap lines
        if self.score_cap < 1.0:
            ax.axhline(
                y=self.score_cap,
                color="r",
                linestyle=":",
                alpha=0.5,
                label=f"Score Cap ({self.score_cap})",
            )
        if self.score_floor > 0.0:
            ax.axhline(
                y=self.score_floor,
                color="r",
                linestyle=":",
                alpha=0.5,
                label=f"Score Floor ({self.score_floor})",
            )

        ax.set_xlabel(
            "Calibrated Probability (Isotonic Regression Output)", fontsize=12
        )
        ax.set_ylabel("Scaled Score (Final Output)", fontsize=12)
        ax.set_title(
            "Linear Scaling: Calibrated Probability → Scaled Score", fontsize=14
        )
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        # Set axis limits
        ax.set_xlim([0, 1])
        ax.set_ylim([max(0, self.score_floor - 0.05), min(1, self.score_cap + 0.05)])

        fig.tight_layout()
        plt.close(fig)

        return fig

    @classmethod
    def __check_type(cls, data):
        """Convert various input types to list."""
        if isinstance(data, (list, pd.Series, np.ndarray)):
            lst_data = list(data)
        elif isinstance(data, pd.DataFrame):
            lst_data = data[data.columns.item()].tolist()
        else:
            raise TypeError("Expected data type: DataFrame, List, Series or Array")
        return lst_data
