#!/usr/bin/env python
# Version: 0.4.0
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


class Calibration(TransformerMixin):
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
        >>> calibrator = Calibration()
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
        calibration_method: str = "polynomial",
        mapping_base: Optional[Dict[int, float]] = None,
        score_cap: Optional[float] = None,
        score_floor: Optional[float] = None,
        high_score_threshold: Optional[float] = None,
    ) -> None:
        """
        Initialize the Calibration transformer.

        Args:
            n_bins: Number of bins for calibration analysis and probability binning.
            n_degree: Polynomial degree for probability calibration curve fitting
                (only used when calibration_method='polynomial').
            calibration_method: Calibration method to use. Options:
                - 'polynomial': Polynomial fitting in log-odds space (default)
                - 'isotonic': Isotonic regression in probability space
            mapping_base: Custom score-to-probability mapping. If provided, transforms
                probabilities to credit scores. Format: {score: probability}, e.g., {600: 0.05}.
            score_cap: Maximum score value. If mapping_base provided, required. For isotonic
                auto-mapping, triggers auto-generation (default: 100).
            score_floor: Minimum score value (default: 0).
            high_score_threshold: High-risk threshold for auto-mapping (isotonic only, range: 0-1).
                Only applies when mapping_base=None and score_cap is provided.
                - None: uniform stretch to [score_floor, score_cap]
                - 0.7: high quantiles (P70-P100) stretched to [70, 100]

        Examples:
            >>> # Probability calibration with polynomial (default)
            >>> calibrator = Calibration()

            >>> # Isotonic calibration with score mapping
            >>> calibrator = Calibration(
            ...     calibration_method='isotonic',
            ...     mapping_base={500: 0.1, 600: 0.05, 700: 0.02},
            ...     score_cap=800,
            ...     score_floor=400
            ... )
        """
        self.mapping_base = mapping_base
        self.n_bins = n_bins
        self.n_degree = n_degree
        self.calibration_method = calibration_method
        self.high_score_threshold = high_score_threshold

        if calibration_method not in ["polynomial", "isotonic"]:
            raise ValueError(
                f"calibration_method must be 'polynomial' or 'isotonic', got '{calibration_method}'"
            )

        if mapping_base is not None:
            if score_cap is None or score_floor is None:
                raise ValueError(
                    "score_cap and score_floor are required when mapping_base is provided."
                )
            if score_floor >= score_cap:
                raise ValueError(
                    f"score_floor ({score_floor}) must be less than score_cap ({score_cap})."
                )

            self._use_score_mapping = True
            self._auto_mapping = False
            self.score_cap = score_cap
            self.score_floor = score_floor

        elif calibration_method == "isotonic" and (
            score_cap is not None or score_floor is not None
        ):
            if score_cap is None or score_floor is None:
                raise ValueError(
                    "For isotonic auto-mapping, both score_cap and score_floor must be provided together."
                )
            if score_floor >= score_cap:
                raise ValueError(
                    f"score_floor ({score_floor}) must be less than score_cap ({score_cap})."
                )

            self._use_score_mapping = True
            self._auto_mapping = True
            self.score_cap = score_cap
            self.score_floor = score_floor

        else:
            self._use_score_mapping = False
            self._auto_mapping = False
            self.score_cap = 1
            self.score_floor = 0

        if high_score_threshold is not None:
            if calibration_method != "isotonic":
                raise ValueError(
                    "high_score_threshold is only supported for calibration_method='isotonic'"
                )
            if not (0 < high_score_threshold < 1):
                raise ValueError(
                    f"high_score_threshold must be between 0 and 1, got {high_score_threshold}"
                )
            if not self._auto_mapping:
                logging.warning(
                    "high_score_threshold is specified but auto-mapping is not enabled. "
                    "It will be ignored unless score_cap is provided without mapping_base."
                )

        self.calibrate_detail: Optional[pd.DataFrame] = None
        self.calibrate_coef: Optional[np.ndarray] = None
        self.iso_model: Optional[Any] = None
        self.mapping_intercept: Optional[float] = None
        self.mapping_slope: Optional[float] = None
        self.calibrate_plot: Optional[plt.Figure] = None

    def fit(
        self,
        df_prob: Union[pd.Series, np.ndarray],
        df_label: Union[pd.Series, np.ndarray],
    ) -> "Calibration":
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
            >>> calibrator = Calibration()
            >>> calibrator.fit(y_prob_train, y_true_train)
            >>> print("Calibration fitted successfully")
        """
        lst_prob = self.__check_type(df_prob)
        lst_label = self.__check_type(df_label)

        # Auto-generate mapping for isotonic if needed
        if self._auto_mapping and self.calibration_method == "isotonic":
            self.mapping_base = self._generate_auto_mapping(
                lst_prob,
                self.score_cap,
                self.score_floor,
                self.high_score_threshold,
                self.n_bins,
            )
            logging.info(
                f"Auto-generated mapping for isotonic: "
                f"{'uniform' if self.high_score_threshold is None else 'high-risk stretched'}"
            )

        # Set mapping parameters if in score mapping mode
        if self.mapping_base is not None:
            self._use_score_mapping = True
            logging.info(
                "Score mapping mode: using mapping_base, score_cap, and score_floor"
            )
            self.mapping_slope, self.mapping_intercept = self._set_mapping_base(
                self.mapping_base, self.calibration_method
            )
        else:
            self._use_score_mapping = False
            logging.info("Probability mode: output calibrated probabilities directly")

        if self.calibration_method == "polynomial":
            # Polynomial calibration: binning in log-odds space
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

            # Generate calibrated probabilities for plot
            lst_lnodds_prob = [self.prob2lnodds(x) for x in lst_prob]
            lst_lnodds_cal_prob = [
                np.poly1d(self.calibrate_coef)(x) for x in lst_lnodds_prob
            ]
            y_prob_after = np.array([self.lnodds2prob(x) for x in lst_lnodds_cal_prob])

        elif self.calibration_method == "isotonic":
            # Isotonic calibration: binning in probability space
            df_data = pd.DataFrame(
                {
                    "yprob": lst_prob,
                    "label": lst_label,
                }
            )
            df_data["prob_bin"] = pd.qcut(
                df_data["yprob"], self.n_bins, duplicates="drop"
            )

            df_cal = df_data.groupby("prob_bin").agg(
                total=("label", "count"),
                bad_rate=("label", "mean"),
                prob_mean_x=("yprob", "mean"),
            )
            df_cal["adj_bad_rate"] = df_cal.apply(
                lambda x: max(x["bad_rate"], 1 / x["total"], 0.0001), axis=1
            )

            lst_col = [
                "total",
                "bad_rate",
                "adj_bad_rate",
                "prob_mean_x",
            ]
            self.calibrate_detail = df_cal[lst_col]

            # Fit isotonic regression
            self.iso_model = IsotonicRegression(
                y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip"
            )
            self.iso_model.fit(
                df_cal["prob_mean_x"].values, df_cal["adj_bad_rate"].values
            )

            # Generate calibrated probabilities for plot
            y_prob_after = self.iso_model.predict(np.array(lst_prob))

        # Generate and store calibration plot (always generate regardless of mode)
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

        if self.calibration_method == "polynomial":
            # Polynomial calibration: log-odds transformation
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

        elif self.calibration_method == "isotonic":
            # Isotonic calibration: direct probability prediction
            cal_prob = self.iso_model.predict(np.array(lst_prob))

            if self._use_score_mapping:
                # Score mapping mode: linear mapping from calibrated probability to score
                # Need to convert probability to score using mapping_base
                lst_score = [
                    self.mapping_intercept + self.mapping_slope * p for p in cal_prob
                ]
                lst_score = [max(x, self.score_floor) for x in lst_score]
                lst_score = [min(x, self.score_cap) for x in lst_score]
                return np.array(lst_score)
            else:
                # Probability mode: return calibrated probabilities
                return cal_prob

    def compare_calibrate_result(self, df_score, df_label, bins=None):
        """
        Compare calibration results by generating a detailed scorecard analysis.

        Analyzes score distribution, bad rates, lift metrics, and KS/IV statistics
        across score bins to evaluate calibration quality.

        Args:
            df_score: Calibrated scores from transform().
            df_label: True binary labels (0 or 1).
            bins: Score bin boundaries. Required parameter.
                - For probability mode: use bins like [0, 0.1, 0.2, ..., 1.0]
                - For score mapping mode: use bins like [300, 400, 500, ..., 1000]

        Returns:
            pd.DataFrame: Scorecard with columns including:
                - score_bin: Score range intervals
                - total, total_pct: Sample counts and percentages
                - bad_rate, good_rate: Observed rates per bin
                - approval_rate, bad_aft_rate: Cumulative rates
                - odds_ratio, inv_odds_ratio: Odds comparison metrics
                - ks, iv: KS statistic and Information Value

        Raises:
            ValueError: If bins not provided.

        Example:
            >>> scorecard = calibrator.compare_calibrate_result(
            ...     scores, y_test, bins=[0, 500, 600, 700, 800, 1000]
            ... )
        """
        if bins is None:
            raise ValueError("bins is required for compare_calibrate_result()")

        lst_score = self.__check_type(df_score)
        lst_label = self.__check_type(df_label)
        df_data = pd.DataFrame({"score": lst_score, "label": lst_label})
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
        fig, ax = plt.subplots(figsize=(8, 6))

        CalibrationDisplay.from_predictions(
            y_true,
            y_prob_before,
            n_bins=n_bins,
            strategy=strategy,
            ax=ax,
            name="Before Calibration",
            color="blue",
        )

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
        return self.calibrate_plot

    def get_lnodds_calibrate_plot(self) -> plt.Axes:
        f, ax = plt.subplots(figsize=(8, 6))

        if self.calibration_method == "polynomial":
            x = self.calibrate_detail["lnodds_prob_mean_x"]
            y_actual = self.calibrate_detail["lnodds_bad_rate_y"]
            y_pred = np.poly1d(self.calibrate_coef)(x)

            ax.plot(x, y_actual, "o", label="Observed", markersize=6)
            ax.plot(x, y_pred, "-", label="Polynomial Fit", linewidth=2)
            ax.set_xlabel("Log-Odds(Predicted Prob)", fontsize=11)
            ax.set_ylabel("Log-Odds(Observed Rate)", fontsize=11)
            ax.set_title(
                "Polynomial Calibration (Log-Odds Space)",
                fontsize=12,
                fontweight="bold",
            )

        elif self.calibration_method == "isotonic":
            x = self.calibrate_detail["prob_mean_x"]
            y_actual = self.calibrate_detail["adj_bad_rate"]
            y_pred = self.iso_model.predict(x.values)

            ax.plot(x, y_actual, "o", label="Observed", markersize=6)
            ax.plot(x, y_pred, "-", label="Isotonic Fit", linewidth=2)
            ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect Calibration")
            ax.set_xlabel("Mean Predicted Probability", fontsize=11)
            ax.set_ylabel("Observed Rate", fontsize=11)
            ax.set_title(
                "Isotonic Calibration (Probability Space)",
                fontsize=12,
                fontweight="bold",
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        return ax

    @classmethod
    def _set_mapping_base(
        cls, dict_base: Dict[int, float], calibration_method: str
    ) -> Tuple[float, float]:
        items = sorted(dict_base.items(), key=lambda x: x[0])
        lst_score = [item[0] for item in items]
        lst_bad_rate = [item[1] for item in items]

        score_max, score_min = lst_score[-1], lst_score[0]

        if calibration_method == "polynomial":
            lst_lnodds_bad_rate = [cls.prob2lnodds(x) for x in lst_bad_rate]
            lnodds_max, lnodds_min = lst_lnodds_bad_rate[-1], lst_lnodds_bad_rate[0]
            slope = (score_max - score_min) / (lnodds_max - lnodds_min)
            intercept = score_max - slope * lnodds_max

        elif calibration_method == "isotonic":
            prob_max, prob_min = lst_bad_rate[-1], lst_bad_rate[0]
            slope = (score_max - score_min) / (prob_max - prob_min)
            intercept = score_max - slope * prob_max

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

    @classmethod
    def _generate_auto_mapping(
        cls,
        prob_data: List[float],
        score_cap: float,
        score_floor: float,
        high_score_threshold: Optional[float],
        n_bins: int,
    ) -> Dict[int, float]:
        prob_array = np.array(prob_data)
        mapping = {}

        if high_score_threshold is None:
            percentiles = np.linspace(0, 1.0, n_bins + 1)
            prob_quantiles = np.quantile(prob_array, percentiles)

            for i, prob in enumerate(prob_quantiles):
                score = score_floor + (score_cap - score_floor) * (i / n_bins)
                mapping[int(score)] = float(prob)

        else:
            threshold_score = (
                score_floor + (score_cap - score_floor) * high_score_threshold
            )

            n_low_bins = int(n_bins * high_score_threshold)
            low_percentiles = np.linspace(0, high_score_threshold, n_low_bins + 1)
            low_prob_quantiles = np.quantile(prob_array, low_percentiles)

            for i, prob in enumerate(low_prob_quantiles):
                score = score_floor + (threshold_score - score_floor) * (i / n_low_bins)
                mapping[int(score)] = float(prob)

            n_high_bins = n_bins - n_low_bins
            high_percentiles = np.linspace(high_score_threshold, 1.0, n_high_bins + 1)
            high_prob_quantiles = np.quantile(prob_array, high_percentiles)

            for i, prob in enumerate(high_prob_quantiles):
                if i == 0:
                    continue
                score = threshold_score + (score_cap - threshold_score) * (
                    i / n_high_bins
                )
                mapping[int(score)] = float(prob)

        return mapping
