#!/usr/bin/env python
# Version: 0.3.0
# Created: 2024-04-07
# Author: ["Hanyuan Zhang"]

import io
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
from openpyxl.drawing.image import Image as XLImage
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet
from pandas.api.types import is_numeric_dtype

from .calibrator import Calibration
from .metrics import Metrics


class Reporter:
    """
    Comprehensive report generator for machine learning model analysis and validation.

    The Reporter class creates detailed Excel reports containing various aspects of model
    performance, feature analysis, calibration metrics, and data quality assessments.
    It provides formatted Excel output with proper styling for professional reporting.

    Key Features:
    - **Sample Overview**: Statistical summaries and performance metrics across samples
    - **Feature Analysis**: Detailed feature statistics, WOE binning, and quality metrics
    - **Feature Selection**: Documentation of feature selection pipeline and criteria
    - **Model Performance**: Benchmark comparisons and hyperparameter tuning results
    - **Calibration Analysis**: Score distribution, PSI tracking, and scorecard generation
    - **Professional Formatting**: Automatic Excel formatting with proper column widths and number formats

    Attributes:
        report_path: Directory path where generated reports will be saved.
                    If None, reports will be saved in the current working directory.

    Example:
        >>> # Initialize reporter with custom path
        >>> reporter = Reporter(report_path='/path/to/reports')
        >>>
        >>> # Generate comprehensive model report
        >>> performance_data = {
        ...     'model_id': 'risk_model_v1',
        ...     'sample_set': {'train': train_data, 'test': test_data},
        ...     'label': 'default_flag',
        ...     'woe_df': woe_results,
        ...     'feature_selection': selection_pipeline,
        ...     'calibrate_detail': calibration_metrics
        ... }
        >>> reporter.generate_report(performance_data)
    """

    def __init__(self, report_file_name: Optional[str] = None) -> None:
        """
        Initialize the Reporter with optional custom report directory.

        Args:
            report_file_name: File name for the generated report. If None, a default name will be used.
        """
        self.report_file_name = report_file_name or "model_report.xlsx"

    @classmethod
    def _set_col_format(
        cls, worksheet: Worksheet, col_ids: List[str], format: str
    ) -> None:
        """
        Apply number formatting to specified columns in an Excel worksheet.

        Args:
            worksheet: The openpyxl worksheet object to format.
            col_ids: List of column letters (e.g., ['A', 'B', 'C']) to format.
            format: Excel number format string (e.g., '0.000%' for percentages).
        """
        for col_idx in col_ids:
            for cell in worksheet[col_idx]:
                cell.number_format = format

    @classmethod
    def _set_col_width(
        cls, worksheet: Worksheet, col_ids: List[str], width: Union[int, float]
    ) -> None:
        """
        Set column widths for specified columns in an Excel worksheet.

        Args:
            worksheet: The openpyxl worksheet object to modify.
            col_ids: List of column letters (e.g., ['A', 'B', 'C']) to resize.
            width: Column width value in Excel units.
        """
        for col_idx in col_ids:
            worksheet.column_dimensions[col_idx].width = width

    def _format_overview_stats_df(self, worksheet: Worksheet) -> None:
        """
        Format the sample overview statistics worksheet with appropriate styling.

        Applies percentage formatting to statistical columns and sets optimal column widths
        for readability of sample statistics including counts, percentages, and PSI values.

        Args:
            worksheet: The openpyxl worksheet containing sample overview statistics.
        """
        perc_cols = ["C", "D", "E"]
        Reporter._set_col_format(worksheet, perc_cols, "0.000%")
        w20 = ["A", "B", "C", "D", "E"]
        Reporter._set_col_width(worksheet, w20, 20)

    def _format_overview_perf_df(self, worksheet: Worksheet) -> None:
        """
        Format the sample overview performance worksheet with appropriate styling.

        Applies percentage formatting to performance metrics and sets optimal column widths
        for model performance indicators including AUC, KS, Gini, and IV values.

        Args:
            worksheet: The openpyxl worksheet containing sample performance metrics.
        """
        perc_cols = ["B", "C", "D", "E", "F", "G"]
        Reporter._set_col_format(worksheet, perc_cols, "0.000%")
        w20 = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
        Reporter._set_col_width(worksheet, w20, 20)

    def _format_feat_df(self, worksheet: Worksheet) -> None:
        """
        Format the feature analysis worksheet with comprehensive styling.

        Applies percentage formatting to feature quality metrics and sets varied column widths
        optimized for feature names, statistics, and analytical metrics display.

        Args:
            worksheet: The openpyxl worksheet containing feature analysis data.
        """
        perc_cols = ["D", "F", "G", "H", "S", "U", "W", "X", "Y"]
        Reporter._set_col_format(worksheet, perc_cols, "0.000%")
        w40 = ["A"]
        w25 = ["G", "H", "I", "J"]
        w20 = [
            "B",
            "C",
            "D",
            "E",
            "F",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
            "W",
        ]
        w15 = ["X", "Y"]
        Reporter._set_col_width(worksheet, w40, 40)
        Reporter._set_col_width(worksheet, w25, 25)
        Reporter._set_col_width(worksheet, w20, 20)
        Reporter._set_col_width(worksheet, w15, 10)

    def _format_woe_df(self, worksheet: Worksheet) -> None:
        """
        Format the Weight of Evidence (WOE) binning worksheet with appropriate styling.

        Applies decimal and percentage formatting to WOE metrics and sets optimal column
        widths for feature names, bin descriptions, and statistical measures.

        Args:
            worksheet: The openpyxl worksheet containing WOE binning analysis.
        """
        num_cols_4dec = ["N", "O", "Q"]
        perc_cols = ["D", "E", "G", "H", "I", "K", "L", "M", "P", "S", "T"]
        Reporter._set_col_format(worksheet, num_cols_4dec, "0.0000")
        Reporter._set_col_format(worksheet, perc_cols, "0.000%")
        w40 = ["A"]
        w20 = ["B"]
        w15 = [get_column_letter(i) for i in range(3, worksheet.max_column + 1)]
        Reporter._set_col_width(worksheet, w40, 40)
        Reporter._set_col_width(worksheet, w20, 20)
        Reporter._set_col_width(worksheet, w15, 15)

    def _format_feature_selection_overview_df(self, worksheet: Worksheet) -> None:
        """
        Format the feature selection overview worksheet with appropriate styling.

        Sets optimal column width for feature names to accommodate long feature identifiers
        in the feature selection summary table.

        Args:
            worksheet: The openpyxl worksheet containing feature selection overview.
        """
        w40 = ["A"]
        Reporter._set_col_width(worksheet, w40, 40)

    def _format_feature_selection_df(self, worksheet: Worksheet) -> None:
        """
        Format individual feature selection method worksheets with appropriate styling.

        Sets varied column widths optimized for feature selection criteria display,
        including feature names and selection metrics.

        Args:
            worksheet: The openpyxl worksheet containing specific feature selection results.
        """
        w20 = ["C", "D", "E"]
        w40 = ["B"]
        Reporter._set_col_width(worksheet, w20, 20)
        Reporter._set_col_width(worksheet, w40, 40)

    def _format_benchmark_df(self, worksheet: Worksheet) -> None:
        """
        Format the benchmark model results worksheet with appropriate styling.

        Sets uniform column widths for benchmark model comparison metrics and parameters.

        Args:
            worksheet: The openpyxl worksheet containing benchmark model details.
        """
        w20 = ["A", "B", "C", "D", "E", "F", "G", "H"]
        Reporter._set_col_width(worksheet, w20, 20)

    def _format_tuning_df(self, worksheet: Worksheet) -> None:
        """
        Format the hyperparameter tuning results worksheet with comprehensive styling.

        Applies percentage formatting to performance metrics and sets varied column widths
        optimized for hyperparameter names, values, and performance indicators.

        Args:
            worksheet: The openpyxl worksheet containing hyperparameter tuning results.
        """
        perc_cols = ["P", "Q", "R", "S", "T", "U", "V", "W", "X"]
        Reporter._set_col_format(worksheet, perc_cols, "0.000%")
        w15 = [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "T",
            "U",
            "V",
            "W",
            "X",
        ]
        w20 = ["P", "Q", "R", "S"]
        w40 = ["L", "M", "N", "O"]
        Reporter._set_col_width(worksheet, w15, 15)
        Reporter._set_col_width(worksheet, w20, 20)
        Reporter._set_col_width(worksheet, w40, 40)

    def _format_calibration_reg_df(self, worksheet: Worksheet) -> None:
        """
        Format the calibration regression results worksheet with appropriate styling.

        Sets varied column widths optimized for regression parameters and calibration metrics.

        Args:
            worksheet: The openpyxl worksheet containing calibration regression details.
        """
        w15 = ["A", "B", "C", "D"]
        w20 = ["E", "F"]
        Reporter._set_col_width(worksheet, w15, 15)
        Reporter._set_col_width(worksheet, w20, 20)

    def _format_calibration_scorecard_df(self, worksheet: Worksheet) -> None:
        """
        Format the calibration scorecard worksheet with comprehensive styling.

        Applies percentage formatting to scorecard metrics and sets uniform column widths
        for scorecard elements including score bands, population distributions, and rates.

        Args:
            worksheet: The openpyxl worksheet containing calibration scorecard data.
        """
        perc_cols = [
            "D",
            "E",
            "G",
            "H",
            "I",
            "J",
            "L",
            "M",
            "N",
            "O",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
        ]
        Reporter._set_col_format(worksheet, perc_cols, "0.000%")
        w15 = [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
        ]
        Reporter._set_col_width(worksheet, w15, 15)

    def _format_calibration_scoredist_df(self, worksheet: Worksheet) -> None:
        """
        Format the calibration score distribution worksheet with appropriate styling.

        Applies percentage formatting to distribution metrics and sets optimal column widths
        for score distribution analysis across different sample types.

        Args:
            worksheet: The openpyxl worksheet containing score distribution analysis.
        """
        perc_cols = ["C", "E", "G", "I", "K", "M", "O", "Q", "S", "U"]
        Reporter._set_col_format(worksheet, perc_cols, "0.000%")
        w15 = ["C", "E", "G", "I", "K", "M", "O", "Q", "S", "U"]
        Reporter._set_col_width(worksheet, w15, 15)

    def _format_calibration_score_psi_df(self, worksheet: Worksheet) -> None:
        """
        Format the calibration score PSI worksheet with appropriate styling.

        Applies percentage formatting to PSI metrics and sets optimal column widths
        for Population Stability Index tracking and analysis.

        Args:
            worksheet: The openpyxl worksheet containing score PSI analysis.
        """
        perc_cols = ["B"]
        Reporter._set_col_format(worksheet, perc_cols, "0.00%")
        w15 = ["B"]
        Reporter._set_col_width(worksheet, w15, 15)
        w20 = ["A"]
        Reporter._set_col_width(worksheet, w20, 20)

    def _insert_figure_to_excel(
        self,
        writer: pd.ExcelWriter,
        fig: plt.Figure,
        sheet_name: str,
        anchor: str = "A3",
        title: str = "",
        dpi: int = 50,
    ) -> None:
        """
        Insert a matplotlib figure into an Excel worksheet.

        Creates a new worksheet and embeds the figure as a PNG image.

        Args:
            writer: Excel writer object for output file generation.
            fig: Matplotlib Figure object to insert.
            sheet_name: Name for the new worksheet.
            anchor: Cell anchor position for the image (default: 'A3').
            title: Optional title to display in the sheet header.
        """
        # Create a placeholder DataFrame for the sheet
        header_text = title if title else sheet_name
        pd.DataFrame({sheet_name: [header_text]}).to_excel(
            writer, sheet_name=sheet_name, index=False
        )

        worksheet = writer.sheets[sheet_name]

        # Save figure to byte buffer and add to worksheet
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        buf.seek(0)
        img = XLImage(buf)
        img.anchor = anchor
        worksheet.add_image(img)

    def _stat_feat(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        woe_df: pd.DataFrame,
        missing_values: List[Any],
    ) -> pd.DataFrame:
        """
        Generate comprehensive feature statistics and quality metrics.

        Analyzes each feature in the dataset to provide detailed statistics including
        missing value rates, bad rates, descriptive statistics for numerical features,
        mode analysis for categorical features, and WOE-derived metrics.

        Statistical Metrics Calculated:
        - **Missing Value Analysis**: Count and percentage of missing values
        - **Bad Rate Analysis**: Target rate with and without missing values
        - **Uniqueness Metrics**: Unique value counts with/without missing values
        - **Numerical Statistics**: Min, max, mean, percentiles, standard deviation, CV
        - **Zero Analysis**: Count and percentage of zero values (numerical features)
        - **Mode Analysis**: Top 3 most frequent values and their rates
        - **WOE Metrics**: KS and IV values from Weight of Evidence analysis

        Args:
            X: Feature dataset with all input variables for analysis.
            y: Target variable (binary) for bad rate calculations.
            woe_df: Weight of Evidence DataFrame containing KS and IV metrics by feature.
            missing_values: List of values to be treated as missing (e.g., [None, np.nan, -999]).

        Returns:
            Comprehensive feature statistics DataFrame with the following columns:
                - Basic counts: total, #missing, %missing, #zero, %zero
                - Bad rates: with/without missing values
                - Uniqueness: unique counts with/without missing values
                - Descriptive stats: min, 25%, mean, 75%, max, std, cva
                - Mode analysis: mode, mode_rate, second_mode, second_mode_rate, third_mode, third_mode_rate
                - WOE metrics: KS, IV values from binning analysis

        Example:
            >>> # Analyze feature quality
            >>> missing_vals = [None, np.nan, -999, '']
            >>> feature_stats = reporter._stat_feat(X_train, y_train, woe_results, missing_vals)
            >>> print(feature_stats[['%missing', 'Bad Rate (without missing)', 'KS', 'IV']])
        """
        feature_summaries = []
        for column in X.columns:
            total = len(X)
            missing = X[column].apply(lambda x: x in missing_values).sum()
            unique = X[column].nunique()
            unique_without_missing = X.loc[
                X[column].apply(lambda x: x not in missing_values), column
            ].nunique()
            bad_rate_with_missing = y.mean()
            bad_rate_without_missing = y[
                X[column].apply(lambda x: x not in missing_values)
            ].mean()

            X_normal = X.loc[X[column].apply(lambda x: x not in missing_values)]
            mode = (
                X_normal[column].mode().iloc[0]
                if len(X_normal[column].mode()) > 0
                else None
            )
            mode_rate = (X_normal[column] == mode).mean()
            second_mode = (
                X_normal[column].value_counts().index[1]
                if len(X_normal[column].value_counts()) > 1
                else None
            )
            second_mode_rate = (
                (X_normal[column] == second_mode).mean() if second_mode else None
            )
            third_mode = (
                X_normal[column].value_counts().index[2]
                if len(X_normal[column].value_counts()) > 2
                else None
            )
            third_mode_rate = (
                (X_normal[column] == third_mode).mean() if third_mode else None
            )

            # for numerical features
            is_numerical = is_numeric_dtype(X_normal[column])
            desc_stats = X_normal[column].describe() if is_numerical else None
            stats_min = desc_stats["min"] if is_numerical else None
            stats_25 = desc_stats["25%"] if is_numerical else None
            stats_mean = desc_stats["mean"] if is_numerical else None
            stats_75 = desc_stats["75%"] if is_numerical else None
            stats_max = desc_stats["max"] if is_numerical else None
            stats_std = desc_stats["std"] if is_numerical else None
            zero = (X_normal[column] == 0).sum() if is_numerical else None
            zero_rate = zero / total if is_numerical else None
            cva = stats_std / stats_mean if is_numerical else None

            feature_summary = {
                "Feature": column,
                "total": total,
                "#missing": missing,
                "%missing": missing / total,
                "#zero": zero,
                "%zero": zero_rate,
                "Bad Rate (with missing)": bad_rate_with_missing,
                "Bad Rate (without missing)": bad_rate_without_missing,
                "Unique (with missing)": unique,
                "Unique (without missing)": unique_without_missing,
                "min": stats_min,
                "25%": stats_25,
                "mean": stats_mean,
                "75%": stats_75,
                "max": stats_max,
                "std": stats_std,
                "cva": cva,
                "mode": mode,
                "mode_rate": mode_rate,
                "second_mode": second_mode,
                "second_mode_rate": second_mode_rate,
                "third_mode": third_mode,
                "third_mode_rate": third_mode_rate,
            }

            feature_summaries.append(feature_summary)

        summary_df = pd.DataFrame(feature_summaries).set_index("Feature")
        return pd.concat(
            [
                summary_df,
                woe_df[["KS", "IV"]]
                .reset_index()
                .groupby("feature_name")
                .agg(KS=(("KS", ""), "max"), IV=(("IV", "bin"), "sum")),
            ],
            axis=1,
        )

    def _stat_perf(self, gp: pd.DataFrame, target_label: str) -> pd.DataFrame:
        """
        Calculate comprehensive performance statistics for model predictions.

        Computes key performance metrics including AUC, KS, Gini, and IV for both
        benchmark and main model predictions. Handles edge cases where target
        variable has only one unique value.

        Performance Metrics:
        - **AUC (Area Under Curve)**: Discrimination ability measure (0.5-1.0)
        - **KS (Kolmogorov-Smirnov)**: Maximum separation between distributions
        - **Gini Coefficient**: Concentration measure derived from AUC
        - **IV (Information Value)**: Predictive power indicator

        Args:
            gp: DataFrame containing predictions and actual target values with columns:
                - target_label: Actual binary target variable
                - 'bm_proba': Benchmark model probability predictions
                - 'proba': Main model probability predictions
            target_label: Column name of the target variable in the DataFrame.

        Returns:
            Performance metrics DataFrame with columns:
                - BM_AUC, BM_KS: Benchmark model performance metrics
                - AUC, KS, Gini, IV: Main model performance metrics
                Returns None values if target has only one unique value.

        Example:
            >>> # Calculate performance for validation set
            >>> val_predictions = pd.DataFrame({
            ...     'default_flag': [0, 1, 0, 1, 0],
            ...     'bm_proba': [0.1, 0.8, 0.2, 0.9, 0.15],
            ...     'proba': [0.05, 0.85, 0.25, 0.95, 0.1]
            ... })
            >>> perf_stats = reporter._stat_perf(val_predictions, 'default_flag')
        """
        y_true = gp[target_label]
        y_bm_proba = gp["bm_proba"]
        y_proba = gp["proba"]
        if y_true.nunique() == 1:
            return pd.DataFrame(
                {
                    "BM_AUC": [None],
                    "AUC": [None],
                    "BM_KS": [None],
                    "KS": [None],
                    "Gini": [None],
                    "IV": [None],
                }
            )
        return pd.DataFrame(
            {
                "BM_AUC": [Metrics.get_auc(y_true, y_bm_proba)],
                "AUC": [Metrics.get_auc(y_true, y_proba)],
                "BM_KS": [Metrics.get_ks(y_true, y_bm_proba)],
                "KS": [Metrics.get_ks(y_true, y_proba)],
                "Gini": [Metrics.get_gini(y_true, y_proba)],
                "IV": [Metrics.get_iv(y_true, y_proba)],
            }
        )

    def _generate_sample_overview_basic(
        self, writer: pd.ExcelWriter, sample_set: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        Generate basic sample overview (no score/performance required).

        Creates a simple worksheet with sample sizes and bad rates.
        Used when model scores are not yet available (e.g., preprocessor stage).

        Args:
            writer: Excel writer object for output file generation.
            sample_set: Dictionary containing sample data with 'X' and 'y' keys.
        """
        sample_stats = []
        for sample_type, sample in sample_set.items():
            y = sample["y"]
            sample_stats.append(
                {
                    "sample_type": sample_type,
                    "# Sample": len(y),
                    "% Bad": y.mean() if hasattr(y, "mean") else sum(y) / len(y),
                }
            )
        df_basic = pd.DataFrame(sample_stats).set_index("sample_type")
        df_basic["% Sample"] = df_basic["# Sample"] / df_basic["# Sample"].sum()
        df_basic = df_basic[["# Sample", "% Sample", "% Bad"]]
        df_basic.sort_index(ascending=False).to_excel(
            writer, sheet_name="Sample Overview", freeze_panes=(1, 1)
        )

    def generate_sample_overview_report(
        self, writer: pd.ExcelWriter, sample_set: Dict[str, Dict[str, Any]], label: str
    ) -> None:
        """
        Generate comprehensive sample overview report with statistics and performance metrics.

        Creates two worksheets containing sample-level analysis:
        1. **Sample Overview - Statistics**: Sample sizes, bad rates, and PSI values
        2. **Sample Overview - Performance**: Model performance metrics by sample

        Args:
            writer: Excel writer object for output file generation.
            sample_set: Dictionary containing sample data with structure:
                {'train': {'e': extended_df, 'X': features, 'y': target},
                 'test': {'e': extended_df, 'X': features, 'y': target}, ...}
                The 'e' DataFrame must contain 'sample_type' and 'score' columns.
            label: Target variable column name for bad rate calculations.

        Raises:
            KeyError: If required columns ('sample_type', 'score') are missing from sample data.
            ValueError: If sample_set structure is invalid or empty.

        Example:
            >>> # Generate sample overview
            >>> sample_data = {
            ...     'train': {'e': train_extended, 'X': X_train, 'y': y_train},
            ...     'test': {'e': test_extended, 'X': X_test, 'y': y_test}
            ... }
            >>> with pd.ExcelWriter('report.xlsx') as writer:
            ...     reporter.generate_sample_overview_report(writer, sample_data, 'default_flag')
        """
        df_extra = pd.concat([sample["e"] for _, sample in sample_set.items()])
        df_basic_summary = df_extra.groupby("sample_type").agg(
            sample_size=("sample_type", "size"),
            sample_size_pct=("sample_type", lambda x: x.size / df_extra.shape[0]),
            positive_rate=(label, "mean"),
        )
        df_basic_summary.columns = ["# Sample", "% Sample", "% Bad"]
        df_basic_summary.sort_index(ascending=False).to_excel(
            writer, sheet_name="Sample Overview - Statistics", freeze_panes=(1, 1)
        )
        self._format_overview_stats_df(writer.sheets["Sample Overview - Statistics"])

        df_extra.groupby("sample_type").apply(
            self._stat_perf, target_label=label
        ).droplevel(level=1).sort_index(ascending=False).to_excel(
            writer, sheet_name="Sample Overview - Performance", freeze_panes=(1, 1)
        )
        self._format_overview_perf_df(writer.sheets["Sample Overview - Performance"])

    def generate_single_feature_eda_report(
        self,
        writer: pd.ExcelWriter,
        X: pd.DataFrame,
        y: pd.Series,
        woe_df: pd.DataFrame,
        missing_values: List[Any],
        prefix: str = "",
    ) -> None:
        """
        Generate comprehensive Exploratory Data Analysis (EDA) report for features.

        Creates detailed feature analysis worksheets containing:
        1. **Feature Overview**: Comprehensive feature statistics and quality metrics
        2. **Feature Binning Report**: Weight of Evidence binning analysis with KS/IV

        Args:
            writer: Excel writer object for output file generation.
            X: Feature dataset for analysis.
            y: Target variable for bad rate and correlation analysis.
            woe_df: Weight of Evidence DataFrame containing binning results with KS and IV metrics.
            missing_values: List of values to treat as missing (e.g., [None, np.nan, -999]).
            prefix: Optional prefix for worksheet names (e.g., "Train", "Test").

        Raises:
            AssertionError: If number of features in X doesn't match unique features in woe_df.
            ValueError: If woe_df doesn't contain required columns or structure.

        Example:
            >>> # Generate feature EDA report
            >>> missing_vals = [None, np.nan, -999]
            >>> with pd.ExcelWriter('feature_analysis.xlsx') as writer:
            ...     reporter.generate_single_feature_eda_report(
            ...         writer, X_train, y_train, woe_results, missing_vals, "Train"
            ...     )
        """
        feature_names = woe_df.reset_index()["feature_name"].unique().tolist()
        assert X.shape[1] == len(
            feature_names
        ), f"The number of feature dataframe columns ({X.shape[1]}) is not match the woe_df ({len(feature_names)})!"
        feat_df = self._stat_feat(X, y, woe_df, missing_values)
        feat_df.to_excel(
            writer, sheet_name=f"{prefix} - Feature Overview", freeze_panes=(1, 1)
        )
        woe_df.to_excel(
            writer, sheet_name=f"{prefix} - Feature Binning Report", freeze_panes=(3, 2)
        )
        self._format_feat_df(writer.sheets[f"{prefix} - Feature Overview"])
        self._format_woe_df(writer.sheets[f"{prefix} - Feature Binning Report"])

    def generate_feature_selection_report(
        self, writer: pd.ExcelWriter, fitted_steps: Any
    ) -> None:
        """
        Generate comprehensive feature selection pipeline report.

        Creates detailed analysis of the feature selection process including:
        1. **Feature Selection - Overview**: Summary showing which features survived each selection step
        2. **Feature Selection - [Method]**: Detailed results for each selection method

        Args:
            writer: Excel writer object for output file generation.
            fitted_steps: Fitted steps from Preprocess. Can be either:
                - Dict[str, transformer]: From preprocessor.named_steps
                - List[Tuple[str, transformer]]: From preprocessor._fitted_steps
                Each selector should have 'selected_features' and 'detail' attributes.

        Selection Methods Supported:
        - **Correlation (Corr)**: Uses selector.detail['after'] for results
        - **Other Methods**: Uses selector.detail directly for results

        Example:
            >>> # Generate feature selection report
            >>> with pd.ExcelWriter('feature_selection.xlsx') as writer:
            ...     reporter.generate_feature_selection_report(writer, preprocessor.named_steps)
        """
        # Normalize input to list of tuples
        if isinstance(fitted_steps, dict):
            steps_list = list(fitted_steps.items())
        else:
            steps_list = list(fitted_steps)

        # Filter to only selector steps (by checking module and attribute)
        selector_steps = [
            (name, step)
            for name, step in steps_list
            if "feature_selection" in getattr(step.__class__, "__module__", "")
            and hasattr(step, "selected_features")
        ]

        if not selector_steps:
            return

        # Find "Original" step for baseline features
        original_features = None
        for name, selector in selector_steps:
            if name == "Original":
                original_features = selector.selected_features
                break

        # Fallback: use first selector's features as baseline
        if original_features is None:
            original_features = selector_steps[0][1].selected_features

        # Handle empty features case
        if original_features is None or len(original_features) == 0:
            # Create an empty overview sheet with a message
            pd.DataFrame(
                {"Message": ["No features available for selection overview"]}
            ).to_excel(writer, sheet_name="Feature Selection - Overview", index=False)
            return

        # Build overview DataFrame
        original_cols = pd.DataFrame(original_features, columns=["feature"]).set_index(
            "feature"
        )

        for name, selector in selector_steps:
            if name == "Original":
                continue
            original_cols[name] = 0
            if (
                selector.selected_features is not None
                and len(selector.selected_features) > 0
            ):
                # Mark selected features as 1
                for feat in selector.selected_features:
                    if feat in original_cols.index:
                        original_cols.loc[feat, name] = 1

        # Sort by selection order (last selector first)
        sort_columns = [name for name, _ in selector_steps if name != "Original"][::-1]
        if sort_columns:
            original_cols.sort_values(sort_columns, ascending=False).to_excel(
                writer, sheet_name="Feature Selection - Overview"
            )
        else:
            original_cols.to_excel(writer, sheet_name="Feature Selection - Overview")

        self._format_feature_selection_overview_df(
            writer.sheets["Feature Selection - Overview"]
        )

        # Write detail sheets for each selector
        for name, selector in selector_steps:
            if name == "Original":
                continue
            if not hasattr(selector, "detail") or selector.detail is None:
                continue

            if name == "Corr" and isinstance(selector.detail, dict):
                detail_df = selector.detail.get("after")
                if detail_df is not None and not detail_df.empty:
                    detail_df.to_excel(
                        writer,
                        sheet_name=f"Feature Selection - {name}",
                        freeze_panes=(1, 1),
                    )
            elif (
                isinstance(selector.detail, pd.DataFrame) and not selector.detail.empty
            ):
                selector.detail.to_excel(
                    writer,
                    sheet_name=f"Feature Selection - {name}",
                    freeze_panes=(1, 1),
                )
                self._format_feature_selection_df(
                    writer.sheets[f"Feature Selection - {name}"]
                )

    def generate_benchmark_report(
        self, writer: pd.ExcelWriter, benchmark_detail: pd.DataFrame
    ) -> None:
        """
        Generate benchmark model analysis report.

        Creates detailed analysis of the benchmark model including model parameters,
        performance metrics, and comparison baseline for main model evaluation.

        Args:
            writer: Excel writer object for output file generation.
            benchmark_detail: Benchmark model details DataFrame.

        Example:
            >>> # Generate benchmark report
            >>> with pd.ExcelWriter('benchmark_analysis.xlsx') as writer:
            ...     reporter.generate_benchmark_report(writer, benchmark_detail)
        """
        if benchmark_detail is None:
            return
        benchmark_detail.to_excel(
            writer, sheet_name="Model - Benchmark Details", freeze_panes=(1, 1)
        )
        self._format_benchmark_df(writer.sheets["Model - Benchmark Details"])

    def generate_model_tuning_report(
        self, writer: pd.ExcelWriter, tune_results: Optional[pd.DataFrame]
    ) -> None:
        """
        Generate hyperparameter tuning results report.

        Creates detailed analysis of hyperparameter optimization including parameter
        combinations tested, performance metrics, and optimal parameter selection.

        Args:
            writer: Excel writer object for output file generation.
            tune_results: DataFrame containing tuning results with hyperparameters and metrics.
                         If None, no tuning report will be generated.

        Example:
            >>> # Generate tuning report (if tuning was performed)
            >>> with pd.ExcelWriter('tuning_results.xlsx') as writer:
            ...     if tune_results is not None:
            ...         reporter.generate_model_tuning_report(writer, tune_results)
        """
        if tune_results is not None:
            tune_results.to_excel(
                writer, sheet_name="Model - Tuning Report", freeze_panes=(1, 1)
            )
            self._format_tuning_df(writer.sheets["Model - Tuning Report"])

    def generate_feature_importance_report(
        self,
        writer: pd.ExcelWriter,
        feature_importance: pd.DataFrame,
    ) -> None:
        """
        Generate feature importance report with visualization.

        Creates a detailed report of feature importance from the trained model,
        including a bar chart visualization and sorted importance values.

        Args:
            writer: Excel writer object for output file generation.
            feature_importance: DataFrame with 'feature' and 'importance' columns.

        Example:
            >>> # Generate feature importance report
            >>> with pd.ExcelWriter('report.xlsx') as writer:
            ...     reporter.generate_feature_importance_report(writer, feat_imp_df)
        """
        if feature_importance is None or len(feature_importance) == 0:
            return

        # Sort by importance descending
        df_imp = feature_importance.sort_values(
            "importance", ascending=False
        ).reset_index(drop=True)
        df_imp.to_excel(
            writer,
            sheet_name="Model - Feature Importance",
            freeze_panes=(1, 1),
            index=False,
        )

        # Format the worksheet
        worksheet = writer.sheets["Model - Feature Importance"]
        self._set_col_width(worksheet, ["A"], 40)
        self._set_col_width(worksheet, ["B"], 15)
        self._set_col_format(worksheet, ["B"], "0.0000")

        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, max(6, len(df_imp) * 0.3)))
        top_n = min(30, len(df_imp))  # Show top 30 features in chart
        df_plot = df_imp.head(top_n).iloc[::-1]  # Reverse for horizontal bar
        ax.barh(df_plot["feature"], df_plot["importance"], color="steelblue")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        ax.set_title(f"Top {top_n} Feature Importance")
        plt.tight_layout()

        # Insert chart into worksheet
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        img_buffer.seek(0)
        img = XLImage(img_buffer)
        img.anchor = "D2"
        worksheet.add_image(img)

    def generate_calibration_report(
        self,
        writer: pd.ExcelWriter,
        calibrator: Calibration,
        scorecard: Dict[str, pd.DataFrame],
        scoredist: Dict[str, pd.DataFrame],
    ) -> None:
        """
        Generate comprehensive model calibration analysis report.

        Creates detailed calibration analysis including regression parameters,
        scorecards for different sample types, score distributions, PSI tracking,
        and optional calibration plot.

        Report Components:
        1. **Calibration - Regression**: Calibration regression parameters and fit metrics
        2. **Calibration - [Sample] Score Card**: Score bands with population and risk metrics
        3. **Calibration - Score Distribution**: Score distribution across samples
        4. **Calibration - Score PSI**: Population Stability Index for score drift monitoring
        5. **Calibration - Plot**: Before/after calibration curves (if available)

        Args:
            writer: Excel writer object for output file generation.
            calibrator: Fitted Calibration instance containing calibrate_detail and
                       optionally calibrate_plot.
            scorecard: Dictionary mapping sample names to scorecard DataFrames:
                      {'train': scorecard_df, 'test': scorecard_df, ...}
            scoredist: Dictionary containing distribution analysis:
                      {'Distribution': dist_df, 'PSI': psi_df}

        Example:
            >>> # Generate calibration report with calibrator (probability mode)
            >>> from optimus.calibrator import Calibration
            >>> calibrator = Calibration()  # probability mode (no mapping_base)
            >>> calibrator.fit(y_prob_train, y_train)
            >>>
            >>> scorecard_data = {'train': train_scorecard, 'test': test_scorecard}
            >>> dist_data = {'Distribution': score_dist, 'PSI': psi_analysis}
            >>> with pd.ExcelWriter('calibration_report.xlsx') as writer:
            ...     reporter.generate_calibration_report(
            ...         writer, calibrator, scorecard_data, dist_data
            ...     )
        """
        if calibrator is None:
            return

        if calibrator.calibrate_detail is not None:
            calibrator.calibrate_detail.to_excel(
                writer, sheet_name="Calibration - Regression", freeze_panes=(1, 1)
            )
            self._format_calibration_reg_df(writer.sheets["Calibration - Regression"])

        for name, df_scorecard in scorecard.items():
            df_scorecard.to_excel(
                writer,
                sheet_name=f"Calibration - {name} Score Card",
                freeze_panes=(1, 1),
            )
            self._format_calibration_scorecard_df(
                writer.sheets[f"Calibration - {name} Score Card"]
            )

        scoredist["Distribution"].to_excel(
            writer,
            sheet_name=f"Calibration - Score Distribution",
            freeze_panes=(3, 1),
        )
        scoredist["PSI"].to_excel(
            writer,
            sheet_name=f"Calibration - Score PSI",
            freeze_panes=(1, 1),
        )
        self._format_calibration_scoredist_df(
            writer.sheets[f"Calibration - Score Distribution"]
        )
        self._format_calibration_score_psi_df(writer.sheets[f"Calibration - Score PSI"])

        # Insert calibration plot if calibrator has a plot
        if calibrator.calibrate_plot is not None:
            # Insert calibration plot with smaller dpi for more compact image
            self._insert_figure_to_excel(
                writer=writer,
                fig=calibrator.calibrate_plot,
                sheet_name="Calibration - Plot",
                title="Calibration Curve (Before vs After)",
                dpi=100,
            )

    def generate_report(self, performance: Dict[str, Any], **kwargs) -> None:
        """
        Generate comprehensive model performance report with all analysis components.

        Creates a complete Excel report containing all available analysis sections
        based on the provided performance data. Use eda_only=True for lightweight
        EDA-only reports at preprocessor stage (no model scores required).

        Report Sections Generated (based on available data):
        - **Sample Overview**: Always generated
            - Full version (with PSI/Performance) if eda_only=False
            - Basic version (sizes/bad rates only) if eda_only=True
        - **Feature Analysis**: Generated if 'woe_df' provided
        - **Feature Selection**: Generated if 'feature_selection' provided
        - **Benchmark Model**: Generated if 'benchmark' provided (eda_only=False)
        - **Model Tuning**: Generated if 'tune_results' provided (eda_only=False)
        - **Calibration Analysis**: Generated if calibration data provided (eda_only=False)

        Args:
            performance: Dictionary containing model performance data:
                Required:
                - **sample_set** (Dict): Sample data with 'train', 'test', etc. keys
                    Each sample must have 'X' and 'y'. Optionally 'e' for extended info.

                Required for eda_only=False:
                - **label** (str): Target column name in 'e' DataFrame.

                Optional:
                - **woe_df** (Dict): WOE analysis by sample type
                - **missing_values** (List): Values to treat as missing (default: [])
                - **feature_selection**: Feature selection pipeline results
                - **benchmark**: Benchmark model analysis
                - **tune_results** (DataFrame): Hyperparameter tuning results
                - **calibrate_detail** (DataFrame): Calibration regression details
                - **scorecard** (Dict): Scorecard analysis by sample
                - **scoredist** (Dict): Score distribution and PSI analysis
            **kwargs: Additional keyword arguments, including eda_only (bool, False by default).

        Output:
            Creates Excel file at self.report_file_name

        Example:
            >>> # EDA-only report (preprocessor stage)
            >>> reporter.generate_report({
            ...     'sample_set': {'train': {'X': X_train, 'y': y_train},
            ...                    'test': {'X': X_test, 'y': y_test}},
            ...     'woe_df': {'train': train_woe, 'test': test_woe},
            ...     'missing_values': [-999999],
            ... }, eda_only=True)

            >>> # Full report with model scores
            >>> reporter.generate_report({
            ...     'sample_set': {'train': {'X': X_train, 'y': y_train, 'e': train_ext},
            ...                    'test': {'X': X_test, 'y': y_test, 'e': test_ext}},
            ...     'label': 'default_flag',
            ...     'woe_df': {'train': train_woe, 'test': test_woe},
            ...     'missing_values': [-999999],
            ...     'feature_selection': preprocessor.named_steps,
            ...     'benchmark': benchmark_results,
            ...     'calibrate_detail': calib_df,
            ...     'scorecard': scorecard_dict,
            ...     'scoredist': scoredist_dict,
            ... })
        """
        writer = pd.ExcelWriter(
            self.report_file_name,
            engine="openpyxl",
        )

        sample_set = performance["sample_set"]

        # Sample Overview
        if kwargs.get("eda_only", False):
            self._generate_sample_overview_basic(writer, sample_set)
        else:
            self.generate_sample_overview_report(
                writer, sample_set, performance["label"]
            )

        # Feature EDA
        if "woe_df" in performance:
            missing_values = performance.get("missing_values", [])
            for sample_type, woe_df in performance["woe_df"].items():
                if sample_type in sample_set:
                    self.generate_single_feature_eda_report(
                        writer,
                        sample_set[sample_type]["X"],
                        sample_set[sample_type]["y"],
                        woe_df,
                        missing_values,
                        sample_type,
                    )

        # Feature Selection
        if "feature_selection" in performance:
            self.generate_feature_selection_report(
                writer, performance["feature_selection"]
            )

        # Benchmark
        if "benchmark_detail" in performance:
            self.generate_benchmark_report(writer, performance["benchmark_detail"])

        # Model Tuning
        if "tune_results" in performance:
            self.generate_model_tuning_report(writer, performance["tune_results"])

        # Feature Importance
        if "feature_importance" in performance:
            self.generate_feature_importance_report(
                writer, performance["feature_importance"]
            )

        if (
            "calibrator" in performance
            and "scorecard" in performance
            and "scoredist" in performance
        ):
            self.generate_calibration_report(
                writer,
                performance["calibrator"],
                performance["scorecard"],
                performance["scoredist"],
            )

        writer.close()
        print(f"[INFO] Report saved to: {self.report_file_name}")
