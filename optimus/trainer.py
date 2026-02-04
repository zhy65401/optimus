import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
import shap
from termcolor import cprint

from .calibrator import IsotonicCalibrator, PlattCalibrator
from .estimator import Estimators
from .metrics import Metrics
from .pipeliner import Model, Preprocess, _DefaultParams
from .reporter import Reporter


class Train:
    """
    A comprehensive machine learning training pipeline for risk modeling and credit scoring.

    This class provides an end-to-end solution for training, calibrating, and evaluating
    machine learning models specifically designed for financial risk assessment.

    Main Methods:
        fit(X, y, e): Train model with automatic encoding, feature selection, tuning, and calibration
        transform(X, y, e, ts): Generate predictions and reports using trained model
        refit_model(ts, trial_index, params): Retrain model with different hyperparameters
        write_report(performance, ts, report_path, report_name): Generate Excel report
        dump_artifacts(): Save all model artifacts to disk

    Attributes:
        ts (str): Timestamp string in format YYYYMMDD_HHMMSS
        version (str): Version identifier for the model
        model_path (str): Directory path for saving model artifacts
        report_path (str): Directory path for saving model reports
        model_type (str): Type of model to train ('LR', 'XGB', 'LGBM')
        tune_method (str): Hyperparameter tuning method ('BO', 'GS')
        mapping_base (Dict): Score mapping base for calibration (None for probability mode)

    Examples:
        >>> # Basic usage
        >>> trainer = Train(
        ...     model_path='./models',
        ...     model_type='LR',
        ...     tune_method='BO'
        ... )
        >>> trainer.fit(X, y, e)
        >>> performance = trainer.transform(X, y, e)

        >>> # Load and use existing model
        >>> predictor = Train(model_path='./models')
        >>> predictions = predictor.transform(X, y, e, ts='20260201_210105')

        >>> # Refit with different hyperparameters
        >>> trainer.refit_model(ts='20260201_210105', trial_index=5)
        >>> new_performance = trainer.transform(X, y, e)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        report_path: Optional[str] = None,
        spec: Optional[Dict[str, Union[str, List[float]]]] = None,
        model_type: str = "LR",
        missing_values: Optional[List[str]] = None,
        impute_strategy: Optional[Dict[str, str]] = None,
        tune_method: str = "BO",
        n_bins: int = 25,
        n_degree: int = 1,
        calibration_method: str = "isotonic",
        max_evals: int = 100,
        mapping_base: Optional[Dict[int, float]] = None,
        score_cap: Optional[float] = None,
        score_floor: Optional[float] = None,
        scale_threshold: Optional[float] = None,
        version: str = "",
        labeled_sample_type: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the training pipeline with specified parameters.

        Args:
            model_path: Directory path for saving model artifacts
            report_path: Directory path for saving model reports
            spec: Dictionary mapping feature names to binning strategies.
                If None (default), uses 'auto' for all features.
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
            model_type: Type of model to train. Options:
                - 'LR': Logistic Regression
                - 'XGB': XGBoost
                - 'LGBM': LightGBM
            missing_values: List of values to treat as missing
            impute_strategy: Dictionary mapping feature names to imputation strategies.
                Options: 'mean', 'median', 'min', 'max', 'mode', 'separate'.
                If None (default), no imputation is performed before WOE encoding.
            tune_method: Hyperparameter tuning method. Options:
                - 'BO': Bayesian Optimization
                - 'GS': Grid Search
            n_bins: Number of bins for score calibration
            n_degree: Degree for polynomial features in calibration (only for polynomial method)
            calibration_method: Calibration method to use. Options:
                - 'polynomial': Polynomial fitting in log-odds space (default)
                - 'isotonic': Isotonic regression in probability space
            max_evals: Maximum evaluations for hyperparameter tuning
            mapping_base: Custom score mapping dictionary. If provided, transforms
                probabilities to credit scores. If None (default), outputs calibrated
                probabilities directly.
            score_cap: Maximum score value (required when mapping_base is provided, or for isotonic auto-mapping)
            score_floor: Minimum score value (required when mapping_base is provided, or for isotonic auto-mapping)
            scale_threshold: High-risk threshold for isotonic auto-mapping (0-1 range).
                Only applies when calibration_method='isotonic' and score_cap/score_floor provided without mapping_base.
            version: Version identifier for the model
            labeled_sample_type: List of sample types to include in training
            **kwargs: Additional parameters passed to preprocessing and modeling

        Raises:
            ValueError: If invalid model_type or tune_method is specified.

        Examples:
            >>> # Simple initialization
            >>> trainer = Train(model_path='./models')

            >>> # With custom binning specification
            >>> spec = {
            ...     'age': 'bestKS',
            ...     'income': [0, 30000, 60000, 100000, float('inf')],
            ...     'education': False
            ... }
            >>> trainer = Train(model_path='./models', spec=spec)

            >>> # Advanced initialization with custom parameters
            >>> trainer = Train(
            ...     model_path='./models',
            ...     spec=spec,
            ...     model_type='XGB',
            ...     tune_method='BO',
            ...     max_evals=150,
            ...     corr_threshold=0.95,
            ...     iv_threshold=0.02
            ... )
        """
        self.ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.version = version
        self.model_path = model_path
        self.report_path = report_path
        self.spec = spec
        self.missing_values = missing_values or _DefaultParams.missing_values.value
        self.impute_strategy = impute_strategy
        self.labeled_sample_type = ["train", "test"] + (labeled_sample_type or [])

        self.model_type = model_type
        self.tune_method = tune_method
        self.max_evals = max_evals
        self.n_bins = n_bins
        self.n_degree = n_degree
        self.calibration_method = calibration_method
        self.mapping_base = mapping_base
        self.score_cap = score_cap
        self.score_floor = score_floor
        self.scale_threshold = scale_threshold

        self._artifacts = {}
        # Preprocessing arguments
        self.corr_threshold = kwargs.get(
            "corr_threshold", _DefaultParams.corr_threshold.value
        )
        self.psi_threshold = kwargs.get(
            "psi_threshold", _DefaultParams.psi_threshold.value
        )
        self.iv_threshold = kwargs.get(
            "iv_threshold", _DefaultParams.iv_threshold.value
        )
        self.vif_threshold = kwargs.get(
            "vif_threshold", _DefaultParams.vif_threshold.value
        )
        self.boosting_select_frac = kwargs.get(
            "boosting_select_frac", _DefaultParams.boosting_select_frac.value
        )
        self.stability_threshold = kwargs.get(
            "stability_threshold", _DefaultParams.stability_threshold.value
        )
        self.treat_missing = kwargs.get(
            "treat_missing", _DefaultParams.treat_missing.value
        )
        self.user_model = kwargs.get("user_model", None)
        self.user_params = kwargs.get("user_params", None)
        self.ignore_preprocessors = kwargs.get("ignore_preprocessors", [])
        self.drop_features = kwargs.get("drop_features", [])

        self.score_bins = kwargs.get("score_bins", 10)
        self.sample_types = kwargs.get("sample_types", ["train", "test"])

    def fit(self, X: pd.DataFrame, y: pd.Series, e: pd.DataFrame):
        """
        Train the machine learning model with preprocessing, feature selection, and calibration.

        This method performs the complete training pipeline:
        1. Input validation and data splitting
        2. Feature preprocessing (WOE encoding)
        3. Feature selection (correlation, IV, PSI, VIF, etc.)
        4. Model training with hyperparameter tuning
        5. Model calibration for score mapping
        6. Artifact saving

        Args:
            X (pd.DataFrame): Feature matrix with shape (n_samples, n_features).
            y (pd.Series): Target variable with binary labels (0, 1).
            e (pd.DataFrame): External data containing 'sample_type' column with 'train' and 'test' labels.

        Returns:
            Train: Self instance for method chaining.

        Raises:
            ValueError: If input data is empty, inconsistent lengths, or missing required sample types.
            KeyError: If 'sample_type' column is missing from external data.
            FileNotFoundError: If model_path is None or invalid.
            NotImplementedError: If unsupported model_type is specified.

        Example:
            >>> trainer = Train(model_path='/path/to/models', model_type='LR')
            >>> trainer.fit(X_train, y_train, external_data)
        """
        # Input validation
        if X.empty or y.empty or e.empty:
            raise ValueError("Input data cannot be empty.")

        if len(X) != len(y) or len(X) != len(e):
            raise ValueError("X, y, and e must have the same length.")

        if "sample_type" not in e.columns:
            raise KeyError("External data must contain 'sample_type' column.")

        self._artifacts = {}
        required_types = {"train", "test"}
        available_types = set(e["sample_type"].unique())
        if not required_types.issubset(available_types):
            raise ValueError(
                f"External data must contain 'train' and 'test' sample types, but got {available_types}."
            )

        if self.model_path is None:
            raise ValueError("model_path must be specified before fitting.")
        if not os.path.exists(self.model_path):
            cprint(
                f"[WARN] Model path {self.model_path} does not exist, creating...",
                "yellow",
            )
            os.makedirs(self.model_path, exist_ok=True)
        train_mask = e["sample_type"] == "train"
        test_mask = e["sample_type"] == "test"
        trainX = X[train_mask]
        trainy = y[train_mask]
        testX = X[test_mask]
        testy = y[test_mask]
        self._artifacts["fitting_set"] = {
            "trainX": trainX,
            "trainy": trainy,
            "testX": testX,
            "testy": testy,
        }

        if self.spec is not None:
            spec = self.spec
        else:
            spec = {col: "auto" for col in X.columns}

        preprocessor = Preprocess(
            spec=spec,
            impute_strategy=self.impute_strategy,
            corr_threshold=self.corr_threshold,
            psi_threshold=self.psi_threshold,
            iv_threshold=self.iv_threshold,
            vif_threshold=self.vif_threshold,
            boosting_select_frac=self.boosting_select_frac,
            stability_threshold=self.stability_threshold,
            missing_values=self.missing_values,
            treat_missing=self.treat_missing,
            ignore_preprocessors=self.ignore_preprocessors,
            drop_features=self.drop_features,
        )

        preprocessor.fit(trainX, trainy, X_ref=testX, y_ref=testy)

        train_set = preprocessor.transform(trainX)
        test_set = preprocessor.transform(testX)
        self._artifacts["preprocessor"] = preprocessor

        if train_set.shape[1] == 0 or test_set.shape[1] == 0:
            cprint("[ERROR] No features remaining after preprocessing!", "red")
            self.dump_artifacts()
            return self

        # Train main model on the same feature set (after preprocessing)
        model_generator = Model(
            model_type=self.model_type,
            tune_method=self.tune_method,
            max_evals=self.max_evals,
            param_grid=None,
        ).build_model(self.user_model)
        if self.user_model is None:
            if self.model_type in ["LR", "XGB", "LGBM"]:
                tuner = model_generator.fit(train_set, trainy, test_set, testy)

            else:
                raise NotImplementedError(
                    f"{self.model_type} model has not implemented yet!"
                )
            model = tuner.best_estimator
        else:
            model = model_generator.fit(train_set, trainy)
            tuner = None
        train_proba = model.predict_proba(train_set)[:, 1]
        self._artifacts["tuner"] = tuner
        self._artifacts["model"] = model

        if self.calibration_method == "isotonic":
            calibrator = IsotonicCalibrator(
                n_bins=self.n_bins,
                score_floor=self.score_floor or 0.0,
                score_cap=self.score_cap or 1.0,
                scale_threshold=self.scale_threshold,
            )
        else:
            calibrator = PlattCalibrator(
                n_bins=self.n_bins,
                n_degree=self.n_degree,
                mapping_base=self.mapping_base,
                score_cap=self.score_cap,
                score_floor=self.score_floor,
            )
        calibrator = calibrator.fit(train_proba, trainy)
        self._artifacts["calibrator"] = calibrator
        self.dump_artifacts()

        return self

    def transform(self, X: pd.DataFrame, y: pd.Series, e: pd.DataFrame, ts: str = None):
        """
        Apply trained model to generate predictions and performance metrics.

        This method loads a previously trained model and applies it to new data:
        1. Load model artifacts from specified timestamp
        2. Apply preprocessing and feature selection
        3. Generate predictions and scores
        4. Calculate performance metrics and scorecards
        5. Generate comprehensive performance report

        Args:
            X (pd.DataFrame): Feature matrix for prediction with shape (n_samples, n_features).
            y (pd.Series): Target variable for performance evaluation.
            e (pd.DataFrame): External data with 'sample_type' column indicating data splits.
            ts (str, optional): Timestamp of the model to load. If None, uses the latest training timestamp.

        Returns:
            dict: Comprehensive performance dictionary containing:
                - version: Model version
                - model_id: Model timestamp identifier
                - missing_values: List of missing value indicators
                - label: Target variable name
                - feature_selection: Feature selection pipeline details
                - tune_results: Hyperparameter tuning results
                - calibrate_detail: Score calibration details
                - scorecard: Scorecard analysis for labeled sample types
                - woe_df: WOE analysis with 'binning' and 'summary' DataFrames
                - predictions: Prediction results by sample type

        Raises:
            ValueError: If timestamp is not provided and no training timestamp exists.
            KeyError: If 'sample_type' column is missing from external data.
            FileNotFoundError: If model directory or artifacts don't exist.

        Example:
            >>> performance = trainer.transform(X_test, y_test, external_data, ts='20241201_143022')
            >>> print(performance['scorecard']['test'])
        """
        self.ts = ts or self.ts
        if not self.ts:
            raise ValueError("Please provide a timestamp to transform.")
        if "sample_type" not in e:
            raise KeyError("external data must contain 'sample_type' column.")
        # Make sure train and test are in the first place
        sample_types = self.sample_types + list(
            set(e["sample_type"].unique().tolist()) - set(self.sample_types)
        )
        if len(sample_types) > 10:
            raise ValueError(f"Over 10 sample types: {sample_types}!")
        # Load model artifacts
        model_dir = os.path.join(self.model_path, self.ts)
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory {model_dir} does not exist.")

        if self._artifacts == {}:
            artifact_names = [
                "preprocessor",
                "model",
                "calibrator",
                "tuner",
            ]
            for name in artifact_names:
                artifact_path = os.path.join(model_dir, name)
                if not os.path.exists(artifact_path):
                    cprint("[ERROR] Model is not complete!", "red")
                    return self._artifacts
                self._artifacts[name] = joblib.load(artifact_path)

        preprocess_pipe = self._artifacts["preprocessor"]
        model = self._artifacts["model"]
        calibrator = self._artifacts["calibrator"]
        tuner = self._artifacts["tuner"]

        label = y.name
        predictions = {}
        woe_dfs = {}
        scorecards = {}
        df_distribution = []

        for sample_type in sample_types:
            sample_mask = e["sample_type"] == sample_type
            X_ = X[sample_mask]
            y_ = y[sample_mask]
            e_ = e[sample_mask].copy()

            transX = preprocess_pipe.transform(X_)
            if (
                sample_type in self.labeled_sample_type
                and "WOE" in preprocess_pipe.named_steps
            ):
                woe_dfs[sample_type] = preprocess_pipe.get_step("WOE").get_woe_df(
                    X_, y_
                )

            proba = model.predict_proba(transX)[:, 1]
            score = calibrator.transform(proba)

            if sample_type in self.labeled_sample_type:
                scorecards[sample_type] = calibrator.compare_calibrate_result(
                    score, y_, bins=self.score_bins
                )

            e_.loc[:, "proba"] = [round(i, 6) for i in proba]
            e_.loc[:, "score"] = [round(i, 6) for i in score]
            e_.loc[:, "score_bin"] = pd.cut(e_["score"], bins=self.score_bins)

            # Add instance-level SHAP explanations (JSON format)
            if self.model_type in ["LR", "XGB", "LGBM"]:
                try:
                    # Determine model type for SHAP
                    if self.model_type == "LR":
                        explainer = shap.LinearExplainer(model, transX)
                    else:  # XGB or LGBM
                        explainer = shap.TreeExplainer(model)

                    shap_values = explainer.shap_values(transX)

                    # For binary classification, shap_values might be 2D array
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]  # Get positive class

                    # Convert SHAP values to JSON for each instance
                    feature_names = list(transX.columns)
                    shap_json_list = []
                    for i in range(len(transX)):
                        shap_dict = {
                            feat: round(float(shap_values[i, j]), 6)
                            for j, feat in enumerate(feature_names)
                        }
                        shap_json_list.append(json.dumps(shap_dict, ensure_ascii=False))

                    e_.loc[:, "shap_values"] = shap_json_list
                except Exception as ex:
                    cprint(
                        f"[WARN] Failed to compute SHAP values for {sample_type}: {ex}",
                        "yellow",
                    )
                    e_.loc[:, "shap_values"] = None

            score_dist = (
                e_.groupby(["score_bin"], observed=False)
                .agg(
                    total=("score", "count"),
                    pct=("score", lambda x: len(x) / e_.shape[0]),
                )
                .sort_index()
            )
            score_dist.columns = pd.MultiIndex.from_tuples(
                [(sample_type, i) for i in ["# Score", "% Score"]]
            )
            df_distribution.append(score_dist)

            predictions[sample_type] = e_

        df_distribution = pd.concat(df_distribution, axis=1)

        # Calculate PSI for each sample type using Metrics module
        psi_results = {}
        train_dist = df_distribution[("train", "% Score")]
        for sample_type in sample_types:
            if sample_type != "train":
                test_dist = df_distribution[(sample_type, "% Score")]
                psi = Metrics.get_psi_from_distributions(train_dist, test_dist)
                psi_results[sample_type] = psi

        df_psi = pd.DataFrame.from_dict(
            psi_results, orient="index", columns=["% Score PSI"]
        )
        df_psi = df_psi.sort_values("% Score PSI")

        fs_steps = {
            name: step
            for name, step in preprocess_pipe.named_steps.items()
            if name not in ["WOE", "Impute"]
        }
        feature_names = (
            list(model.feature_names_in_)
            if hasattr(model, "feature_names_in_")
            else None
        )
        if feature_names:
            if hasattr(model, "feature_importances_"):
                feature_importance = pd.DataFrame(
                    {"feature": feature_names, "importance": model.feature_importances_}
                )
            elif hasattr(model, "coef_"):
                coef = (
                    model.coef_.flatten()
                    if hasattr(model.coef_, "flatten")
                    else model.coef_
                )
                feature_importance = pd.DataFrame(
                    {"feature": feature_names, "importance": np.abs(coef)}
                )
            else:
                feature_importance = None

        performance = {
            "version": self.version,
            "model_id": self.ts,
            "missing_values": self.missing_values,
            "label": label,
            "feature_selection": fs_steps,
            "tune_results": tuner.results,
            "best_params": tuner.best_params if hasattr(tuner, "best_params") else None,
            "feature_importance": feature_importance,
            "calibrator": calibrator,
            "scoredist": {"Distribution": df_distribution, "PSI": df_psi},
            "scorecard": scorecards,
            "woe_df": woe_dfs,
            "predictions": predictions,
        }

        performance_path = os.path.join(model_dir, "performance")
        joblib.dump(performance, performance_path)
        cprint(f"Performance file successfully dumped in path {model_dir}", "green")

        return performance

    def dump_artifacts(self):
        """
        Save all trained model artifacts to disk.

        Saves model components (preprocessor, model, calibrator, tuner)
        to the model directory using joblib serialization.

        The artifacts are saved to: {model_path}/{timestamp}/{artifact_name}

        Example:
            >>> trainer.fit(X_train, y_train, external_data)
            >>> trainer.dump_artifacts()  # Saves to ./models/20241201_143022/
        """
        model_dir = os.path.join(os.path.abspath(self.model_path), self.ts)
        os.makedirs(model_dir, exist_ok=True)
        for name, artifact in self._artifacts.items():
            joblib.dump(artifact, os.path.join(model_dir, name))
        cprint(f"Model files successfully dumped in path {model_dir}", "green")

    def write_report(
        self,
        performance: Optional[Dict[str, Any]] = None,
        ts: Optional[str] = None,
        report_path: Optional[str] = None,
        report_name: Optional[str] = None,
    ):
        """
        Generate comprehensive Excel report from model performance data.

        Creates a detailed Excel report containing sample overview, feature analysis,
        model performance, calibration metrics, and scorecards.

        Args:
            performance: Performance dictionary from transform(). If None, loads from saved artifacts.
            ts: Model timestamp. If None, uses current training timestamp.
            report_path: Directory to save the report. If None, uses initialized report_path.
            report_name: Custom report filename (without .xlsx). If None, uses "model_report_{ts}".

        Returns:
            str: Full path to the generated report file

        Raises:
            ValueError: If timestamp or report_path is not provided
            FileNotFoundError: If performance data cannot be found when performance is None

        Example:
            >>> trainer.fit(X_train, y_train, external_data)
            >>> performance = trainer.transform(X_test, y_test, external_data)
            >>> trainer.write_report(performance, report_name='risk_model_v1')
        """
        ts = ts or self.ts
        if not ts:
            raise ValueError(
                "Timestamp must be provided either via ts parameter or from training/transform."
            )

        if performance is None:
            model_dir = os.path.join(self.model_path, ts)
            performance_path = os.path.join(model_dir, "performance")

            if not os.path.exists(performance_path):
                raise FileNotFoundError(
                    f"Performance file not found at {performance_path}. "
                    f"Please run transform() first or provide performance dictionary."
                )

            performance = joblib.load(performance_path)
            cprint(f"[INFO] Loaded performance from {performance_path}", "cyan")

        report_path = report_path or self.report_path
        if not report_path:
            raise ValueError(
                "Report path must be specified either via report_path parameter "
                "or during initialization."
            )

        if not os.path.exists(report_path):
            cprint(
                f"[WARN] Report path {report_path} does not exist, creating...",
                "yellow",
            )
            os.makedirs(report_path, exist_ok=True)

        if report_name:
            report_file = f"{report_path}/{report_name}.xlsx"
        else:
            report_file = f"{report_path}/model_report_{ts}.xlsx"

        Reporter(report_file).generate_report(performance)
        cprint(f"[SUCCESS] Model report written to {report_file}", "green")

        return report_file

    def refit_model(
        self,
        ts: str,
        trial_index: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        """
        Retrain model with different hyperparameters and replace artifacts in the original directory.

        Loads a previously trained model and refits it with either:
        - Hyperparameters from a specific trial index in the tuning results, or
        - Custom hyperparameters provided directly

        The refitted model and calibrator will replace the original artifacts in the ts directory.
        After refitting, use transform() with the same ts to regenerate performance metrics.

        Args:
            ts: Timestamp of the original trained model to refit
            trial_index: Index of the trial from tuning results to use (0-based).
                Mutually exclusive with params.
            params: Custom hyperparameters dictionary to use for refitting.
                Mutually exclusive with trial_index.

        Raises:
            ValueError: If neither or both trial_index and params are provided
            FileNotFoundError: If model directory doesn't exist
            IndexError: If trial_index is out of range

        Example:
            >>> # Refit using 5th trial from tuning results
            >>> trainer.refit_model(ts='20241201_143022', trial_index=5)
            >>> # Then regenerate performance
            >>> performance = trainer.transform(X, y, e, ts='20241201_143022')
            >>>
            >>> # Refit with custom parameters
            >>> custom_params = {'max_depth': 5, 'learning_rate': 0.01}
            >>> trainer.refit_model(ts='20241201_143022', params=custom_params)
            >>> performance = trainer.transform(X, y, e, ts='20241201_143022')
        """
        if trial_index is None and params is None:
            raise ValueError("Either trial_index or params must be provided.")
        self.ts = ts
        model_dir = os.path.join(self.model_path, self.ts)

        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory {model_dir} does not exist.")

        fitting_set = joblib.load(os.path.join(model_dir, "fitting_set"))
        preprocessor = joblib.load(os.path.join(model_dir, "preprocessor"))
        tuner = joblib.load(os.path.join(model_dir, "tuner"))

        trainX = fitting_set["trainX"]
        trainy = fitting_set["trainy"]

        # Determine which parameters to use
        if params is not None:
            selected_params = params
            cprint(f"[INFO] Using custom parameters: {selected_params}", "cyan")
        else:
            # Get parameters from tuning results by index
            if trial_index < 0 or trial_index >= len(tuner.results):
                raise IndexError(
                    f"trial_index {trial_index} is out of range. "
                    f"Valid range: 0 to {len(tuner.results) - 1}"
                )

            # Extract hyperparameters from results (exclude metric columns)
            result_row = tuner.results.iloc[trial_index]
            metric_cols = [
                "iteration",
                "loss",
                "status",
                "cv_train_auc",
                "cv_valid_auc",
                "cv_train_ks",
                "cv_valid_ks",
                "cv_train_avg_auc",
                "cv_valid_avg_auc",
                "cv_train_avg_ks",
                "cv_valid_avg_ks",
                "cv_ks_gap",
                "train_auc",
                "test_auc",
                "train_ks",
                "test_ks",
                "ks_gap",
            ]
            selected_params = {
                k: v
                for k, v in result_row.to_dict().items()
                if k not in metric_cols and pd.notna(v)
            }
            cprint(
                f"[INFO] Using parameters from trial {trial_index}: {selected_params}",
                "cyan",
            )

        # Get preprocessed training data
        train_set = preprocessor.transform(trainX)

        model = Estimators[self.model_type].value
        model.set_params(**selected_params)
        model = model.fit(train_set, trainy)

        train_proba = model.predict_proba(train_set)[:, 1]
        if self.calibration_method == "isotonic":
            calibrator = IsotonicCalibrator(
                n_bins=self.n_bins,
                score_floor=self.score_floor if self.score_floor is not None else 0.0,
                score_cap=self.score_cap if self.score_cap is not None else 1.0,
                scale_threshold=self.scale_threshold,
            )
        else:
            calibrator = PlattCalibrator(
                n_bins=self.n_bins,
                n_degree=self.n_degree,
                mapping_base=self.mapping_base,
                score_cap=self.score_cap,
                score_floor=self.score_floor,
            )
        calibrator = calibrator.fit(train_proba, trainy)

        cprint(f"[INFO] Replacing model and calibrator in {model_dir}", "cyan")
        joblib.dump(model, os.path.join(model_dir, "model"))
        joblib.dump(calibrator, os.path.join(model_dir, "calibrator"))

        # Update artifacts in memory
        self._artifacts = {
            "fitting_set": fitting_set,
            "preprocessor": preprocessor,
            "model": model,
            "calibrator": calibrator,
            "tuner": tuner,
        }

        cprint(f"[SUCCESS] Refitted model saved in path {model_dir}", "green")
        cprint(
            f"[INFO] Model and calibrator have been replaced. "
            f"Parameters used: {selected_params}",
            "cyan",
        )

        return self
