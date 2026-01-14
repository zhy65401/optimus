import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from termcolor import cprint

from .calibrator import Calibration
from .estimator import Benchmark
from .pipeliner import Model, Preprocess, _DefaultParams
from .reporter import Reporter


class Train:
    """
    A comprehensive machine learning training pipeline for risk modeling and credit scoring.

    This class provides an end-to-end solution for training, calibrating, and evaluating
    machine learning models specifically designed for financial risk assessment.

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
        >>> results = trainer.transform(X, y, e)

        >>> # Advanced configuration
        >>> trainer = Train(
        ...     model_path='./models',
        ...     report_path='./reports',
        ...     model_type='XGB',
        ...     tune_method='BO',
        ...     max_evals=200,
        ...     corr_threshold=0.9,
        ...     iv_threshold=0.03
        ... )
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
        max_evals: int = 100,
        mapping_base: Optional[Dict[int, float]] = None,
        score_cap: Optional[float] = None,
        score_floor: Optional[float] = None,
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
            n_degree: Degree for polynomial features in calibration
            max_evals: Maximum evaluations for hyperparameter tuning
            mapping_base: Custom score mapping dictionary. If provided, transforms
                probabilities to credit scores. If None (default), outputs calibrated
                probabilities directly.
            score_cap: Maximum score value (required when mapping_base is provided)
            score_floor: Minimum score value (required when mapping_base is provided)
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
        # Default Mapping Base:
        # {
        #     500: 0.128,
        #     550: 0.0671,
        #     600: 0.0341,
        #     650: 0.017,
        #     700: 0.0084,
        #     750: 0.0041,
        #     800: 0.002,
        #     850: 0.001
        # }
        self.mapping_base = mapping_base
        self.score_cap = score_cap
        self.score_floor = score_floor

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
        self.ignore_preprocessors = kwargs.get("ignore_preprocessors", [])
        self.drop_features = kwargs.get("drop_features", [])

        self.score_bins = kwargs.get(
            "score_bins",
            (
                [self.score_floor] + sorted(self.mapping_base.keys()) + [self.score_cap]
                if self.mapping_base
                else [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            ),
        )
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

        # Build feature specification
        # Use custom spec if provided, otherwise default to 'auto' for all features
        if self.spec is not None:
            spec = self.spec
        else:
            spec = {col: "auto" for col in X.columns}

        # Initialize and fit preprocessor with new API
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

        # Fit with reference data (test set) for PSI/GINI/Boosting selectors
        preprocessor.fit(trainX, trainy, X_ref=testX, y_ref=testy)

        train_set = preprocessor.transform(trainX)
        test_set = preprocessor.transform(testX)

        model_generator = Model(
            model_type=self.model_type,
            tune_method=self.tune_method,
            max_evals=self.max_evals,
            param_grid=None,
        ).build_model()
        if self.model_type in ["LR", "XGB", "LGBM"]:
            tuner = model_generator.fit(train_set, trainy, test_set, testy)
        else:
            raise NotImplementedError(
                f"{self.model_type} model has not implemented yet!"
            )
        model = tuner.best_estimator.fit(train_set, trainy)
        train_proba = model.predict_proba(train_set)[:, 1]
        calibrator = Calibration(
            n_bins=self.n_bins,
            n_degree=self.n_degree,
            mapping_base=self.mapping_base,
            score_cap=self.score_cap,
            score_floor=self.score_floor,
        )
        calibrator = calibrator.fit(train_proba, trainy)
        fitting_set = {
            "trainX": trainX,
            "trainy": trainy,
            "testX": testX,
            "testy": testy,
        }

        # Save model artifacts
        model_dir = os.path.join(os.path.abspath(self.model_path), self.ts)
        os.makedirs(model_dir, exist_ok=True)

        artifacts = {
            "fitting_set": fitting_set,
            "preprocessor": preprocessor,
            "model": model,
            "calibrator": calibrator,
            "tuner": tuner,
        }

        for name, artifact in artifacts.items():
            joblib.dump(artifact, os.path.join(model_dir, name))

        cprint(f"Model files successfully dumped in path {model_dir}", "green")

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
                - woe_df: Weight of Evidence analysis for labeled sample types
                - sample_set: Processed datasets with predictions

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

        artifacts = {}
        artifact_names = ["preprocessor", "model", "calibrator", "tuner"]

        for name in artifact_names:
            artifact_path = os.path.join(model_dir, name)
            if not os.path.exists(artifact_path):
                raise FileNotFoundError(
                    f"Model artifact {artifact_path} does not exist."
                )
            artifacts[name] = joblib.load(artifact_path)

        preprocess_pipe = artifacts["preprocessor"]
        model = artifacts["model"]
        calibrator = artifacts["calibrator"]
        tuner = artifacts["tuner"]

        label = y.name
        sample_set = {}
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

            bm_proba = preprocess_pipe.get_step("Benchmark").predict_proba(transX)[:, 1]
            proba = model.predict_proba(transX)[:, 1]
            score = calibrator.transform(proba)

            if sample_type in self.labeled_sample_type:
                scorecards[sample_type] = calibrator.compare_calibrate_result(
                    score, y_, self.score_bins
                )

            e_.loc[:, "bm_proba"] = [round(i, 6) for i in bm_proba]
            e_.loc[:, "proba"] = [round(i, 6) for i in proba]
            e_.loc[:, "score"] = [round(i, 6) for i in score]
            e_.loc[:, "score_bin"] = pd.cut(e_["score"], bins=self.score_bins)

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

            sample_set[sample_type] = {"X": X_, "y": y_, "e": e_}

        df_distribution = pd.concat(df_distribution, axis=1)
        df_psi = (
            df_distribution.apply(
                lambda x: (
                    np.sum(
                        (x - df_distribution[("train", "% Score")])
                        * np.log(
                            x
                            / (
                                df_distribution[("train", "% Score")]
                                + np.finfo(float).eps
                            )
                            + np.finfo(float).eps
                        )
                    )
                    if x.name[1] == "% Score"
                    else np.nan
                )
            )
            .dropna()
            .droplevel(1)
            .to_frame(name="% Score PSI")
            .sort_values("% Score PSI")
        )

        benchmark_detail = preprocess_pipe.get_step("Benchmark").model_detail
        fs_steps = {
            name: step
            for name, step in preprocess_pipe.named_steps.items()
            if name not in ["WOE", "Benchmark", "Impute"]
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

        # Prepare SHAP analysis data
        shap_data = None
        if self.model_type in ["LR", "XGB", "LGBM"]:
            # Determine model type for SHAP
            if self.model_type == "LR":
                shap_model_type = "linear"
            else:
                shap_model_type = "tree"

            # Use train set for SHAP analysis
            train_X = sample_set.get("train", {}).get("X")
            train_y = sample_set.get("train", {}).get("y")

            if train_X is not None and len(train_X) > 0:
                shap_data = {
                    "model": model,
                    "X": train_X,
                    "y": train_y,
                    "sample_size": 1000,
                    "model_type": shap_model_type,
                }

        performance = {
            "version": self.version,
            "model_id": self.ts,
            "missing_values": self.missing_values,
            "label": label,
            "feature_selection": fs_steps,
            "benchmark_detail": benchmark_detail,
            "tune_results": tuner.results,
            "feature_importance": feature_importance,
            "shap_data": shap_data,
            "calibrator": calibrator,
            "scoredist": {"Distribution": df_distribution, "PSI": df_psi},
            "scorecard": scorecards,
            "woe_df": woe_dfs,
            "sample_set": sample_set,
        }

        performance_path = os.path.join(model_dir, "performance")
        joblib.dump(performance, performance_path)
        cprint(f"Performance file successfully dumped in path {model_dir}", "green")

        if self.report_path:
            if not os.path.exists(self.report_path):
                cprint(
                    f"[WARN] Report path {self.report_path} does not exist, creating...",
                    "yellow",
                )
                os.makedirs(self.report_path, exist_ok=True)
            Reporter(f"{self.report_path}/model_report_{self.ts}.xlsx").generate_report(
                performance
            )
            cprint(
                f"Model report successfully dumped in {self.report_path}/model_report_{self.ts}.xlsx",
                "green",
            )
        else:
            cprint("Report path not specified, skipping report generation", "yellow")

        return performance

    def refit(
        self,
        ts: str,
        trial_index: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        """
        Refit the model with a different hyperparameter configuration from tuning results.

        This method allows you to select a different model from the tuning results
        without re-running the entire preprocessing pipeline. It loads the saved
        preprocessor and fitting set, then trains a new model with the specified
        hyperparameters.

        Args:
            ts (str): Timestamp of the model to refit.
            trial_index (int, optional): Index of the trial in tuning results to use.
                If provided, the hyperparameters from that trial will be used.
            params (dict, optional): Custom hyperparameters to use for training.
                If provided, this takes precedence over trial_index.

        Returns:
            Train: Self instance for method chaining.

        Raises:
            ValueError: If neither trial_index nor params is provided.
            FileNotFoundError: If model artifacts don't exist.
            IndexError: If trial_index is out of range.

        Example:
            >>> # After running fit() and reviewing tuning results
            >>> trainer.refit(ts='20241201_143022', trial_index=5)
            >>> # Or with custom parameters
            >>> trainer.refit(ts='20241201_143022', params={'C': 0.5})
        """
        if trial_index is None and params is None:
            raise ValueError("Either trial_index or params must be provided.")

        self.ts = ts
        model_dir = os.path.join(self.model_path, self.ts)

        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory {model_dir} does not exist.")

        # Load required artifacts
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

        # Create and train new model with selected parameters
        from .estimator import Estimators

        model = Estimators[self.model_type].value
        model.set_params(**selected_params)
        model = model.fit(train_set, trainy)

        # Generate new calibrator
        train_proba = model.predict_proba(train_set)[:, 1]
        calibrator = Calibration(
            n_bins=self.n_bins,
            n_degree=self.n_degree,
            mapping_base=self.mapping_base,
            score_cap=self.score_cap,
            score_floor=self.score_floor,
        )
        calibrator = calibrator.fit(train_proba, trainy)

        # Update saved artifacts with new model and calibrator
        # Create a new timestamp for the refitted model
        new_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_model_dir = os.path.join(os.path.abspath(self.model_path), new_ts)
        os.makedirs(new_model_dir, exist_ok=True)

        artifacts = {
            "fitting_set": fitting_set,
            "preprocessor": preprocessor,
            "model": model,
            "calibrator": calibrator,
            "tuner": tuner,
            "refit_info": {
                "original_ts": ts,
                "trial_index": trial_index,
                "params": selected_params,
            },
        }

        for name, artifact in artifacts.items():
            joblib.dump(artifact, os.path.join(new_model_dir, name))

        self.ts = new_ts
        cprint(f"[SUCCESS] Refitted model saved in path {new_model_dir}", "green")
        cprint(f"[INFO] Use ts='{new_ts}' for transform()", "cyan")

        return self
