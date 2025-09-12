import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from termcolor import cprint

from .calibrator import Calibration
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
        score_type (str): Type of score calibration ('mega_score', 'sub_score', etc.)
    """

    def __init__(
        self,
        model_path: str = None,
        report_path: str = None,
        model_type: str = "LR",
        missing_values: list[str] = None,
        tune_method: str = "BO",
        n_bins: int = 25,
        n_degree: int = 1,
        max_evals: int = 100,
        score_type: str = "mega_score",
        mapping_base: dict = None,
        score_cap: float = 300,
        score_floor: float = 1000,
        version: str = "",
        labeled_sample_type: list = None,
        **kwargs,
    ):
        """
        Initialize the training pipeline with specified parameters.

        Args:
            model_path (str, optional): Directory path for saving model artifacts. Required for fit().
            report_path (str, optional): Directory path for saving model reports.
            model_type (str): Machine learning model type. Options: 'LR', 'XGB', 'LGBM'. Default: 'LR'.
            missing_values (list[str], optional): List of values to treat as missing. Default: predefined list.
            tune_method (str): Hyperparameter tuning method. Options: 'BO' (Bayesian), 'GS' (Grid Search). Default: 'BO'.
            n_bins (int): Number of bins for calibration. Default: 25.
            n_degree (int): Polynomial degree for calibration. Default: 1.
            max_evals (int): Maximum evaluations for hyperparameter tuning. Default: 100.
            score_type (str): Score calibration type. Options: 'mega_score', 'sub_score', 'probability', 'self-defining'. Default: 'mega_score'.
            mapping_base (dict, optional): Custom score mapping base for 'self-defining' score_type.
            score_cap (float): Maximum score value. Default: 300.
            score_floor (float): Minimum score value. Default: 1000.
            version (str): Version identifier for tracking model versions.
            labeled_sample_type (list, optional): Additional sample types with labels for scorecard generation.
            **kwargs: Additional preprocessing parameters (corr_threshold, psi_threshold, iv_threshold, etc.)

        Raises:
            ValueError: If invalid model_type or tune_method is specified.
        """

        self.ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.version = version
        self.model_path = model_path
        self.report_path = report_path
        self.missing_values = missing_values or _DefaultParams.missing_values.value
        self.labeled_sample_type = ["train", "test"] + (labeled_sample_type or [])

        self.model_type = model_type
        self.tune_method = tune_method
        self.max_evals = max_evals
        self.n_bins = n_bins
        self.n_degree = n_degree
        self.score_type = score_type
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
        self.select_frac = kwargs.get("select_frac", _DefaultParams.select_frac.value)
        self.treat_missing = kwargs.get(
            "treat_missing", _DefaultParams.treat_missing.value
        )
        self.ignore_preprocessors = kwargs.get("ignore_preprocessors", [])
        self.drop_features = kwargs.get("drop_features", [])

        self.score_bins = kwargs.get(
            "score_bins", [0, 300, 400, 500, 550, 600, 650, 700, 750, 800, 850, 1000]
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
        train_mask = e["sample_type"] == "train"
        test_mask = e["sample_type"] == "test"
        trainX = X[train_mask]
        trainy = y[train_mask]
        testX = X[test_mask]
        testy = y[test_mask]

        preprocess_pipe = Preprocess().build_pipeline(
            dict([(i, "auto") for i in X.columns])
        )
        transtestX = (
            preprocess_pipe.named_steps["WOE"].fit(trainX, trainy).transform(testX)
        )
        kwargs = {}
        fs_params = {
            "Gini": {"refX": transtestX, "refy": testy},
            "PSI": {"refX": transtestX, "refy": testy},
            "Boosting": {"refX": transtestX, "refy": testy},
        }

        for fs_name, params in fs_params.items():
            if fs_name not in self.ignore_preprocessors:
                for param_name, param_value in params.items():
                    kwargs[f"FS__{fs_name}__{param_name}"] = param_value
        preprocess_pipe = preprocess_pipe.fit(trainX, trainy, **kwargs)
        train_set = preprocess_pipe.transform(trainX)
        test_set = preprocess_pipe.transform(testX)

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
            score_type=self.score_type,
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
            "preprocessor": preprocess_pipe,
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
            if sample_type in self.labeled_sample_type:
                preprocess_pipe["WOE"].transform(X_, y_)
                woe_dfs[sample_type] = preprocess_pipe["WOE"].woe_df

            bm_proba = preprocess_pipe["Benchmark"].predict_proba(transX)[:, 1]
            proba = model.predict_proba(transX)[:, 1]
            score = calibrator.transform(proba)

            if sample_type in self.labeled_sample_type:
                scorecards[sample_type] = calibrator.compare_calibrate_result(score, y_)

            e_.loc[:, "bm_proba"] = [round(i, 6) for i in bm_proba]
            e_.loc[:, "proba"] = [round(i, 6) for i in proba]
            e_.loc[:, "score"] = [int(i) for i in score]
            e_.loc[:, "score_bin"] = pd.cut(e_["score"], bins=self.score_bins)

            score_dist = (
                e_.groupby(["score_bin"])
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
                        * np.log(x / df_distribution[("train", "% Score")])
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

        performance = {
            "version": self.version,
            "model_id": self.ts,
            "missing_values": self.missing_values,
            "label": label,
            "feature_selection": preprocess_pipe["FS"],
            "tune_results": tuner.results,
            "calibrate_detail": calibrator.calibrate_detail,
            "scoredist": {"Distribution": df_distribution, "PSI": df_psi},
            "scorecard": scorecards,
            "woe_df": woe_dfs,
            "sample_set": sample_set,
        }

        performance_path = os.path.join(model_dir, "performance")
        joblib.dump(performance, performance_path)
        cprint(f"Performance file successfully dumped in path {model_dir}", "green")

        if self.report_path:
            Reporter(f"{self.report_path}").generate_report(performance)
            cprint(
                f"Model report successfully dumped in {self.report_path}/model_report_{self.ts}.xlsx",
                "green",
            )
        else:
            cprint("Report path not specified, skipping report generation", "yellow")

        return performance
