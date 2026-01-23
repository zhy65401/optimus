from enum import Enum
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from sklearn.base import clone
from termcolor import cprint

from .encoder import Encoder
from .estimator import Benchmark
from .feature_selection import (
    BoostingTreeSelector,
    CorrSelector,
    GINISelector,
    IVSelector,
    ManualSelector,
    PSISelector,
    StabilitySelector,
    VIFSelector,
)
from .imputer import Imputer
from .tuner import BO, GridSearch

# Transformers that require reference dataset for fitting
_REF_AWARE_TRANSFORMERS = frozenset(
    ["PSISelector", "GINISelector", "BoostingTreeSelector"]
)


class _DefaultParams(Enum):
    """Default parameters for preprocessing pipeline components."""

    missing_values = [-999999, -999998, -990000, "__N.A.__"]
    treat_missing = "mean"
    corr_threshold = 0.95
    psi_threshold = 0.1
    iv_threshold = 0.02
    vif_threshold = 10
    boosting_select_frac = 0.95
    stability_threshold = 0.1


class Preprocess:
    """
    Preprocessing pipeline builder for feature engineering and selection in risk modeling.

    This class creates a comprehensive preprocessing pipeline that includes:
    - Missing value imputation with feature-level strategy control (optional)
    - WOE (Weight of Evidence) encoding for categorical and numerical features
    - Multiple feature selection methods (correlation, IV, PSI, VIF, GINI, boosting)
    - Flexible pipeline configuration with customizable parameters

    Supports reference dataset (e.g., test set) for PSI/GINI/Boosting selectors
    via fit(X_train, y_train, X_ref=X_test, y_ref=y_test).

    The preprocessing pipeline follows this sequence:
    0. Impute (optional): Fill missing values using statistical measures
    1. WOE Encoding: Transform features using Weight of Evidence
    2. IV Selection: Remove features with low Information Value
    3. PSI Selection: Remove features with high Population Stability Index
    4. GINI Selection: Remove features with wrong GINI sign
    5. Correlation Selection: Remove highly correlated features
    6. VIF Selection: Remove features with high multicollinearity
    7. Boosting Selection: Select top features using tree-based importance
    8. Stability Selection: Select stable features across subsamples

    Attributes:
        impute_strategy (dict): Feature-level imputation strategies (default: None)
        corr_threshold (float): Correlation threshold for feature removal (default: 0.95)
        psi_threshold (float): PSI threshold for feature stability (default: 0.1)
        iv_threshold (float): Information Value threshold for feature importance (default: 0.02)
        vif_threshold (float): Variance Inflation Factor threshold for multicollinearity (default: 10)
        boosting_select_frac (float): Fraction of features to select in boosting tree selector (default: 0.9)
        stability_threshold (float): Stability selection threshold (default: 0.6)
        missing_values (list): List of values to treat as missing
        treat_missing (str): Strategy for handling missing values ('mean', 'min', 'max', 'zero')
        ignore_preprocessors (list): List of preprocessor names to skip during pipeline building
        drop_features (list): List of feature names to manually drop before preprocessing
    """

    def __init__(
        self, spec: Optional[Dict[str, Union[str, List[float]]]] = None, **kwargs: Any
    ) -> None:
        """
        Initialize preprocessing pipeline with custom parameters.

        Args:
            spec: Feature specification dict mapping feature names to binning strategies.
                Can also be set later via fit(). Options:
                - 'auto': Automatically choose best strategy based on feature type
                - 'qcut': Equal frequency binning
                - 'chiMerge': Chi-square based binning
                - 'bestKS': KS statistic based optimal binning
                - 'woeMerge': WOE based categorical merging
                - 'optimal': OptimalBinning algorithm
                - List[float]: Custom bin boundaries for numerical features
                - False: No binning (keep original categories)
            **kwargs: Keyword arguments for preprocessing parameters:
                impute_strategy (dict): Dictionary mapping feature names to imputation strategies.
                    Options: 'mean', 'median', 'min', 'max', 'mode', 'separate'. Default: None
                corr_threshold (float): Correlation threshold for removing highly correlated features.
                    Features with correlation above this threshold will be removed. Default: 0.95
                psi_threshold (float): Population Stability Index threshold for feature stability.
                    Features with PSI above this threshold indicate distribution drift. Default: 0.1
                iv_threshold (float): Information Value threshold for feature importance.
                    Features with IV below this threshold have weak predictive power. Default: 0.02
                vif_threshold (float): Variance Inflation Factor threshold for multicollinearity.
                    Features with VIF above this threshold indicate multicollinearity. Default: 10
                boosting_select_frac (float): Fraction of features to select in boosting tree selector.
                    Must be between 0 and 1. Default: 0.9
                stability_threshold (float): Stability selection threshold for feature stability.
                    Features with stability below this threshold will be removed. Default: 0.6
                missing_values (list): List of values to treat as missing data.
                    Default: [-999999, -999998, -990000, "__N.A.__"]
                treat_missing (str): Strategy for handling missing values in WOE encoding.
                    Options: 'mean', 'min', 'max', 'zero'. Default: 'mean'
                ignore_preprocessors (list): List of preprocessor names to skip during pipeline building.
                    Options: ['Impute', 'WOE', 'Original', 'Manual', 'IV', 'PSI', 'Gini', 'Corr', 'VIF', 'Boosting', 'Benchmark']. Default: []
                drop_features (list): List of feature names to manually drop before preprocessing.
                    These features will be excluded from the entire pipeline. Default: []

        Examples:
            >>> # Default configuration
            >>> preprocessor = Preprocess()

            >>> # Custom thresholds
            >>> preprocessor = Preprocess(
            ...     corr_threshold=0.9,
            ...     iv_threshold=0.03,
            ...     vif_threshold=5
            ... )

            >>> # Skip certain preprocessors
            >>> preprocessor = Preprocess(
            ...     ignore_preprocessors=['VIF', 'Boosting'],
            ...     drop_features=['id', 'timestamp', 'created_at']
            ... )
        """
        self.spec = spec
        self.impute_strategy = kwargs.get("impute_strategy", None)
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
        self.missing_values = kwargs.get(
            "missing_values", _DefaultParams.missing_values.value
        )
        self.treat_missing = kwargs.get(
            "treat_missing", _DefaultParams.treat_missing.value
        )
        self.ignore_preprocessors = kwargs.get("ignore_preprocessors", [])
        self.drop_features = kwargs.get("drop_features", [])
        self.boosting_select_frac = kwargs.get(
            "boosting_select_frac", _DefaultParams.boosting_select_frac.value
        )
        self.stability_threshold = kwargs.get(
            "stability_threshold", _DefaultParams.stability_threshold.value
        )

        # Fitted state
        self._steps: List[tuple] = []
        self._fitted_steps: List[tuple] = []

    def _build_steps(self, spec: Dict[str, Union[str, List[float]]]) -> List[tuple]:
        """Build pipeline steps from spec."""
        # Safety check: if WOE is ignored, impute_strategy must be provided
        if "WOE" in self.ignore_preprocessors and not self.impute_strategy:
            raise ValueError(
                "When 'WOE' is in ignore_preprocessors, impute_strategy must be provided. "
                "WOE encoder handles missing values, but if skipped, Imputer must replace it."
            )

        all_steps = []
        if self.impute_strategy:
            all_steps.append(
                (
                    "Impute",
                    Imputer(
                        impute_strategy=self.impute_strategy,
                        missing_values=self.missing_values,
                    ),
                )
            )
        all_steps.append(
            (
                "WOE",
                Encoder(
                    spec=spec,
                    treat_missing=self.treat_missing,
                    missing_values=self.missing_values,
                ),
            )
        )
        all_steps.extend(
            [
                ("Original", ManualSelector(drop_features=[])),
                ("Manual", ManualSelector(drop_features=self.drop_features)),
                ("IV", IVSelector(iv_threshold=self.iv_threshold)),
                ("PSI", PSISelector(psi_threshold=self.psi_threshold)),
                ("Gini", GINISelector()),
                ("Corr", CorrSelector(corr_threshold=self.corr_threshold)),
                ("VIF", VIFSelector(vif_threshold=self.vif_threshold)),
                (
                    "Boosting",
                    BoostingTreeSelector(select_frac=self.boosting_select_frac),
                ),
                ("Stability", StabilitySelector(threshold=self.stability_threshold)),
            ]
        )
        # Note: Benchmark is no longer part of Preprocess pipeline
        # It's now handled separately in Train class
        steps = [
            (name, transformer)
            for name, transformer in all_steps
            if name not in self.ignore_preprocessors
        ]

        return steps

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_ref: Optional[pd.DataFrame] = None,
        y_ref: Optional[pd.Series] = None,
        spec: Optional[Dict[str, Union[str, List[float]]]] = None,
    ) -> "Preprocess":
        """
        Fit the preprocessing pipeline.

        Args:
            X: Training feature matrix
            y: Training target variable
            X_ref: Reference feature matrix (e.g., test set) for PSI/GINI/Boosting.
                If None, X will be used as reference (not recommended).
            y_ref: Reference target variable
            spec: Feature specification (overrides __init__ spec if provided)

        Returns:
            self: Fitted preprocessor
        """

        spec = spec or self.spec
        if spec is None:
            raise ValueError("spec must be provided either in __init__ or fit()")

        self._steps = self._build_steps(spec)
        self._fitted_steps = []

        Xt = X.copy()
        Xt_ref = X_ref.copy() if X_ref is not None else None

        for idx, (name, transformer) in enumerate(self._steps):
            fitted_transformer = clone(transformer)
            is_last = idx == len(self._steps) - 1

            # Skip feature selection if no features remain
            if Xt.shape[1] == 0:
                cprint(
                    f"[WARN] No features remaining, skipping {name} and subsequent steps.",
                    "yellow",
                )
                # Store a passthrough transformer that keeps empty state
                fitted_transformer.selected_features = []
                fitted_transformer.removed_features = []
                self._fitted_steps.append((name, fitted_transformer))
                continue

            if transformer.__class__.__name__ in _REF_AWARE_TRANSFORMERS:
                fitted_transformer.fit(
                    Xt,
                    y,
                    refX=Xt_ref if Xt_ref is not None else Xt,
                    refy=y_ref if y_ref is not None else y,
                )
            else:
                fitted_transformer.fit(Xt, y)

            self._fitted_steps.append((name, fitted_transformer))

            # Transform for next step (unless last)
            if not is_last and hasattr(fitted_transformer, "transform"):
                Xt = fitted_transformer.transform(Xt)
                if Xt_ref is not None:
                    Xt_ref = fitted_transformer.transform(Xt_ref)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted pipeline.

        Args:
            X: Feature matrix to transform

        Returns:
            Transformed feature matrix
        """
        if not self._fitted_steps:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")

        Xt = X.copy()
        for name, transformer in self._fitted_steps:
            if hasattr(transformer, "transform"):
                Xt = transformer.transform(Xt)
        return Xt

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_ref: Optional[pd.DataFrame] = None,
        y_ref: Optional[pd.Series] = None,
        spec: Optional[Dict[str, Union[str, List[float]]]] = None,
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(X, y, X_ref=X_ref, y_ref=y_ref, spec=spec)
        return self.transform(X)

    def get_step(self, name: str) -> Any:
        """Get a fitted step by name."""
        for step_name, transformer in self._fitted_steps:
            if step_name == name:
                return transformer
        raise KeyError(f"Step '{name}' not found")

    @property
    def named_steps(self) -> Dict[str, Any]:
        """Get fitted steps as a dictionary."""
        return {name: t for name, t in self._fitted_steps}


class Model:
    """
    Model builder for machine learning algorithms with hyperparameter tuning.

    This class provides a unified interface for creating and configuring different
    machine learning models with automated hyperparameter optimization. It supports
    both Bayesian Optimization and Grid Search for finding optimal parameters.

    Supported Models:
    - **LR**: Logistic Regression with regularization tuning
    - **XGB**: XGBoost with tree-based parameters
    - **LGBM**: LightGBM with gradient boosting parameters
    - **BM**: Benchmark model (simple logistic regression)

    Tuning Methods:
    - **BO**: Bayesian Optimization for efficient parameter search
    - **GS**: Grid Search for exhaustive parameter exploration

    Attributes:
        model_type (str): Type of machine learning model
        tunner: Hyperparameter tuning object (BO or GridSearch instance)

    Examples:
        >>> # LightGBM with Bayesian Optimization
        >>> model_builder = Model(
        ...     model_type='LGBM',
        ...     tune_method='BO',
        ...     max_evals=100
        ... )
        >>> tuner = model_builder.build_model()

        >>> # XGBoost with custom parameter grid
        >>> custom_params = {
        ...     'n_estimators': [100, 200, 300],
        ...     'max_depth': [3, 5, 7],
        ...     'learning_rate': [0.01, 0.1, 0.2]
        ... }
        >>> model_builder = Model(
        ...     model_type='XGB',
        ...     tune_method='GS',
        ...     param_grid=custom_params
        ... )
    """

    def __init__(
        self,
        model_type: str = "LGBM",
        tune_method: str = "BO",
        max_evals: int = 500,
        param_grid: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize model builder with specified algorithm and tuning method.

        Args:
            model_type: Machine learning algorithm type. Options:
                - 'LR': Logistic Regression
                - 'XGB': XGBoost Classifier
                - 'LGBM': LightGBM Classifier (default)
                - 'BM': Benchmark Model (simple logistic regression)
            tune_method: Hyperparameter tuning method. Options:
                - 'BO': Bayesian Optimization (default, recommended for efficiency)
                - 'GS': Grid Search (exhaustive but slower)
            max_evals: Maximum number of evaluations for Bayesian Optimization.
                Higher values allow more thorough search but take longer. Default: 500
            param_grid: Custom parameter grid for tuning. If None, uses predefined
                optimal parameter spaces for each model type.

        Raises:
            ValueError: If unsupported tune_method is specified.

        Examples:
            >>> # Default LightGBM with Bayesian Optimization
            >>> model = Model()

            >>> # XGBoost with Grid Search and custom parameters
            >>> custom_grid = {
            ...     'n_estimators': [50, 100, 200],
            ...     'max_depth': [3, 5, 7]
            ... }
            >>> model = Model(
            ...     model_type='XGB',
            ...     tune_method='GS',
            ...     param_grid=custom_grid
            ... )

            >>> # Quick Bayesian Optimization with fewer evaluations
            >>> model = Model(
            ...     model_type='LGBM',
            ...     tune_method='BO',
            ...     max_evals=50
            ... )
        """
        self.model_type = model_type
        if model_type in ["LR", "XGB", "LGBM"]:
            if tune_method == "BO":
                self.tunner = BO(
                    self.model_type, max_evals=max_evals, param_grid=param_grid
                )
            elif tune_method == "GS":
                self.tunner = GridSearch(self.model_type, param_grid=param_grid)
            else:
                raise ValueError(f"Unsupported tunner: {tune_method}")

    def build_model(self, user_model: Optional[Any] = None) -> Any:
        """
        Build and return the configured machine learning model or tuner.

        This method creates the appropriate model instance based on the specified model_type.
        For tunable models (LR, XGB, LGBM), it returns a hyperparameter tuner that can be
        used to find optimal parameters. For benchmark models, it returns a ready-to-use model.

        Model Creation Logic:
        - **'LR', 'XGB', 'LGBM'**: Returns tuning object (GridSearch or BO) configured
          with the appropriate parameter search space
        - **'BM'**: Returns Benchmark logistic regression model with predefined settings
        - **Custom Model**: Returns user-provided model instance

        Args:
            user_model: Custom model instance to use instead of built-in models.
                Can be any scikit-learn compatible estimator or custom model class.

        Returns:
            Model instance ready for training:
                - **GridSearch or BO tuner**: For 'LR', 'XGB', 'LGBM' models.
                  Use tuner.fit(X_train, y_train, X_val, y_val) to get optimized model.
                - **Benchmark model**: For 'BM' type. Ready for direct fit/predict.
                - **User model**: Custom model instance if provided.

        Examples:
            >>> # Build XGBoost tuner with Bayesian Optimization
            >>> model_builder = Model(model_type='XGB', tune_method='BO', max_evals=100)
            >>> tuner = model_builder.build_model()
            >>> best_model = tuner.fit(X_train, y_train, X_val, y_val)

            >>> # Build benchmark model
            >>> model_builder = Model(model_type='BM')
            >>> benchmark = model_builder.build_model()
            >>> benchmark.fit(X_train, y_train)

            >>> # Use custom model
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> custom_rf = RandomForestClassifier(n_estimators=100)
            >>> model_builder = Model()
            >>> model = model_builder.build_model(user_model=custom_rf)
        """
        if user_model:
            return user_model
        if self.model_type == "BM":
            return Benchmark(
                positive_coef=False, remove_method="iv", pvalue_threshold=0.05
            )
        return self.tunner
