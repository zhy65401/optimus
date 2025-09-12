from enum import Enum
from typing import Any, Dict, List, Optional, Union

from sklearn.pipeline import Pipeline

from .encoder import Encoder
from .estimator import Benchmark
from .feature_selection import (
    BoostingTreeSelector,
    CorrSelector,
    GINISelector,
    IVSelector,
    ManualSelector,
    PSISelector,
    VIFSelector,
)
from .tuner import BO, GridSearch


class _DefaultParams(Enum):
    """Default parameters for preprocessing pipeline components."""

    missing_values = [-999999, -999998, -990000, "__N.A.__"]
    treat_missing = "mean"
    corr_threshold = 0.95
    psi_threshold = 0.1
    iv_threshold = 0.02
    vif_threshold = 10
    select_frac = 0.9


class Preprocess:
    """
    Preprocessing pipeline builder for feature engineering and selection in risk modeling.

    This class creates a comprehensive preprocessing pipeline that includes:
    - WOE (Weight of Evidence) encoding for categorical and numerical features
    - Multiple feature selection methods (correlation, IV, PSI, VIF, GINI, boosting)
    - Flexible pipeline configuration with customizable parameters

    The preprocessing pipeline follows this sequence:
    1. WOE Encoding: Transform features using Weight of Evidence
    2. IV Selection: Remove features with low Information Value
    3. PSI Selection: Remove features with high Population Stability Index
    4. GINI Selection: Remove features with wrong GINI sign
    5. Correlation Selection: Remove highly correlated features
    6. VIF Selection: Remove features with high multicollinearity
    7. Boosting Selection: Select top features using tree-based importance

    Attributes:
        corr_threshold (float): Correlation threshold for feature removal (default: 0.95)
        psi_threshold (float): PSI threshold for feature stability (default: 0.1)
        iv_threshold (float): Information Value threshold for feature importance (default: 0.02)
        vif_threshold (float): Variance Inflation Factor threshold for multicollinearity (default: 10)
        select_frac (float): Fraction of features to select in boosting tree selector (default: 0.9)
        missing_values (list): List of values to treat as missing
        treat_missing (str): Strategy for handling missing values ('mean', 'min', 'max', 'zero')
        ignore_preprocessors (list): List of preprocessor names to skip during pipeline building
        drop_features (list): List of feature names to manually drop before preprocessing

    Examples:
        >>> # Basic usage
        >>> preprocessor = Preprocess()
        >>> pipeline = preprocessor.build_pipeline(feature_spec)

        >>> # Custom configuration
        >>> preprocessor = Preprocess(
        ...     corr_threshold=0.9,
        ...     iv_threshold=0.03,
        ...     ignore_preprocessors=['VIF'],
        ...     drop_features=['id', 'timestamp']
        ... )
        >>> pipeline = preprocessor.build_pipeline(feature_spec)
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize preprocessing pipeline with custom parameters.

        Args:
            **kwargs: Keyword arguments for preprocessing parameters:
                corr_threshold (float): Correlation threshold for removing highly correlated features.
                    Features with correlation above this threshold will be removed. Default: 0.95
                psi_threshold (float): Population Stability Index threshold for feature stability.
                    Features with PSI above this threshold indicate distribution drift. Default: 0.1
                iv_threshold (float): Information Value threshold for feature importance.
                    Features with IV below this threshold have weak predictive power. Default: 0.02
                vif_threshold (float): Variance Inflation Factor threshold for multicollinearity.
                    Features with VIF above this threshold indicate multicollinearity. Default: 10
                select_frac (float): Fraction of features to select in boosting tree selector.
                    Must be between 0 and 1. Default: 0.9
                missing_values (list): List of values to treat as missing data.
                    Default: [-999999, -999998, -990000, "__N.A.__"]
                treat_missing (str): Strategy for handling missing values.
                    Options: 'mean', 'min', 'max', 'zero'. Default: 'mean'
                ignore_preprocessors (list): List of preprocessor names to skip during pipeline building.
                    Options: ['IV', 'PSI', 'GINI', 'Corr', 'VIF', 'Boosting']. Default: []
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
        self.missing_values = kwargs.get(
            "missing_values", _DefaultParams.missing_values.value
        )
        self.treat_missing = kwargs.get(
            "treat_missing", _DefaultParams.treat_missing.value
        )
        self.ignore_preprocessors = kwargs.get("ignore_preprocessors", [])
        self.drop_features = kwargs.get("drop_features", [])

    def build_pipeline(self, spec: Dict[str, Union[str, List[float]]]) -> Pipeline:
        """
        Build a complete preprocessing pipeline for risk modeling features.

        Creates a scikit-learn Pipeline with comprehensive feature engineering and selection:

        Pipeline Structure:
        1. **WOE Encoder**: Transforms features using Weight of Evidence encoding
        2. **Feature Selection**: Applies multiple selection methods sequentially:
           - Manual feature dropping (if specified)
           - IV filtering (removes low-importance features)
           - PSI filtering (removes unstable features)
           - GINI filtering (removes features with wrong signs)
           - Correlation filtering (removes highly correlated features)
           - VIF filtering (removes multicollinear features)
           - Boosting selection (selects top features by importance)
        3. **Benchmark Model**: Integrates a logistic regression benchmark

        Args:
            spec: Feature specification dictionary mapping feature names to binning strategies.
                Keys: feature names (str)
                Values: binning strategy options:
                    - 'auto': Automatically choose best strategy based on feature type
                    - 'qcut': Equal frequency binning
                    - 'chiMerge': Chi-square based binning
                    - 'bestKS': KS statistic based optimal binning
                    - 'woeMerge': WOE based categorical merging
                    - 'optimal': OptimalBinning algorithm
                    - List[float]: Custom bin boundaries for numerical features
                    - False: No binning (keep original categories)

        Returns:
            Pipeline: Complete preprocessing pipeline with named steps:
                - 'WOE': Weight of Evidence encoder
                - 'FS': Feature selection pipeline with configurable selectors
                - 'Benchmark': Benchmark logistic regression model

        Raises:
            ValueError: If spec is empty or contains invalid binning strategies

        Examples:
            >>> # Basic feature specification
            >>> spec = {
            ...     'age': 'bestKS',
            ...     'income': 'chiMerge',
            ...     'education': 'woeMerge',
            ...     'employment_length': 'optimal'
            ... }
            >>> preprocessor = Preprocess()
            >>> pipeline = preprocessor.build_pipeline(spec)

            >>> # Custom binning with specific thresholds
            >>> spec = {
            ...     'age': [18, 25, 35, 45, 55, 65, 100],
            ...     'score': 'bestKS',
            ...     'category': False  # No binning
            ... }
            >>> preprocessor = Preprocess(
            ...     corr_threshold=0.9,
            ...     ignore_preprocessors=['VIF']
            ... )
            >>> pipeline = preprocessor.build_pipeline(spec)

            >>> # Using the pipeline
            >>> pipeline.fit(X_train, y_train)
            >>> X_transformed = pipeline.transform(X_test)
        """
        fs_func_bank = {
            "Original": ManualSelector(drop_features=[]),
            "Manual": ManualSelector(drop_features=self.drop_features),
            "Corr": CorrSelector(corr_threshold=self.corr_threshold),
            "Gini": GINISelector(),
            "PSI": PSISelector(psi_threshold=self.psi_threshold),
            "IV": IVSelector(iv_threshold=self.iv_threshold),
            "VIF": VIFSelector(vif_threshold=self.vif_threshold),
            "Boosting": BoostingTreeSelector(select_frac=self.select_frac),
        }
        feature_selection_pipe = Pipeline(
            [
                (name, fs_func)
                for name, fs_func in fs_func_bank.items()
                if name not in self.ignore_preprocessors
            ]
        )

        return Pipeline(
            [
                (
                    "WOE",
                    Encoder(
                        spec=spec,
                        treat_missing="mean",
                        missing_values=self.missing_values,
                    ),
                ),
                ("FS", feature_selection_pipe),
                (
                    "Benchmark",
                    Benchmark(
                        positive_coef=False, remove_method="iv", pvalue_threshold=0.05
                    ),
                ),
            ]
        )


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
