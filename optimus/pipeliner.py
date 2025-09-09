from enum import Enum

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
    - Benchmark model integration

    Attributes:
        corr_threshold (float): Correlation threshold for feature removal
        psi_threshold (float): PSI threshold for feature stability
        iv_threshold (float): Information Value threshold for feature importance
        vif_threshold (float): Variance Inflation Factor threshold for multicollinearity
        select_frac (float): Fraction of features to select in boosting tree selector
        missing_values (list): List of values to treat as missing
        treat_missing (str): Strategy for handling missing values
        ignore_preprocessors (list): List of preprocessor names to skip
        drop_features (list): List of feature names to manually drop
    """

    def __init__(self, **kwargs):
        """
        Initialize preprocessing pipeline with custom parameters.

        Args:
            **kwargs: Keyword arguments for preprocessing parameters:
                - corr_threshold (float): Correlation threshold. Default: 0.95
                - psi_threshold (float): PSI threshold. Default: 0.1
                - iv_threshold (float): IV threshold. Default: 0.02
                - vif_threshold (float): VIF threshold. Default: 10
                - select_frac (float): Boosting selection fraction. Default: 0.9
                - missing_values (list): Missing value indicators. Default: predefined list
                - treat_missing (str): Missing value treatment. Default: 'mean'
                - ignore_preprocessors (list): Preprocessors to skip. Default: []
                - drop_features (list): Features to manually drop. Default: []
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

    def build_pipeline(self, spec):
        """
        Build a complete preprocessing pipeline for risk modeling features.

        Creates a scikit-learn Pipeline with three main components:
        1. WOE Encoder: Transforms features using Weight of Evidence encoding
        2. Feature Selection: Applies multiple feature selection methods sequentially
        3. Benchmark Model: Integrates a benchmark logistic regression model

        Args:
            spec (dict): Feature specification dictionary mapping feature names to binning strategies.
                        Keys: feature names, Values: binning strategy ('auto', 'qcut', 'chiMerge',
                        'bestKS', 'woeMerge', 'optimal') or list of bin boundaries.

        Returns:
            sklearn.pipeline.Pipeline: Complete preprocessing pipeline with named steps:
                - 'WOE': Weight of Evidence encoder
                - 'FS': Feature selection pipeline with multiple selectors
                - 'Benchmark': Benchmark logistic regression model

        Example:
            >>> preprocessor = Preprocess(corr_threshold=0.9, iv_threshold=0.03)
            >>> spec = {'age': 'bestKS', 'income': 'chiMerge', 'category': 'woeMerge'}
            >>> pipeline = preprocessor.build_pipeline(spec)
            >>> pipeline.fit(X_train, y_train)
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
    machine learning models with various hyperparameter tuning strategies.

    Attributes:
        model_type (str): Type of machine learning model
        tunner: Hyperparameter tuning object (GridSearch or BO)
    """

    def __init__(
        self, model_type="LGBM", tune_method="BO", max_evals=500, param_grid=None
    ):
        """
        Initialize model builder with specified algorithm and tuning method.

        Args:
            model_type (str): Machine learning algorithm type. Options: 'LR', 'XGB', 'LGBM', 'BM'. Default: 'LGBM'.
            tune_method (str): Hyperparameter tuning method. Options: 'BO' (Bayesian Optimization), 'GS' (Grid Search). Default: 'BO'.
            max_evals (int): Maximum number of evaluations for Bayesian Optimization. Default: 500.
            param_grid (dict, optional): Custom parameter grid for tuning. If None, uses default grids.

        Raises:
            ValueError: If unsupported tune_method is specified.
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

    def build_model(self, user_model=None):
        """
        Build and return the configured machine learning model.

        Creates the appropriate model instance based on the specified model_type:
        - For 'LR', 'XGB', 'LGBM': Returns tuning object (GridSearch or BO)
        - For 'BM': Returns Benchmark logistic regression model
        - For custom models: Returns user-provided model

        Args:
            user_model (optional): Custom model instance to use instead of built-in models.

        Returns:
            Model instance ready for training:
                - GridSearch or BO tuner for 'LR', 'XGB', 'LGBM'
                - Benchmark model for 'BM'
                - User-provided model if specified

        Example:
            >>> model_builder = Model(model_type='XGB', tune_method='BO', max_evals=100)
            >>> model = model_builder.build_model()
            >>> tuned_model = model.fit(X_train, y_train, X_val, y_val)
        """
        if user_model:
            return user_model
        if self.model_type == "BM":
            return Benchmark(
                positive_coef=False, remove_method="iv", pvalue_threshold=0.05
            )
        return self.tunner
