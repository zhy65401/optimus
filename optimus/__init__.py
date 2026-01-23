"""
Optimus: An end-to-end machine learning toolkit for financial risk modeling and credit scoring.

This package provides comprehensive tools for:
- Missing value imputation with feature-level control
- Feature engineering and WOE encoding
- Multi-dimensional feature selection
- Model training and hyperparameter tuning
- Model calibration and scoring
- Automated report generation
- Imbalanced data handling
"""

# Binning classes
from .binner import BestKSCut, ChiMergeCut, OptimalCut, QCut, SimpleCut, WOEMerge
from .calibrator import IsotonicCalibrator, PlattCalibrator
from .encoder import Encoder
from .estimator import Benchmark

# Feature selection classes
from .feature_selection import (
    BoostingTreeSelector,
    CorrSelector,
    GINISelector,
    IVSelector,
    PSISelector,
    StabilitySelector,
    VIFSelector,
)
from .imputer import Imputer

# Metrics and tuning
from .metrics import Metrics
from .pipeliner import Model, Preprocess
from .reporter import Reporter

# Advanced features
from .sampler import ImbalanceSampler
from .trainer import Train
from .tuner import BO, GridSearch

__version__ = "0.4.1"
__author__ = "Hanyuan Zhang"
__email__ = "klesterchueng@gmail.com"

__all__ = [
    # Main classes
    "Train",
    "Encoder",
    "Imputer",
    "IsotonicCalibrator",
    "PlattCalibrator",
    "Preprocess",
    "Model",
    "Benchmark",
    "Reporter",
    # Feature selection
    "CorrSelector",
    "IVSelector",
    "PSISelector",
    "GINISelector",
    "VIFSelector",
    "BoostingTreeSelector",
    "StabilitySelector",
    # Binning
    "QCut",
    "SimpleCut",
    "ChiMergeCut",
    "BestKSCut",
    "OptimalCut",
    "WOEMerge",
    # Utilities
    "Metrics",
    "GridSearch",
    "BO",
    # Advanced
    "ImbalanceSampler",
]
