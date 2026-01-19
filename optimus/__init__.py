"""
Optimus: An end-to-end machine learning toolkit for financial risk modeling and credit scoring.

This package provides comprehensive tools for:
- Missing value imputation with feature-level control
- Feature engineering and WOE encoding
- Multi-dimensional feature selection
- Model training and hyperparameter tuning
- Model calibration and scoring
- Automated report generation
"""

# Binning classes
from .binner import BestKSCut, ChiMergeCut, OptimalCut, QCut, SimpleCut, WOEMerge
from .calibrator import Calibration
from .encoder import Encoder
from .estimator import Benchmark

# Feature selection classes
from .feature_selection import (
    BoostingTreeSelector,
    CorrSelector,
    GINISelector,
    IVSelector,
    PSISelector,
    VIFSelector,
)
from .imputer import Imputer

# Metrics and tuning
from .metrics import Metrics
from .pipeliner import Model, Preprocess
from .reporter import Reporter
from .trainer import Train
from .tuner import BO, GridSearch

__version__ = "0.4.0"
__author__ = "Hanyuan Zhang"
__email__ = "klesterchueng@gmail.com"

__all__ = [
    # Main classes
    "Train",
    "Encoder",
    "Imputer",
    "Calibration",
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
]
