# Optimus

An end-to-end machine learning toolkit designed specifically for financial risk modeling and credit scoring.

## Introduction

Optimus is a powerful machine learning toolkit specifically designed for the development needs of financial risk control and credit scoring models. It provides a complete workflow from data preprocessing to model deployment, including feature engineering, model training, hyperparameter tuning, model calibration, and report generation.

### Key Advantages

- **Comprehensive Training Pipeline**: End-to-end `Train` class for streamlined model development
- **Intelligent Missing Value Handling**: Feature-level `Imputer` class with multiple imputation strategies (mean, median, mode, etc.)
- **Professional Binning Strategies**: Support for multiple intelligent binning algorithms (ChiMerge, BestKS, WOEMerge, OptimalBinning, etc.)
- **Smart Feature Selection**: Statistical metric-based feature filtering (IV, KS, Gini, PSI, VIF, correlation)
- **Multi-Model Support**: Integration of mainstream algorithms like Logistic Regression, XGBoost, LightGBM
- **Automated Hyperparameter Tuning**: Support for Grid Search and Bayesian Optimization
- **Model Calibration**: Scorecard mapping and probability calibration functionality
- **Professional Reports**: Automated generation of detailed modeling reports and visualizations
- **Robust Architecture**: Enhanced error handling, input validation, and comprehensive documentation

## Quick Start

### Installation

```bash
# Install from GitHub (recommended)
pip install git+https://github.com/zhy65401/optimus.git

# Install specific version/branch
pip install git+https://github.com/zhy65401/optimus.git@v0.3.1

# Install from source for development
git clone https://github.com/zhy65401/optimus.git
cd optimus
pip install -e .
```

### Basic Usage

```python
import pandas as pd
from optimus.trainer import Train

# Quick start with end-to-end pipeline
trainer = Train(
    model_path='./models',
    model_type='LR',           # Logistic Regression
    tune_method='BO'           # Bayesian Optimization
)

# Fit model (X: features, y: target, e: external data with sample_type)
trainer.fit(X, y, e)

# Generate predictions and reports
performance = trainer.transform(X, y, e)
```

**Traditional Component-Based Approach:**

```python
import pandas as pd
from optimus.imputer import Imputer
from optimus.encoder import Encoder
from optimus.feature_selection import IVSelector, CorrSelector
from optimus.estimator import Benchmark
from optimus.calibrator import Calibration
from sklearn.pipeline import Pipeline

# 1. Missing value imputation (NEW in v0.3.1)
imputer = Imputer(
    impute_strategy={
        'age': 'median',
        'income': 'mean',
        'occupation': 'mode'
    },
    missing_values=[-999, '__N.A__']
)

# 2. Data preprocessing and feature encoding
encoder = Encoder(spec={
    'age': 'bestKS',           # Numerical feature using BestKS binning
    'income': 'chiMerge',      # Numerical feature using ChiMerge binning
    'occupation': 'woeMerge',   # Categorical feature using WOE merging
    'education': [1, 2, 3, 4]   # Custom binning boundaries
})

# 3. Feature selection pipeline
feature_selector = Pipeline([
    ('iv', IVSelector(iv_threshold=0.02)),
    ('corr', CorrSelector(corr_threshold=0.95))
])

# 4. Model training
benchmark = Benchmark(positive_coef=False, remove_method="iv")

# 5. Model calibration
calibrator = Calibration(score_type='mega_score')

# Complete workflow example
X_train_imputed = imputer.fit_transform(X_train)
X_train_encoded = encoder.fit_transform(X_train_imputed, y_train)
X_selected = feature_selector.fit_transform(X_train_encoded, y_train)
model = benchmark.fit(X_selected, y_train)
```

## Feature Overview

### Data Preprocessing

#### Missing Value Imputation

The `Imputer` class provides flexible missing value handling with feature-level control:

```python
from optimus.imputer import Imputer
import pandas as pd
import numpy as np

# Sample data with missing values
X = pd.DataFrame({
    'age': [25, np.nan, 45, -999999, 65],
    'income': [30000, 50000, np.nan, 90000, 110000],
    'education': ['high_school', 'bachelor', '__N.A__', 'phd', 'bachelor']
})

# Define imputation strategies for each feature
impute_strategy = {
    'age': 'mean',       # Fill age with mean
    'income': 'median',  # Fill income with median
    'education': 'mode'  # Fill education with most frequent value
}

# Initialize and apply imputer
imputer = Imputer(
    impute_strategy=impute_strategy,
    missing_values=[-999999, '__N.A__']  # Custom missing value markers
)
imputer.fit(X)
X_imputed = imputer.transform(X)

# Available strategies:
# - 'mean': Mean of non-missing values (numerical only)
# - 'median': Median of non-missing values (numerical only)
# - 'min': Minimum of non-missing values (numerical only)
# - 'max': Maximum of non-missing values (numerical only)
# - 'mode': Most frequent value (numerical and categorical)
# - 'separate': Keep missing values as-is for separate category handling
```

#### Smart Binning
- **QCut**: Equal frequency binning for continuous numerical features
- **SimpleCut**: Simple binning for strategy rule formulation
- **ChiMerge**: Chi-square test based binning merging
- **BestKS**: KS value optimization based binning method
- **WOEMerge**: WOE value based categorical feature merging
- **OptimalCut**: Optimal binning based on OptimalBinning library

```python
from optimus.binner import BestKSCut, WOEMerge

# Numerical feature binning
num_binner = BestKSCut(target_bin_cnt=5, min_bin_rate=0.05)
binned_feature = num_binner.fit_transform(X['income'], y)

# Categorical feature merging
cat_binner = WOEMerge(target_bin_cnt=4, min_bin_rate=0.05)
merged_feature = cat_binner.fit_transform(X['occupation'], y)
```

#### WOE Encoding
- Automatic handling of missing values and outliers
- Support for custom missing value handling strategies
- Integrated WOE analysis report generation

```python
from optimus.encoder import Encoder

encoder = Encoder(
    spec={
        'feature1': 'auto',      # Automatically select best binning strategy
        'feature2': 'bestKS',    # Specify binning method
        'feature3': [0, 10, 50, 100]  # Custom binning points
    },
    missing_values=[-999, 'NULL'],  # Custom missing value identifiers
    treat_missing='mean'            # Missing value handling strategy
)

# Transform data and automatically generate WOE report when y is provided
X_encoded = encoder.fit_transform(X, y)

# Access WOE analysis report (automatically generated)
woe_report = encoder.woe_df
```

### Feature Selection

Multi-dimensional feature filtering strategies:

```python
from optimus.feature_selection import (
    IVSelector, PSISelector, GINISelector,
    CorrSelector, VIFSelector, BoostingTreeSelector,
    StabilitySelector  # NEW in v0.3.1
)

# IV value filtering
iv_selector = IVSelector(iv_threshold=0.02)

# PSI stability testing
psi_selector = PSISelector(psi_threshold=0.1)

# GINI sign filtering
gini_selector = GINISelector()

# Correlation filtering
corr_selector = CorrSelector(corr_threshold=0.95, method='iv_descending')

# Multicollinearity testing
vif_selector = VIFSelector(vif_threshold=10)

# Feature Stability testing (tree-based)
boost_selector = BoostingTreeSelector(select_frac=0.9)

# Stability selection (NEW in v0.3.1)
# Identifies features consistently selected across multiple subsamples
stability_selector = StabilitySelector(
    n_iterations=100,      # Number of subsampling iterations
    sample_fraction=0.75,  # Fraction of data to sample
    threshold=0.6,         # Minimum selection frequency (0-1)
    select_frac=0.5,       # Fraction of top features per iteration
    random_state=42
)

# Combined usage
pipeline = Pipeline([
    ('iv', iv_selector),
    ('psi', psi_selector),
    ('gini', gini_selector),
    ('corr', corr_selector),
    ('vif', vif_selector),
    ('boost', boost_selector),
    ('stability', stability_selector),
])
```

### Model Training

#### Supported Algorithms
- **Benchmark**: Benchmark model with a basic logistic regression without any tunning in parameters.
- **Logistic Regression**: With coefficient testing and significance analysis
- **XGBoost**: Gradient boosting tree model
- **LightGBM**: Efficient gradient boosting framework

```python
from optimus.estimator import Benchmark, Estimators

# Benchmark logistic regression model (with built-in feature filtering)
benchmark = Benchmark(
    positive_coef=False,           # Require negative coefficients (risk features)
    remove_method="iv",            # Removal strategy
    pvalue_threshold=0.05          # Significance threshold
)
benchmark.fit(X, y)

# Using other algorithms
lgbm_model = Estimators.LGBM.value
lgbm_model.fit(X, y)
```

#### Hyperparameter Optimization

```python
from hyperopt import hp
from optimus.tuner import GridSearch, BO

# Grid search
gs = GridSearch(
    model_type='LGBM',
    param_grid={
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }
)
gs.fit(X_train, y_train, X_test, y_test)

# Bayesian optimization
bo = BO(
    model_type='LR', 
    max_evals=100,
    param_grid={
        'C': hp.choice("C", np.arange(1, 100, 1)), 
        'penalty': hp.choice("penalty", ['l2'])
    }
)
bo.fit(X_train, y_train, X_test, y_test)
```

### Model Calibration

Multiple score mapping strategies:

```python
from optimus.calibrator import Calibration

# Standard credit score (300-850 points)
calibrator = Calibration(score_type='mega_score')
calibrator.fit(y_prob, y_true)
scores = calibrator.transform(y_prob_test)

# Custom score mapping
calibrator = Calibration(
    score_type='self-defining',
    mapping_base={500: 0.1, 600: 0.05, 700: 0.01},
    score_cap=800,
    score_floor=300
)
```

### Report Generation

Automated professional modeling reports with SHAP analysis:

```python
from optimus.reporter import Reporter

# Using the new v0.4.0 structure
reporter = Reporter('./reports')
reporter.generate_report(performance_data)

# Reports automatically include:
# - Sample overview and distribution analysis
# - Feature statistics and WOE analysis  
# - Feature selection details
# - Model performance metrics
# - Hyperparameter tuning results
# - Model calibration analysis
# - **NEW**: SHAP analysis (feature importance, dependence plots, case studies)
```

Report contents include:
- Sample overview and distribution analysis
- Feature statistics and WOE analysis
- Feature selection details
- Model performance metrics
- Hyperparameter tuning results
- **SHAP analysis** (automatically generated for LR, XGB, LGBM models):
  - Global feature importance with summary plots
  - Feature dependence plots showing non-linear relationships
  - Sample-level explanations with waterfall plots
- Model calibration analysis

## Usage Examples

### End-to-End Training Pipeline (v0.4.0)

```python
import pandas as pd
from optimus.trainer import Train

# 1. Data preparation
df = pd.read_csv('credit_data.csv')
X = df.drop(['target', 'sample_type'], axis=1)
y = df['target']
e = df[['sample_type']]  # Must contain 'train', 'test' sample types

# 2. Initialize training pipeline
trainer = Train(
    model_path='./models',
    report_path='./reports',
    model_type='LR',           # Options: 'LR', 'XGB', 'LGBM'
    tune_method='BO',          # Options: 'BO', 'GS'
    score_type='mega_score',   # Options: 'mega_score', 'sub_score', 'probability'
    max_evals=100,
    corr_threshold=0.95,
    iv_threshold=0.02,
    version="v1.0"
)

# 3. Train the model
trainer.fit(X, y, e)

# 4. Generate predictions and reports
performance = trainer.transform(X, y, e)

# 5. Access results
print("Model Performance:")
print(performance['scorecard']['test'])
print("\nFeature Selection Results:")
print(performance['feature_selection'])
```

### Complete Modeling Workflow (Traditional Approach)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from optimus.imputer import Imputer
from optimus.encoder import Encoder
from optimus.feature_selection import IVSelector, PSISelector, CorrSelector
from optimus.estimator import Benchmark
from optimus.calibrator import Calibration
from optimus.reporter import Reporter

# 1. Data loading
df = pd.read_csv('credit_data.csv')
X = df.drop('target', axis=1)
y = df['target']

# 2. Data splitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. Feature engineering pipeline
preprocessing_pipeline = Pipeline([
    # Missing value imputation (NEW in v0.3.1)
    ('imputer', Imputer(
        impute_strategy={
            'age': 'median',
            'income': 'mean',
            'education': 'mode',
            'employment_length': 'median'
        },
        missing_values=[-999, '__N.A__']
    )),

    # WOE encoding
    ('encoder', Encoder(spec={
        'age': 'bestKS',
        'income': 'chiMerge',
        'education': 'woeMerge',
        'employment_length': 'optimal'
    })),

    # Feature selection
    ('feature_selection', Pipeline([
        ('iv', IVSelector(iv_threshold=0.02)),
        ('psi', PSISelector(psi_threshold=0.1)),
        ('corr', CorrSelector(corr_threshold=0.95))
    ]))
])

# 4. Preprocessing
X_train_processed = preprocessing_pipeline.fit_transform(X_train, y_train)
X_test_processed = preprocessing_pipeline.transform(X_test)

# 5. Model training
model = Benchmark()
model.fit(X_train_processed, y_train)

# 6. Model calibration
y_prob = model.predict_proba(X_test_processed)[:, 1]
calibrator = Calibration(score_type='mega_score')
calibrator.fit(y_prob, y_test)
scores = calibrator.transform(y_prob)

# 7. Report generation
reporter = Reporter('credit_model_report.xlsx')
performance_data = {
    'sample_set': {
        'test': {
            'X': X_test,
            'y': y_test,
            'e': pd.DataFrame({
                'sample_type': ['test'] * len(X_test),
                'proba': y_prob,
                'score': scores
            })
        }
    },
    'model_id': '20241201_143022',
    'label': 'target',
    'missing_values': [-999, 'NULL'],
    'calibrate_detail': calibrator.calibrate_detail,
    'scorecard': {'test': calibrator.compare_calibrate_result(scores, y_test)}
}
reporter.generate_report(performance_data)
```

### Advanced Pipeline Configuration

```python
from optimus.pipeliner import Preprocess, Model

# Custom preprocessing pipeline
preprocessor = Preprocess(
    corr_threshold=0.9,
    psi_threshold=0.15,
    iv_threshold=0.03,
    vif_threshold=5,
    ignore_preprocessors=['VIF'],  # Skip VIF selection
    drop_features=['id', 'timestamp']  # Manual feature removal
)

# Feature specification
feature_spec = {
    'age': 'bestKS',           # Numerical feature using BestKS binning
    'income': 'chiMerge',      # Numerical feature using ChiMerge binning
    'occupation': 'woeMerge',   # Categorical feature using WOE merging
    'education': [1, 2, 3, 4]   # Custom binning boundaries
}

# Build pipeline
pipeline = preprocessor.build_pipeline(feature_spec)

# Custom model with hyperparameter tuning
model_builder = Model(
    model_type='XGB',
    tune_method='BO',
    max_evals=200,
    param_grid={
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }
)

model_tuner = model_builder.build_model()
```

## Release Notes

### v0.4.1 (Current Version)
- **New Feature - ImbalanceSampler**: Advanced sampling techniques for handling imbalanced datasets
  - Support for 4 sampling strategies: under-sampling, over-sampling, SMOTE, and combined
  - Configurable target ratio for flexible class balance adjustment
  - Particularly useful for fraud detection and rare event prediction scenarios
- **Comprehensive Demo Script**: Added `demo.py` showcasing all major features
  - Demonstrations for binning methods (QCut, ChiMerge, OptimalCut)
  - Imbalanced data sampling examples with all 4 strategies
  - Feature selection pipeline demonstrations (IV, PSI, GINI, VIF, Correlation, Boosting Tree, Stability)
  - Model calibration examples (Platt and Isotonic calibration)
  - End-to-end training pipeline with fraud detection dataset
  - Metrics calculation examples (IV, GINI, KS, AUC, PSI)
- **Enhanced Package Exports**: Added `ImbalanceSampler` to `__all__` in `__init__.py` for easier imports
- **Documentation Updates**: Added comprehensive docstrings and usage examples for ImbalanceSampler
- **Bug Fixes**:
  - Fixed demo script compatibility with actual API parameter names
  - Corrected spec format for categorical features (use 'woeMerge' instead of 'c')
  - Added proper type conversion for boolean columns in VIF calculation
  - Fixed PlattCalibrator mapping_base format (dict instead of int)
  - Fixed Metrics input type requirements (pandas Series with names)
  - Added score_bins parameter handling in Train class
  - Fixed Reporter label column requirement in external data

### v0.4.0
- **Isotonic Calibration Support**: Added isotonic regression calibration method for improved calibration on skewed distributions
- **Auto-Mapping Feature**: Automatic score mapping generation for isotonic calibration with uniform and high-risk stretching modes
- **Enhanced Calibration API**: Support for both polynomial (log-odds) and isotonic (probability) calibration methods
- **Trainer Integration**: Full integration of calibration methods into Train class with `calibration_method` and `high_score_threshold` parameters
- **Improved Documentation**: Comprehensive documentation for all calibration modes and parameter combinations

### v0.3.1
- **New Feature - Imputer Class**: Introduced comprehensive `Imputer` class for missing value imputation with feature-level strategy control
  - Support for multiple imputation strategies: mean, median, min, max, mode, and separate
  - Flexible missing value identification with customizable missing value markers
  - Feature-level granular control for different imputation approaches
- **New Feature - StabilitySelector**: Advanced feature selection using stability selection methodology
  - Identifies features consistently selected across multiple data subsamples
  - Reduces risk of selecting spurious features and improves model generalization
  - Configurable iteration count, sampling fraction, and selection threshold
  - Based on bootstrap aggregating (bagging) with LightGBM as base estimator
- **Enhanced Encoder Robustness**: Fixed critical bugs in the encoder module
  - Fixed unknown missing value handling in transform stage
  - Fixed mixed type bugs when using strategy `False`
  - Fixed categorical missing data type conversion issues
- **Improved Binning Stability**: Enhanced binning algorithms reliability
  - Fixed WOEMerge categorical feature handling bug
  - Fixed categorical unique value binning edge cases
- **Module Refactoring and Enhancement**: Comprehensive code quality improvements across core modules
  - Enhanced calibrator module with better error handling and validation
  - Improved feature_selection module with additional robustness checks
  - Optimized pipeliner, reporter, and trainer modules for better performance
  - Updated documentation and type hints throughout the codebase

### v0.4.0
- **Isotonic Calibration Support**: Added isotonic regression calibration method for improved calibration on skewed distributions
- **Auto-Mapping Feature**: Automatic score mapping generation for isotonic calibration with uniform and high-risk stretching modes
- **Enhanced Calibration API**: Support for both polynomial (log-odds) and isotonic (probability) calibration methods
- **Trainer Integration**: Full integration of calibration methods into Train class with `calibration_method` and `high_score_threshold` parameters
- **Improved Documentation**: Comprehensive documentation for all calibration modes and parameter combinations

### v0.3.0
- **Enhanced Training Pipeline**: Introduced comprehensive `Train` class for end-to-end model training
- **Improved Pipeline Architecture**: Added advanced `Preprocess` and `Model` classes for streamlined workflows
- **Advanced Feature Selection**: Enhanced feature selection with VIF bug fixes and improved stability checks
- **Better Report Generation**: Overhauled reporting system with automatic file organization and enhanced visualizations
- **Robust Error Handling**: Added comprehensive input validation and error messages throughout the pipeline
- **PSI Calculation Fix**: Corrected Population Stability Index calculation for accurate distribution comparisons
- **Extended Documentation**: Added detailed function documentation and usage examples for all major components
- **Flexible Sample Types**: Support for custom sample types beyond train/test for comprehensive model evaluation
- **Encoder Code Simplification**: Merged `get_woe_df()` functionality into `transform()` method - when target variable `y` is provided, WOE DataFrame is automatically generated and stored in `self.woe_df`, eliminating the need for separate method calls

### v0.2.0 (Previous Version)
- **New Feature**: Integrated OptimalBinning algorithm into binner
- Added OptimalCut class for smarter optimal binning
- Fixed several issues with categorical feature handling
- Improved stability and efficiency of WOE encoding
- Optimized report generation format and content

### v0.1.0 (Initial Release)
- Initial project release
- Core functional modules:
  - Feature binning and WOE encoding
  - Multi-dimensional feature selection
  - Model training and evaluation
  - Hyperparameter tuning
  - Model calibration
  - Report generation

## Python Requirements
- Python >= 3.8

## Contributing

We welcome community contributions! Please follow these steps:

1. Fork this repository
2. Create a feature branch (`git checkout -b <BRANCH NAME>`)
3. Commit your changes (`git commit -m <YOUR COMMIT MESSAGE>`)
4. Push to the branch (`git push origin <BRANCH NAME>`)
5. Create a Pull Request

## Support

For questions or suggestions, please:
- Contact maintainer: klesterchueng@gmail.com

## Acknowledgments

Thanks to all developers and users who have contributed to the Optimus project.

---

**Optimus** - Making financial risk model development simpler and more efficient!

