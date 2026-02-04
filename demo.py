#!/usr/bin/env python
"""
Optimus Demo Script
===================

This script demonstrates the main features of the Optimus package:
1. Data preparation with imbalanced sampling
2. Feature imputation
3. WOE encoding
4. Feature selection (IV, PSI, GINI, VIF, Correlation, Boosting, Stability)
5. Model training (LR, XGB, LGBM)
6. Model calibration (Platt, Isotonic)
7. Performance evaluation and reporting
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from optimus import (
    BoostingTreeSelector,
    ChiMergeCut,
    CorrSelector,
    Encoder,
    GINISelector,
    ImbalanceSampler,
    Imputer,
    IsotonicCalibrator,
    IVSelector,
    Metrics,
    OptimalCut,
    PlattCalibrator,
    PSISelector,
    QCut,
    StabilitySelector,
    Train,
    VIFSelector,
)


def generate_synthetic_data(n_samples=5000, fraud_rate=0.05, random_state=42):
    """
    Generate synthetic fraud detection dataset.

    Features:
    - transaction_amount: Transaction amount (continuous)
    - account_age_days: Account age in days (continuous)
    - num_transactions_30d: Number of transactions in last 30 days (continuous)
    - device_type: Device type (categorical: mobile/desktop/tablet)
    - transaction_hour: Hour of transaction (continuous)
    - is_international: International transaction flag (binary)
    - avg_transaction_amount: Average historical transaction amount (continuous)
    - account_balance: Account balance (continuous, with missing values)
    - merchant_category: Merchant category (categorical)
    - velocity_score: Transaction velocity score (continuous, with missing values)

    Target:
    - is_fraud: Fraud label (0=normal, 1=fraud)
    """
    np.random.seed(random_state)

    n_fraud = int(n_samples * fraud_rate)
    n_normal = n_samples - n_fraud

    # Generate normal transactions
    normal_data = {
        "transaction_amount": np.random.lognormal(mean=4, sigma=1, size=n_normal),
        "account_age_days": np.random.gamma(shape=5, scale=100, size=n_normal),
        "num_transactions_30d": np.random.poisson(lam=10, size=n_normal),
        "device_type": np.random.choice(
            ["mobile", "desktop", "tablet"], size=n_normal, p=[0.5, 0.3, 0.2]
        ),
        "transaction_hour": np.random.randint(0, 24, size=n_normal),
        "is_international": np.random.choice([0, 1], size=n_normal, p=[0.9, 0.1]),
        "avg_transaction_amount": np.random.lognormal(
            mean=3.5, sigma=0.8, size=n_normal
        ),
        "account_balance": np.random.lognormal(mean=7, sigma=1.5, size=n_normal),
        "merchant_category": np.random.choice(
            ["retail", "food", "travel", "online", "other"], size=n_normal
        ),
        "velocity_score": np.random.normal(loc=50, scale=15, size=n_normal),
        "is_fraud": np.zeros(n_normal, dtype=int),
    }

    # Generate fraud transactions (different distribution)
    fraud_data = {
        "transaction_amount": np.random.lognormal(
            mean=5.5, sigma=1.5, size=n_fraud
        ),  # Higher amounts
        "account_age_days": np.random.gamma(
            shape=2, scale=30, size=n_fraud
        ),  # Newer accounts
        "num_transactions_30d": np.random.poisson(
            lam=25, size=n_fraud
        ),  # More transactions
        "device_type": np.random.choice(
            ["mobile", "desktop", "tablet"], size=n_fraud, p=[0.7, 0.2, 0.1]
        ),
        "transaction_hour": np.random.choice(
            [0, 1, 2, 3, 22, 23], size=n_fraud
        ),  # Unusual hours
        "is_international": np.random.choice(
            [0, 1], size=n_fraud, p=[0.3, 0.7]
        ),  # More international
        "avg_transaction_amount": np.random.lognormal(
            mean=3, sigma=0.5, size=n_fraud
        ),  # Lower avg
        "account_balance": np.random.lognormal(
            mean=6, sigma=1, size=n_fraud
        ),  # Lower balance
        "merchant_category": np.random.choice(
            ["retail", "food", "travel", "online", "other"],
            size=n_fraud,
            p=[0.1, 0.1, 0.2, 0.5, 0.1],
        ),
        "velocity_score": np.random.normal(
            loc=80, scale=20, size=n_fraud
        ),  # Higher velocity
        "is_fraud": np.ones(n_fraud, dtype=int),
    }

    # Combine and shuffle
    df = pd.concat(
        [pd.DataFrame(normal_data), pd.DataFrame(fraud_data)], ignore_index=True
    )

    # Introduce missing values (10% for account_balance, 15% for velocity_score)
    missing_indices_balance = np.random.choice(
        df.index, size=int(0.1 * len(df)), replace=False
    )
    df.loc[missing_indices_balance, "account_balance"] = np.nan

    missing_indices_velocity = np.random.choice(
        df.index, size=int(0.15 * len(df)), replace=False
    )
    df.loc[missing_indices_velocity, "velocity_score"] = np.nan

    # Shuffle
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df


def demo_binning():
    """Demonstrate binning methods."""
    print("\n" + "=" * 80)
    print("DEMO 1: Binning Methods")
    print("=" * 80)

    # Generate small dataset for binning demo
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame(
        {
            "age": np.random.gamma(shape=5, scale=8, size=n),
            "income": np.random.lognormal(mean=10, sigma=1, size=n),
            "is_default": np.random.choice([0, 1], size=n, p=[0.95, 0.05]),
        }
    )

    # QCut - Equal frequency binning
    print("\n[1.1] QCut - Equal Frequency Binning")
    qcut = QCut(target_bin_cnt=5)
    age_binned = qcut.fit_transform(df["age"], df["is_default"])
    print(f"Age bins (QCut): {qcut.bins}")

    # ChiMergeCut - Chi-square based binning
    print("\n[1.2] ChiMergeCut - Chi-square Based Binning")
    chicut = ChiMergeCut(target_intervals=5, initial_intervals=False)
    income_binned = chicut.fit_transform(df["income"], df["is_default"])
    print(f"Income bins (ChiMerge): {chicut.bins}")

    # OptimalCut - Optimal binning using optbinning library
    print("\n[1.3] OptimalCut - Optimal Binning (optbinning)")
    optcut = OptimalCut(name="age", dtype="numerical", max_n_bins=5)
    age_opt_binned = optcut.fit_transform(df["age"], df["is_default"])
    print(f"Age bins (Optimal): {optcut.bins}")


def demo_imbalance_sampling():
    """Demonstrate imbalanced data sampling."""
    print("\n" + "=" * 80)
    print("DEMO 2: Imbalanced Data Sampling")
    print("=" * 80)

    # Generate imbalanced dataset
    df = generate_synthetic_data(n_samples=2000, fraud_rate=0.03)
    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]

    # Select only numeric features for sampling demo
    numeric_features = X.select_dtypes(include=[np.number]).columns
    X_numeric = X[numeric_features]

    print(f"\nOriginal dataset: {len(y)} samples, fraud rate: {y.mean():.2%}")

    # Strategy 1: Under-sampling
    print("\n[2.1] Under-sampling Strategy")
    sampler_under = ImbalanceSampler(
        strategy="under", target_ratio=0.2, random_state=42
    )
    X_under, y_under = sampler_under.fit_resample(X_numeric, y)
    print(
        f"After under-sampling: {len(y_under)} samples, fraud rate: {y_under.mean():.2%}"
    )

    # Strategy 2: Over-sampling
    print("\n[2.2] Over-sampling Strategy")
    sampler_over = ImbalanceSampler(strategy="over", target_ratio=0.2, random_state=42)
    X_over, y_over = sampler_over.fit_resample(X_numeric, y)
    print(
        f"After over-sampling: {len(y_over)} samples, fraud rate: {y_over.mean():.2%}"
    )

    # Strategy 3: SMOTE
    print("\n[2.3] SMOTE Strategy")
    sampler_smote = ImbalanceSampler(
        strategy="smote", target_ratio=0.2, random_state=42
    )
    X_smote, y_smote = sampler_smote.fit_resample(X_numeric, y)
    print(f"After SMOTE: {len(y_smote)} samples, fraud rate: {y_smote.mean():.2%}")

    # Strategy 4: Combined
    print("\n[2.4] Combined Strategy")
    sampler_combined = ImbalanceSampler(
        strategy="combined", target_ratio=0.25, random_state=42
    )
    X_combined, y_combined = sampler_combined.fit_resample(X_numeric, y)
    print(
        f"After combined: {len(y_combined)} samples, fraud rate: {y_combined.mean():.2%}"
    )


def demo_feature_selection():
    """Demonstrate feature selection methods."""
    print("\n" + "=" * 80)
    print("DEMO 3: Feature Selection")
    print("=" * 80)

    # Generate dataset
    df = generate_synthetic_data(n_samples=3000, fraud_rate=0.05)

    # Prepare data
    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]

    # One-hot encode categorical features
    X_encoded = pd.get_dummies(
        X, columns=["device_type", "merchant_category"], drop_first=True
    )

    # Fill missing values for demo
    X_encoded = X_encoded.fillna(X_encoded.median())

    # Ensure all columns are numeric (convert bool to int if needed)
    for col in X_encoded.columns:
        if X_encoded[col].dtype == "bool":
            X_encoded[col] = X_encoded[col].astype(int)

    # Split data
    train_idx = np.random.choice(
        len(X_encoded), size=int(0.7 * len(X_encoded)), replace=False
    )
    test_idx = np.setdiff1d(np.arange(len(X_encoded)), train_idx)

    X_train = X_encoded.iloc[train_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    X_test = X_encoded.iloc[test_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    print(f"\nOriginal features: {X_train.shape[1]}")

    # IV Selector
    print("\n[3.1] IV Selector (threshold=0.02)")
    iv_selector = IVSelector(iv_threshold=0.02)
    iv_selector.fit(X_train, y_train)
    print(f"Selected features: {len(iv_selector.selected_features)}")
    X_train_iv = iv_selector.transform(X_train)

    # PSI Selector
    print("\n[3.2] PSI Selector (threshold=0.1)")
    psi_selector = PSISelector(psi_threshold=0.1)
    psi_selector.fit(X_train_iv, y_train, refX=X_test)
    print(f"Selected features: {len(psi_selector.selected_features)}")

    # GINI Selector
    print("\n[3.3] GINI Selector (sign consistency)")
    gini_selector = GINISelector()
    gini_selector.fit(X_train_iv, y_train, refX=X_test, refy=y_test)
    print(f"Selected features: {len(gini_selector.selected_features)}")

    # Correlation Selector
    print("\n[3.4] Correlation Selector (threshold=0.8)")
    corr_selector = CorrSelector(corr_threshold=0.8, method="iv_ascending")
    corr_selector.fit(X_train, y_train)
    print(f"Selected features: {len(corr_selector.selected_features)}")

    # VIF Selector
    print("\n[3.5] VIF Selector (threshold=10)")
    vif_selector = VIFSelector(vif_threshold=10)
    vif_selector.fit(X_train, y_train)
    print(f"Selected features: {len(vif_selector.selected_features)}")

    # Boosting Tree Selector
    print("\n[3.6] Boosting Tree Selector (select_frac=0.8)")
    boosting_selector = BoostingTreeSelector(select_frac=0.8)
    boosting_selector.fit(X_train, y_train, refX=X_test, refy=y_test)
    print(f"Selected features: {len(boosting_selector.selected_features)}")

    # Stability Selector
    print("\n[3.7] Stability Selector (threshold=0.6, n_iterations=20)")
    stability_selector = StabilitySelector(
        threshold=0.6, n_iterations=20, random_state=42
    )
    stability_selector.fit(X_train, y_train)
    print(f"Selected features: {len(stability_selector.selected_features)}")


def demo_calibration():
    """Demonstrate calibration methods."""
    print("\n" + "=" * 80)
    print("DEMO 4: Model Calibration")
    print("=" * 80)

    # Generate synthetic probabilities and labels
    np.random.seed(42)
    n = 1000

    # Simulate model predictions (uncalibrated)
    y_true = np.random.choice([0, 1], size=n, p=[0.9, 0.1])
    # Add systematic bias to probabilities
    y_prob = np.where(
        y_true == 1,
        np.random.beta(8, 2, size=n),  # Fraud cases: higher probs
        np.random.beta(2, 8, size=n),  # Normal cases: lower probs
    )

    # Add some miscalibration
    y_prob = np.clip(y_prob * 0.7 + 0.15, 0, 1)

    # Split train/test
    train_size = int(0.7 * n)
    y_prob_train, y_prob_test = y_prob[:train_size], y_prob[train_size:]
    y_train, y_test = y_true[:train_size], y_true[train_size:]

    # Platt Calibration
    print("\n[4.1] Platt Calibration")
    platt = PlattCalibrator(
        n_bins=10,
        n_degree=2,
        mapping_base={300: 0.5, 600: 0.1, 900: 0.01},  # {score: probability}
        score_floor=300,
        score_cap=900,
    )
    platt.fit(y_prob_train, y_train)
    scores_platt = platt.transform(y_prob_test)
    print(f"Platt scores range: [{scores_platt.min():.1f}, {scores_platt.max():.1f}]")

    # Isotonic Calibration (standard)
    print("\n[4.2] Isotonic Calibration (standard)")
    isotonic = IsotonicCalibrator(
        n_bins=10,
        score_floor=0.0,
        score_cap=1.0,
        scale_threshold=None,  # Scale all probabilities
    )
    isotonic.fit(y_prob_train, y_train)
    scores_isotonic = isotonic.transform(y_prob_test)
    print(
        f"Isotonic scores range: [{scores_isotonic.min():.3f}, {scores_isotonic.max():.3f}]"
    )

    # Isotonic Calibration (with threshold)
    print("\n[4.3] Isotonic Calibration (high-risk enhanced)")
    isotonic_threshold = IsotonicCalibrator(
        n_bins=10,
        score_floor=0.0,
        score_cap=1.0,
        scale_threshold=0.3,  # Enhance high-risk region
    )
    isotonic_threshold.fit(y_prob_train, y_train)
    scores_isotonic_threshold = isotonic_threshold.transform(y_prob_test)
    print(
        f"Isotonic (threshold) scores range: [{scores_isotonic_threshold.min():.3f}, {scores_isotonic_threshold.max():.3f}]"
    )

    # Compare calibration
    if y_test.sum() > 0:  # If we have positive cases
        print("\n[4.4] Scorecard Comparison (Platt)")
        scorecard_platt = platt.compare_calibrate_result(scores_platt, y_test)
        print(
            scorecard_platt[["score_bin", "total", "bad_rate", "ks"]].to_string(
                index=False
            )
        )


def demo_end_to_end_training():
    """Demonstrate end-to-end model training pipeline."""
    print("\n" + "=" * 80)
    print("DEMO 5: End-to-End Model Training")
    print("=" * 80)

    # Generate dataset
    df = generate_synthetic_data(n_samples=4000, fraud_rate=0.05, random_state=42)

    # Prepare features and target
    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]

    # Create sample_type column for train/test split
    # Note: 'e' must include the label column for the Reporter to work
    train_size = int(0.7 * len(df))
    e = pd.DataFrame(
        {
            "sample_type": ["train"] * train_size + ["test"] * (len(df) - train_size),
            "is_fraud": y.values,  # Add label column for Reporter
        }
    )

    # Define feature specification
    spec = {
        "transaction_amount": "auto",
        "account_age_days": "auto",
        "num_transactions_30d": "auto",
        "device_type": "woeMerge",  # categorical - use WOE merging
        "transaction_hour": "auto",
        "is_international": "auto",
        "avg_transaction_amount": "auto",
        "account_balance": "auto",
        "merchant_category": "woeMerge",  # categorical - use WOE merging
        "velocity_score": "auto",
    }

    print("\n[5.1] Training Logistic Regression Model")
    trainer_lr = Train(
        model_type="LR",
        spec=spec,
        calibration_method="platt",
        n_bins=10,
        score_floor=300,
        score_cap=900,
        mapping_base={300: 0.5, 600: 0.1, 900: 0.01},  # {score: probability}
        score_bins=[300, 400, 500, 600, 700, 800, 900],  # Score bin edges
        model_path="./demo_models",
        iv_threshold=0.02,
        psi_threshold=0.1,
        corr_threshold=0.8,
        vif_threshold=10,
        boosting_select_frac=0.9,
        stability_threshold=0.6,
    )

    # Fit model
    trainer_lr.fit(X, y, e)

    print("\n[5.2] Generating Predictions")
    performance = trainer_lr.transform(X, y, e)

    # Print performance summary
    print("\n[5.3] Performance Summary")
    print(f"Timestamp: {trainer_lr.ts}")

    # Print scorecard summary for train and test
    if "scorecard" in performance:
        print("\nScorecard Summary:")
        for sample_type in ["train", "test"]:
            if sample_type in performance["scorecard"]:
                scorecard = performance["scorecard"][sample_type]
                print(f"\n{sample_type.upper()} Set Scorecard (first 3 bins):")
                print(scorecard.head(3).to_string())

    print(f"\nFeature Importance (Top 5):")
    if performance.get("feature_importance") is not None:
        top_features = performance["feature_importance"].head(5)
        print(top_features.to_string(index=False))

    print(f"\nScore Distribution PSI:")
    if "scoredist" in performance:
        print(performance["scoredist"]["PSI"].to_string())

    # Generate report
    print("\n[5.4] Generating Report")
    trainer_lr.write_report(
        performance=performance,
        report_path="./demo_models",
        report_name=f"fraud_model_report_{trainer_lr.ts}",
    )
    print(f"Report saved to: ./demo_models/fraud_model_report_{trainer_lr.ts}")


def demo_metrics():
    """Demonstrate metrics calculation."""
    print("\n" + "=" * 80)
    print("DEMO 6: Metrics Calculation")
    print("=" * 80)

    # Generate sample predictions
    np.random.seed(42)
    y_true = pd.Series(np.random.choice([0, 1], size=500, p=[0.9, 0.1]), name="label")
    y_pred = pd.Series(
        np.where(
            y_true == 1, np.random.beta(7, 3, size=500), np.random.beta(3, 7, size=500)
        ),
        name="probability",
    )

    # Calculate various metrics
    print("\n[6.1] Information Value (IV)")
    feature_values = pd.Series(np.random.randn(500), name="feature_x")
    iv = Metrics.get_iv(y_true, feature_values)
    print(f"IV: {iv:.4f}")

    print("\n[6.2] GINI Coefficient")
    gini = Metrics.get_gini(y_true, y_pred)
    print(f"GINI: {gini:.4f}")

    print("\n[6.3] KS Statistic")
    ks = Metrics.get_ks(y_true, y_pred)
    print(f"KS: {ks:.4f}")

    print("\n[6.4] AUC")
    auc = Metrics.get_auc(y_true, y_pred)
    print(f"AUC: {auc:.4f}")

    print("\n[6.5] PSI (Population Stability Index)")
    # Simulate two distributions
    base_dist = np.random.randn(500)
    new_dist = np.random.randn(500) + 0.3  # Slight shift
    psi = Metrics.get_psi(base_dist, new_dist)
    print(f"PSI: {psi:.4f}")


def main():
    """Run all demos."""
    print("\n")
    print("=" * 80)
    print(" " * 25 + "OPTIMUS DEMO SCRIPT")
    print("=" * 80)
    print("\nThis script demonstrates the main features of Optimus v0.4.0")
    print("A comprehensive toolkit for fraud/credit/growth risk modeling.\n")

    try:
        # Run demos
        demo_binning()
        demo_imbalance_sampling()
        demo_feature_selection()
        demo_calibration()
        demo_metrics()
        demo_end_to_end_training()

        print("\n" + "=" * 80)
        print(" " * 25 + "DEMO COMPLETED!")
        print("=" * 80)
        print("\nAll features demonstrated successfully.")
        print("Check ./demo_models/ for saved model artifacts and reports.\n")

    except Exception as e:
        print(f"\n[ERROR] Demo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
