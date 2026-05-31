"""
walk_forward_validation.py - Generic walk-forward validation loop
Works with any sklearn-compatible model (solar, price, or otherwise).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def walk_forward_validate(df, features, target, model, W_train=500, W_test=100, purge=7):
    """
    Generic walk-forward validation for any sklearn-compatible model.

    Parameters
    ----------
    df       : pd.DataFrame   — full dataset, sorted by time
    features : list[str]      — input feature column names
    target   : str            — target column name
    model    : estimator      — sklearn-compatible (must have fit/predict)
    W_train  : int            — rows per training window
    W_test   : int            — rows per test window
    purge    : int            — embargo gap between train and test end

    Returns
    -------
    pd.DataFrame with columns: fold, mae, rmse
    """
    results = []
    start = 0

    while start + W_train + purge + W_test <= len(df):
        train_end  = start + W_train
        test_start = train_end + purge
        test_end   = test_start + W_test

        train = df.iloc[start:train_end]
        test  = df.iloc[test_start:test_end]

        model.fit(train[features], train[target])
        preds   = model.predict(test[features])
        actuals = test[target].values

        mae  = mean_absolute_error(actuals, preds)
        rmse = np.sqrt(mean_squared_error(actuals, preds))

        results.append({
            'fold': len(results) + 1,
            'mae':  mae,
            'rmse': rmse,
        })

        start += W_test

    return pd.DataFrame(results)


def summarise(results_df, label="Model"):
    """Print mean/std MAE and RMSE from a walk_forward_validate result."""
    print(f"\n{'=' * 50}")
    print(f"{label} — Walk-Forward Results ({len(results_df)} folds)")
    print(f"{'=' * 50}")
    print(f"  Mean MAE:  {results_df['mae'].mean():.4f}")
    print(f"  Std  MAE:  {results_df['mae'].std():.4f}")
    print(f"  Mean RMSE: {results_df['rmse'].mean():.4f}")
