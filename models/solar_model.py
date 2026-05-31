"""
solar_model.py - Solar Generation Forecasting Model
Predicts next hour solar irradiance using XGBoost
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── FEATURES & TARGET ────────────────────────────────────────────────────────
SOLAR_FEATURES = [
    'CLRSKY_IRRADIANCE',    # 47.4% - keep
    'SOLAR_LAG_1H',         # 38.5% - keep
    'SOLAR_ROLLING_6H_STD', # 7.9%  - keep
    'CLEARNESS_IDX',        # 4.4%  - keep
    'HOUR_OF_DAY',          # 1.3%  - keep
]

TARGET = 'ALLSKY_IRRADIANCE'

# ── WALK-FORWARD VALIDATION ──────────────────────────────────────────────────
def walk_forward_solar(df, features, target, W_train=500, W_test=100, purge=7):
    """
    Walk-forward validation for solar model.
    """
    results = []
    start = 0

    while start + W_train + purge + W_test <= len(df):
        train_end  = start + W_train
        test_start = train_end + purge
        test_end   = test_start + W_test

        train = df.iloc[start:train_end]
        test  = df.iloc[test_start:test_end]

        model = XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            reg_lambda=1.0,
            verbosity=0,
        )

        model.fit(train[features], train[target])
        preds = model.predict(test[features])
        actuals = test[target].values

        mae  = mean_absolute_error(actuals, preds)
        rmse = np.sqrt(mean_squared_error(actuals, preds))

        results.append({
            'fold':  len(results) + 1,
            'mae':   mae,
            'rmse':  rmse,
        })

        start += W_test

    return pd.DataFrame(results)


if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_csv('data/solar_engineered.csv')
    df = df.dropna(subset=SOLAR_FEATURES + [TARGET])

    print(f"Dataset shape: {df.shape}")
    print(f"Target: {TARGET}")
    print(f"Features: {SOLAR_FEATURES}")

    print("\nRunning walk-forward validation...")
    results = walk_forward_solar(df, SOLAR_FEATURES, TARGET)

    print("\n" + "=" * 50)
    print("SOLAR MODEL RESULTS")
    print("=" * 50)
    print(results.to_string(index=False))
    print(f"\nMean MAE:  {results['mae'].mean():.4f} W/m²")
    print(f"Std  MAE:  {results['mae'].std():.4f} W/m²")
    print(f"Mean RMSE: {results['rmse'].mean():.4f} W/m²")

    results.to_csv('backtest/solar_model_results.csv', index=False)
    print("\n✓ Results saved to backtest/solar_model_results.csv")

    # Train final model on full dataset
    final_model = XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        reg_lambda=1.0,
        verbosity=0,
    )
    final_model.fit(df[SOLAR_FEATURES], df[TARGET])

    # Feature importance
    importances = pd.Series(
        final_model.feature_importances_,
        index=SOLAR_FEATURES
    ).sort_values(ascending=False)

    print("\n" + "=" * 50)
    print("FEATURE IMPORTANCE (Influence Factors)")
    print("=" * 50)
    for feat, score in importances.items():
        bar = "█" * int(score * 50)
        print(f"  {feat:<25} {score:.4f}  {bar}")

    # Plot
    plt.figure(figsize=(10, 6))
    importances.plot(kind='bar', color='steelblue')
    plt.title('Feature Importance - Solar Generation Model')
    plt.xlabel('Feature')
    plt.ylabel('Importance Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('backtest/solar_feature_importance.png', dpi=150)
    print("\n✓ Plot saved to backtest/solar_feature_importance.png")