"""
run_pipeline.py - End-to-end pipeline combining solar and price models
Generates predictions from both models and runs dispatch strategy comparison.

Usage:
    python -m pipeline.run_pipeline
"""

import pandas as pd
from xgboost import XGBRegressor
from models.price_model import engineer_price_features, PRICE_FEATURES, PRICE_TARGET
from models.solar_model import SOLAR_FEATURES, TARGET as SOLAR_TARGET
from pipeline.walk_forward_validation import walk_forward_validate, summarise
from pipeline.dispatch_strategy import compare_strategies

# ── LOAD & ENGINEER FEATURES ──────────────────────────────────────────────────
print("Loading and engineering features...")
df = pd.read_csv('data/combined_engineered.csv')
df = engineer_price_features(df)   # adds RRP_NORM, cyclical cols, lags, day_min, day_range

# Align: drop rows where any solar or price feature is NaN
df = df.dropna(subset=SOLAR_FEATURES + PRICE_FEATURES + [SOLAR_TARGET, PRICE_TARGET])
df = df.reset_index(drop=True)
print(f"Dataset shape (both models aligned): {df.shape}")

# ── WALK-FORWARD VALIDATION ───────────────────────────────────────────────────
print("\nRunning walk-forward validation...")

solar_model = XGBRegressor(
    n_estimators=100, max_depth=4, learning_rate=0.05,
    subsample=0.8, reg_lambda=1.0, verbosity=0,
)
price_model = XGBRegressor(
    n_estimators=300,        # more trees to capture complex price patterns
    max_depth=4,
    learning_rate=0.03,      # slower learning rate paired with more trees
    subsample=0.8,
    colsample_bytree=0.7,    # each tree uses 70% of features — reduces overfitting
    reg_lambda=1.0,
    verbosity=0,
)

solar_results = walk_forward_validate(df, SOLAR_FEATURES, SOLAR_TARGET, solar_model)
price_results = walk_forward_validate(df, PRICE_FEATURES, PRICE_TARGET, price_model)

summarise(solar_results, label="Solar Model (ALLSKY_IRRADIANCE W/m²)")
summarise(price_results, label="Price Model (RRP_NORM 0–1 scale)")

solar_results.to_csv('backtest/solar_wfv_results.csv', index=False)
price_results.to_csv('backtest/price_wfv_results.csv', index=False)
print("\n✓ WFV results saved to backtest/")

# ── GENERATE PREDICTIONS ──────────────────────────────────────────────────────
print("\nGenerating predictions on full dataset...")

solar_model.fit(df[SOLAR_FEATURES], df[SOLAR_TARGET])
price_model.fit(df[PRICE_FEATURES], df[PRICE_TARGET])

df['SOLAR_PRED']      = solar_model.predict(df[SOLAR_FEATURES])
df['PRICE_NORM_PRED'] = price_model.predict(df[PRICE_FEATURES])

print(f"  SOLAR_PRED      — mean: {df['SOLAR_PRED'].mean():.2f} W/m²")
print(f"  PRICE_NORM_PRED — mean: {df['PRICE_NORM_PRED'].mean():.3f}  (0=cheapest, 1=most expensive)")

# ── DISPATCH STRATEGY COMPARISON ─────────────────────────────────────────────
compare_strategies(df)
