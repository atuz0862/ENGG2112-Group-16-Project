"""
holdout_backtest.py - Honest out-of-sample evaluation
Trains both models on 2018-2021, tests on unseen 2022 data.
Provides unbiased proof-of-concept for the dispatch strategy.

Usage:
    python -m pipeline.holdout_backtest
"""

import pandas as pd
from xgboost import XGBRegressor
from models.price_model import engineer_price_features, PRICE_FEATURES, PRICE_TARGET
from models.solar_model import SOLAR_FEATURES, TARGET as SOLAR_TARGET
from pipeline.dispatch_strategy import smart_dispatch, dumb_dispatch, backtest

# ── LOAD & ENGINEER ───────────────────────────────────────────────────────────
print("Loading and engineering features...")
df = pd.read_csv('data/combined_engineered.csv')
df = engineer_price_features(df)
df = df.dropna(subset=SOLAR_FEATURES + PRICE_FEATURES + [SOLAR_TARGET, PRICE_TARGET])
df = df.reset_index(drop=True)

# ── SPLIT: TRAIN 2018–2021, TEST 2022 ────────────────────────────────────────
train = df[df['YEAR'] <= 2021].reset_index(drop=True)
test  = df[df['YEAR'] == 2022].reset_index(drop=True)

print(f"Train: {len(train):,} rows  (2018–2021)")
print(f"Test:  {len(test):,} rows  (2022 — never seen during training)")

# ── TRAIN BOTH MODELS ON 2018–2021 ONLY ──────────────────────────────────────
print("\nTraining models on 2018–2021...")

solar_model = XGBRegressor(
    n_estimators=100, max_depth=4, learning_rate=0.05,
    subsample=0.8, reg_lambda=1.0, verbosity=0,
)
price_model = XGBRegressor(
    n_estimators=300, max_depth=4, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0, verbosity=0,
)

solar_model.fit(train[SOLAR_FEATURES], train[SOLAR_TARGET])
price_model.fit(train[PRICE_FEATURES], train[PRICE_TARGET])
print("✓ Models trained")

# ── PREDICT ON UNSEEN 2022 DATA ───────────────────────────────────────────────
print("\nPredicting on unseen 2022 data...")

test['SOLAR_PRED']      = solar_model.predict(test[SOLAR_FEATURES])
test['PRICE_NORM_PRED'] = price_model.predict(test[PRICE_FEATURES])

# ── MEASURE PREDICTION ACCURACY ON 2022 ──────────────────────────────────────
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

solar_mae  = mean_absolute_error(test[SOLAR_TARGET], test['SOLAR_PRED'])
solar_rmse = np.sqrt(mean_squared_error(test[SOLAR_TARGET], test['SOLAR_PRED']))
price_mae  = mean_absolute_error(test[PRICE_TARGET], test['PRICE_NORM_PRED'])
price_rmse = np.sqrt(mean_squared_error(test[PRICE_TARGET], test['PRICE_NORM_PRED']))

print(f"\n{'=' * 50}")
print("OUT-OF-SAMPLE MODEL ACCURACY (2022 test set)")
print(f"{'=' * 50}")
print(f"  Solar model — MAE:  {solar_mae:.4f} W/m²  |  RMSE: {solar_rmse:.4f} W/m²")
print(f"  Price model — MAE:  {price_mae:.4f}        |  RMSE: {price_rmse:.4f}")

# ── RUN DISPATCH STRATEGY ON 2022 PREDICTIONS ─────────────────────────────────
print(f"\n{'=' * 50}")
print("DISPATCH STRATEGY — 2022 OUT-OF-SAMPLE BACKTEST")
print(f"{'=' * 50}")

smart_2022 = smart_dispatch(test)
dumb_2022  = dumb_dispatch(test)

smart_result = backtest(smart_2022)
dumb_result  = backtest(dumb_2022)
uplift = smart_result['net_benefit_$'] - dumb_result['net_benefit_$']

print(f"\n  Smart (ML) — trained on 2018–2021, deployed on unseen 2022:")
print(f"    Revenue from sales:  ${smart_result['total_revenue_$']:>8.2f}")
print(f"    Savings from USE:    ${smart_result['total_savings_$']:>8.2f}")
print(f"    Net benefit:         ${smart_result['net_benefit_$']:>8.2f}")

print(f"\n  Dumb (time-of-day):")
print(f"    Revenue from sales:  ${dumb_result['total_revenue_$']:>8.2f}")
print(f"    Savings from USE:    ${dumb_result['total_savings_$']:>8.2f}")
print(f"    Net benefit:         ${dumb_result['net_benefit_$']:>8.2f}")

print(f"\n  ML uplift over dumb baseline: ${uplift:.2f}")
marker = "✓ ML WINS" if uplift > 0 else "✗ ML loses"
print(f"  Result: {marker}")

# ── DECISION BREAKDOWN ────────────────────────────────────────────────────────
print(f"\n{'=' * 50}")
print("DECISION BREAKDOWN (2022)")
print(f"{'=' * 50}")
counts = smart_2022['DECISION'].value_counts()
for decision in ['SELL', 'STORE', 'USE', 'HOLD']:
    n = counts.get(decision, 0)
    pct = n / len(smart_2022) * 100
    print(f"  {decision:<6} {n:>5} hours  ({pct:.1f}%)")

# ── SAVE RESULTS ──────────────────────────────────────────────────────────────
smart_2022.to_csv('backtest/holdout_2022_decisions.csv', index=False)
print(f"\n✓ 2022 decisions saved to backtest/holdout_2022_decisions.csv")
