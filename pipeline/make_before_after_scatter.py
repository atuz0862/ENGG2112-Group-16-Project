"""
make_before_after_scatter.py - Before/after scatter plots showing model improvements.

Solar:  Linear Regression (prototype) vs XGBoost (final)
Price:  No lag features (prototype) vs all lag features (final)

Usage:
    python -m pipeline.make_before_after_scatter
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from models.solar_model import SOLAR_FEATURES, TARGET as SOLAR_TARGET
from models.price_model import engineer_price_features, PRICE_FEATURES, PRICE_TARGET

warnings.filterwarnings('ignore')
os.makedirs('plots', exist_ok=True)

# ── LOAD & SPLIT ──────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv('data/combined_engineered.csv')
df = engineer_price_features(df)
df = df.dropna(subset=SOLAR_FEATURES + PRICE_FEATURES + [SOLAR_TARGET, PRICE_TARGET])
df = df.reset_index(drop=True)

train = df[df['YEAR'] <= 2021].reset_index(drop=True)
test  = df[df['YEAR'] == 2022].reset_index(drop=True)

# ── SOLAR: train both models ──────────────────────────────────────────────────
print("Training solar models...")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(train[SOLAR_FEATURES])
X_test_s  = scaler.transform(test[SOLAR_FEATURES])

linear_solar = Ridge(alpha=1.0)
linear_solar.fit(X_train_s, train[SOLAR_TARGET])
solar_before = np.clip(linear_solar.predict(X_test_s), 0, None)

xgb_solar = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                          subsample=0.8, reg_lambda=1.0, verbosity=0)
xgb_solar.fit(train[SOLAR_FEATURES], train[SOLAR_TARGET])
solar_after = np.clip(xgb_solar.predict(test[SOLAR_FEATURES]), 0, None)

# Daylight only
day_mask = test[SOLAR_TARGET].values > 50
actual_solar = test[SOLAR_TARGET].values[day_mask]
pred_before_solar = solar_before[day_mask]
pred_after_solar  = solar_after[day_mask]

mae_before_s = mean_absolute_error(actual_solar, pred_before_solar)
mae_after_s  = mean_absolute_error(actual_solar, pred_after_solar)
r2_before_s  = r2_score(actual_solar, pred_before_solar)
r2_after_s   = r2_score(actual_solar, pred_after_solar)

print(f"  Solar — Before (Linear): MAE={mae_before_s:.2f}  R²={r2_before_s:.3f}")
print(f"  Solar — After  (XGBoost): MAE={mae_after_s:.2f}  R²={r2_after_s:.3f}")

# ── PRICE: train both models ──────────────────────────────────────────────────
print("Training price models...")

PRICE_NO_LAGS = [f for f in PRICE_FEATURES
                 if f not in ['RRP_NORM_LAG_1H', 'RRP_NORM_LAG_3H',
                               'RRP_NORM_LAG_24H', 'RRP_NORM_LAG_168H',
                               'ROLL_MEAN_24H', 'ROLL_STD_24H']]

xgb_no_lag = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.03,
                           subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0, verbosity=0)
xgb_no_lag.fit(train[PRICE_NO_LAGS], train[PRICE_TARGET])
price_before = xgb_no_lag.predict(test[PRICE_NO_LAGS])

xgb_price = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.03,
                          subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0, verbosity=0)
xgb_price.fit(train[PRICE_FEATURES], train[PRICE_TARGET])
price_after = xgb_price.predict(test[PRICE_FEATURES])

actual_price = test[PRICE_TARGET].values
mae_before_p = mean_absolute_error(actual_price, price_before)
mae_after_p  = mean_absolute_error(actual_price, price_after)
r2_before_p  = r2_score(actual_price, price_before)
r2_after_p   = r2_score(actual_price, price_after)

print(f"  Price — Before (no lags): MAE={mae_before_p:.4f}  R²={r2_before_p:.3f}")
print(f"  Price — After  (all lags): MAE={mae_after_p:.4f}  R²={r2_after_p:.3f}")


def scatter_panel(ax, actual, predicted, mae, r2, title, colour, subtitle=None):
    """Draw a single scatter panel."""
    # Sample to avoid overcrowding
    idx = np.random.default_rng(42).choice(len(actual), size=min(2000, len(actual)), replace=False)
    ax.scatter(actual[idx], predicted[idx], alpha=0.25, s=10,
               color=colour, linewidths=0)

    max_val = max(actual.max(), predicted.max()) * 1.05
    ax.plot([0, max_val], [0, max_val], color='black', linewidth=1.5,
            linestyle='--', label='y = x (perfect)', zorder=5)

    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_aspect('equal')

    stats = f'MAE = {mae:.2f}\nR²  = {r2:.3f}'
    ax.text(0.05, 0.95, stats, transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#BDC3C7'))

    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    if subtitle:
        ax.set_title(f'{title}\n{subtitle}', fontsize=11, fontweight='bold', pad=8)

    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.2)
    ax.spines[['top', 'right']].set_visible(False)


# ── GRAPH 1a — Solar BEFORE ───────────────────────────────────────────────────
print("\nGenerating solar BEFORE scatter...")

fig, ax = plt.subplots(figsize=(7, 7))
fig.suptitle("Solar Model — BEFORE\n(Linear Regression, 2022 Out-of-Sample)",
             fontsize=14, fontweight='bold')

scatter_panel(ax, actual_solar, pred_before_solar, mae_before_s, r2_before_s,
              'Linear Regression', '#E74C3C',
              subtitle='No feature interactions captured')
ax.set_xlabel('Actual Irradiance (W/m²)', fontsize=11)
ax.set_ylabel('Predicted Irradiance (W/m²)', fontsize=11)

fig.tight_layout()
fig.savefig('plots/solar_before_scatter.png', dpi=180, bbox_inches='tight')
plt.close(fig)
print("  ✓ plots/solar_before_scatter.png")

# ── GRAPH 1b — Solar AFTER ────────────────────────────────────────────────────
print("Generating solar AFTER scatter...")

fig, ax = plt.subplots(figsize=(7, 7))
fig.suptitle("Solar Model — AFTER\n(XGBoost, 2022 Out-of-Sample)",
             fontsize=14, fontweight='bold')

scatter_panel(ax, actual_solar, pred_after_solar, mae_after_s, r2_after_s,
              'XGBoost', '#27AE60',
              subtitle='Captures non-linear patterns & feature interactions')
ax.set_xlabel('Actual Irradiance (W/m²)', fontsize=11)
ax.set_ylabel('Predicted Irradiance (W/m²)', fontsize=11)

fig.tight_layout()
fig.savefig('plots/solar_after_scatter.png', dpi=180, bbox_inches='tight')
plt.close(fig)
print("  ✓ plots/solar_after_scatter.png")


# ── GRAPH 2 — Price Before/After ──────────────────────────────────────────────
print("Generating price before/after scatter...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
fig.suptitle("Price Model — Before vs After Adding Lag Features\n(2022 Out-of-Sample)",
             fontsize=14, fontweight='bold', y=1.01)

scatter_panel(ax1, actual_price, price_before, mae_before_p, r2_before_p,
              'BEFORE', '#E74C3C',
              subtitle='XGBoost — time/weather features only, no price history')
scatter_panel(ax2, actual_price, price_after, mae_after_p, r2_after_p,
              'AFTER', '#27AE60',
              subtitle='XGBoost — with lag features (1h, 3h, 24h, 168h)')

for ax in (ax1, ax2):
    ax.set_xlabel('Actual Normalised Price (0–1)', fontsize=11)
ax1.set_ylabel('Predicted Normalised Price (0–1)', fontsize=11)
ax2.set_ylabel('Predicted Normalised Price (0–1)', fontsize=11)

improvement_p = ((mae_before_p - mae_after_p) / mae_before_p) * 100
fig.text(0.5, -0.01,
         f'MAE improved from {mae_before_p:.4f} → {mae_after_p:.4f}  '
         f'({improvement_p:.0f}% reduction in error)',
         ha='center', fontsize=12, fontweight='bold', color='#27AE60')

fig.tight_layout()
fig.savefig('plots/price_before_after_scatter.png', dpi=180, bbox_inches='tight')
plt.close(fig)
print("  ✓ plots/price_before_after_scatter.png")

print("\nDone — 2 before/after scatter plots saved to plots/")
