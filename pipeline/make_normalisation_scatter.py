"""
make_normalisation_scatter.py - Before/after target normalisation scatter plots.

BEFORE: XGBoost predicting raw RRP ($/MWh) — chaotic due to price spikes
AFTER:  XGBoost predicting RRP_NORM (0–1) — clean bounded predictions

Usage:
    python -m pipeline.make_normalisation_scatter
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

from models.solar_model import SOLAR_FEATURES, TARGET as SOLAR_TARGET
from models.price_model import engineer_price_features, PRICE_FEATURES, PRICE_TARGET

warnings.filterwarnings('ignore')
os.makedirs('plots', exist_ok=True)

# ── LOAD ──────────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv('data/combined_engineered.csv')
df = engineer_price_features(df)
df = df.dropna(subset=SOLAR_FEATURES + PRICE_FEATURES + [SOLAR_TARGET, PRICE_TARGET])
df = df.reset_index(drop=True)

train = df[df['YEAR'] <= 2021].reset_index(drop=True)
test  = df[df['YEAR'] == 2022].reset_index(drop=True)

# ── MODEL 1: Predict raw RRP (unnormalised) ───────────────────────────────────
print("Training model on raw RRP (unnormalised)...")

# Use same features but predict raw RRP
RAW_FEATURES = [f for f in PRICE_FEATURES
                if f not in ['RRP_NORM_LAG_1H', 'RRP_NORM_LAG_3H',
                             'RRP_NORM_LAG_24H', 'RRP_NORM_LAG_168H',
                             'ROLL_MEAN_24H', 'ROLL_STD_24H']]

# Add raw RRP lags instead
train = train.copy()
test  = test.copy()
for lag in [1, 24, 168]:
    train[f'RRP_LAG_{lag}H_FEAT'] = train['RRP'].shift(lag)
    test[f'RRP_LAG_{lag}H_FEAT']  = test['RRP'].shift(lag)

RAW_LAG_FEATURES = RAW_FEATURES + ['RRP_LAG_1H_FEAT', 'RRP_LAG_24H_FEAT', 'RRP_LAG_168H_FEAT']
train_raw = train.dropna(subset=RAW_LAG_FEATURES + ['RRP']).reset_index(drop=True)
test_raw  = test.dropna(subset=RAW_LAG_FEATURES + ['RRP']).reset_index(drop=True)

m_raw = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.03,
                     subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0, verbosity=0)
m_raw.fit(train_raw[RAW_LAG_FEATURES], train_raw['RRP'])
pred_raw = m_raw.predict(test_raw[RAW_LAG_FEATURES])

actual_raw = test_raw['RRP'].values
mae_raw = mean_absolute_error(actual_raw, pred_raw)
r2_raw  = r2_score(actual_raw, pred_raw)
print(f"  Raw RRP — MAE: ${mae_raw:.2f}/MWh  R²: {r2_raw:.3f}")

# ── MODEL 2: Predict normalised RRP_NORM (final model) ───────────────────────
print("Training model on normalised RRP_NORM...")
m_norm = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.03,
                      subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0, verbosity=0)
m_norm.fit(train[PRICE_FEATURES], train[PRICE_TARGET])
pred_norm = m_norm.predict(test[PRICE_FEATURES])

actual_norm = test[PRICE_TARGET].values
mae_norm = mean_absolute_error(actual_norm, pred_norm)
r2_norm  = r2_score(actual_norm, pred_norm)
print(f"  RRP_NORM  — MAE: {mae_norm:.4f}  R²: {r2_norm:.3f}")

# ── PLOT ──────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
fig.suptitle("Price Model — Before vs After Target Normalisation\n(2022 Out-of-Sample)",
             fontsize=14, fontweight='bold')

# BEFORE — raw RRP
rng = np.random.default_rng(42)
idx = rng.choice(len(actual_raw), size=min(2000, len(actual_raw)), replace=False)
ax1.scatter(actual_raw[idx], pred_raw[idx], alpha=0.2, s=10,
            color='#E74C3C', linewidths=0)
max_val = np.percentile(np.concatenate([actual_raw, pred_raw]), 99) * 1.1
ax1.plot([0, max_val], [0, max_val], color='black', linewidth=1.5,
         linestyle='--', label='y = x (perfect)', zorder=5)
ax1.set_xlim(0, max_val)
ax1.set_ylim(0, max_val)
ax1.set_aspect('equal')
ax1.set_title('BEFORE\nPredicting raw RRP ($/MWh)\nPrice spikes compress everything else',
              fontsize=11, fontweight='bold', pad=8)
ax1.set_xlabel('Actual RRP ($/MWh)', fontsize=11)
ax1.set_ylabel('Predicted RRP ($/MWh)', fontsize=11)
ax1.text(0.05, 0.95, f'MAE = ${mae_raw:.2f}/MWh\nR²  = {r2_raw:.3f}',
         transform=ax1.transAxes, fontsize=11, fontweight='bold', va='top',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#BDC3C7'))
ax1.legend(fontsize=9, loc='lower right')
ax1.grid(alpha=0.2)
ax1.spines[['top', 'right']].set_visible(False)

n_spikes = (actual_raw > 300).sum()
ax1.text(0.05, 0.72,
         f'{n_spikes} price spikes >$300/MWh\ncompress all other predictions\ninto the bottom-left corner',
         transform=ax1.transAxes, fontsize=9, color='#E74C3C',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#FDEDEC', edgecolor='#E74C3C'))

# AFTER — normalised
idx2 = rng.choice(len(actual_norm), size=min(2000, len(actual_norm)), replace=False)
ax2.scatter(actual_norm[idx2], pred_norm[idx2], alpha=0.25, s=10,
            color='#27AE60', linewidths=0)
ax2.plot([0, 1], [0, 1], color='black', linewidth=1.5,
         linestyle='--', label='y = x (perfect)', zorder=5)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_aspect('equal')
ax2.set_title('AFTER\nPredicting RRP_NORM (0–1 scale)\n0 = cheapest hour, 1 = most expensive hour of day',
              fontsize=11, fontweight='bold', pad=8)
ax2.set_xlabel('Actual Normalised Price (0–1)', fontsize=11)
ax2.set_ylabel('Predicted Normalised Price (0–1)', fontsize=11)
ax2.text(0.05, 0.95, f'MAE = {mae_norm:.4f}\nR²  = {r2_norm:.3f}',
         transform=ax2.transAxes, fontsize=11, fontweight='bold', va='top',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#BDC3C7'))
ax2.legend(fontsize=9, loc='lower right')
ax2.grid(alpha=0.2)
ax2.spines[['top', 'right']].set_visible(False)

ax2.text(0.05, 0.72,
         'Predictions bounded 0–1\nNo spike distortion\nWorks across years with\ndifferent absolute prices',
         transform=ax2.transAxes, fontsize=9, color='#27AE60',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#EAFAF1', edgecolor='#27AE60'))


fig.tight_layout()
fig.savefig('plots/price_normalisation_scatter.png', dpi=180, bbox_inches='tight')
plt.close(fig)
print("✓ plots/price_normalisation_scatter.png")
