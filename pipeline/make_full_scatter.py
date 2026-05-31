"""
make_full_scatter.py - Full dataset scatter: actual vs predicted solar irradiance.
Includes all 8,760 hours (day and night) with night hours shown separately.

Usage:
    python -m pipeline.make_full_scatter
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from models.solar_model import SOLAR_FEATURES, TARGET as SOLAR_TARGET
from models.price_model import engineer_price_features, PRICE_FEATURES, PRICE_TARGET

os.makedirs('plots', exist_ok=True)

print("Loading data and training solar model on 2018–2021...")
df = pd.read_csv('data/combined_engineered.csv')
df = engineer_price_features(df)
df = df.dropna(subset=SOLAR_FEATURES + PRICE_FEATURES + [SOLAR_TARGET, PRICE_TARGET])
df = df.reset_index(drop=True)

train = df[df['YEAR'] <= 2021].reset_index(drop=True)
test  = df[df['YEAR'] == 2022].reset_index(drop=True)

solar_model = XGBRegressor(
    n_estimators=100, max_depth=4, learning_rate=0.05,
    subsample=0.8, reg_lambda=1.0, verbosity=0,
)
solar_model.fit(train[SOLAR_FEATURES], train[SOLAR_TARGET])
test['SOLAR_PRED'] = solar_model.predict(test[SOLAR_FEATURES]).clip(min=0)

daylight = test[test[SOLAR_TARGET] > 50].copy()
actual = daylight[SOLAR_TARGET].values
pred   = daylight['SOLAR_PRED'].values

mae = mean_absolute_error(actual, pred)
r2  = r2_score(actual, pred)

print(f"MAE (daylight): {mae:.2f} W/m²")
print(f"R²  (daylight): {r2:.4f}")
print(f"Daylight hours: {len(daylight):,}")

fig, ax = plt.subplots(figsize=(8, 8))
fig.suptitle("Solar Model — Predicted vs Actual Irradiance\n(Daylight Hours Only, 2022 Out-of-Sample)",
             fontsize=13, fontweight='bold')

ax.scatter(actual, pred, alpha=0.3, s=10, color='#F39C12', linewidths=0,
           label=f'Daylight hours  (n={len(daylight):,})')

max_val = max(actual.max(), pred.max()) * 1.05
ax.plot([0, max_val], [0, max_val], color='#E74C3C', linewidth=2,
        linestyle='--', label='Perfect prediction (y = x)', zorder=5)

stats_text = (f'MAE: {mae:.2f} W/m²\n'
              f'R²:  {r2:.3f}')
ax.text(0.04, 0.96, stats_text, transform=ax.transAxes,
        fontsize=11, va='top', ha='left', fontweight='bold', color='#2C3E50',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#BDC3C7'))

ax.set_xlabel('Actual Irradiance (W/m²)', fontsize=12)
ax.set_ylabel('Predicted Irradiance (W/m²)', fontsize=12)
ax.set_xlim(0, max_val)
ax.set_ylim(0, max_val)
ax.set_aspect('equal')
ax.legend(fontsize=10, loc='lower right')
ax.grid(alpha=0.25)
ax.spines[['top', 'right']].set_visible(False)

fig.tight_layout()
fig.savefig('plots/solar_scatter_full.png', dpi=180, bbox_inches='tight')
plt.close(fig)
print("✓ plots/solar_scatter_full.png")
