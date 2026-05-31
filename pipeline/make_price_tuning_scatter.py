"""
make_price_tuning_scatter.py - 3-panel price model scatter showing iteration.

Panel 1: Bad hyperparameters (underfitting — too few trees, too shallow)
Panel 2: Bad feature engineering (raw hour integer, no cyclical encoding)
Panel 3: Final tuned model (all lags + cyclical encoding + tuned params)

Usage:
    python -m pipeline.make_price_tuning_scatter
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

# ── MODEL 1: Bad hyperparameters ──────────────────────────────────────────────
# Underfitting: too few trees, too shallow, learning rate too high
print("Training Model 1: bad hyperparameters...")
m_bad_params = XGBRegressor(
    n_estimators=5,       # way too few (final uses 300)
    max_depth=1,          # stumps — can't capture interactions
    learning_rate=0.5,    # too aggressive
    verbosity=0,
)
m_bad_params.fit(train[PRICE_FEATURES], train[PRICE_TARGET])
pred_bad_params = m_bad_params.predict(test[PRICE_FEATURES])

# ── MODEL 2: Overfitting — too deep, too many trees, no regularisation ─────────
print("Training Model 2: overfitting (deep trees, no regularisation)...")
m_overfit = XGBRegressor(
    n_estimators=1000,    # far too many
    max_depth=12,         # memorises training noise
    learning_rate=0.1,
    subsample=1.0,        # no row sampling
    colsample_bytree=1.0, # no column sampling
    reg_lambda=0.0,       # no L2 regularisation
    verbosity=0,
)
m_overfit.fit(train[PRICE_FEATURES], train[PRICE_TARGET])
pred_bad_feats = m_overfit.predict(test[PRICE_FEATURES])

# ── MODEL 3: Final tuned model ─────────────────────────────────────────────────
print("Training Model 3: final tuned model...")
m_final = XGBRegressor(
    n_estimators=300, max_depth=4, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0, verbosity=0,
)
m_final.fit(train[PRICE_FEATURES], train[PRICE_TARGET])
pred_final = m_final.predict(test[PRICE_FEATURES])

# ── METRICS ───────────────────────────────────────────────────────────────────
actual = test[PRICE_TARGET].values

results = {
    'Underfitting\n(5 trees, depth=1, lr=0.5)': (pred_bad_params, '#E74C3C'),
    'Overfitting\n(1000 trees, depth=12, no regularisation)': (pred_bad_feats, '#E67E22'),
    'Final Tuned Model\n(300 trees, depth=4, L2 reg, subsampling)': (pred_final, '#27AE60'),
}

for label, (pred, _) in results.items():
    mae = mean_absolute_error(actual, pred)
    r2  = r2_score(actual, pred)
    print(f"  {label.split(chr(10))[0]:<40} MAE={mae:.4f}  R²={r2:.3f}")

# ── PLOT ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Price Model — Hyperparameter Tuning: Underfit → Overfit → Final Tuned\n(2022 Out-of-Sample)",
             fontsize=14, fontweight='bold', y=1.02)

for ax, (label, (pred, colour)) in zip(axes, results.items()):
    mae = mean_absolute_error(actual, pred)
    r2  = r2_score(actual, pred)

    idx = np.random.default_rng(42).choice(len(actual), size=min(2000, len(actual)), replace=False)
    ax.scatter(actual[idx], pred[idx], alpha=0.2, s=10, color=colour, linewidths=0)

    ax.plot([0, 1], [0, 1], color='black', linewidth=1.5,
            linestyle='--', label='y = x (perfect)', zorder=5)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')

    stats = f'MAE = {mae:.4f}\nR²  = {r2:.3f}'
    ax.text(0.05, 0.95, stats, transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#BDC3C7'))

    ax.set_title(label, fontsize=10, fontweight='bold', pad=8)
    ax.set_xlabel('Actual Normalised Price (0–1)', fontsize=10)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.2)
    ax.spines[['top', 'right']].set_visible(False)

axes[0].set_ylabel('Predicted Normalised Price (0–1)', fontsize=10)

# Bottom summary
mae_vals = [mean_absolute_error(actual, pred) for pred, _ in results.values()]
fig.text(0.5, -0.03,
         f'MAE:  {mae_vals[0]:.4f} (underfit)  →  {mae_vals[1]:.4f} (overfit)  →  {mae_vals[2]:.4f} (tuned)   '
         f'— tuning reduced error by {((max(mae_vals[0],mae_vals[1])-mae_vals[2])/max(mae_vals[0],mae_vals[1])*100):.0f}% vs worst',
         ha='center', fontsize=12, fontweight='bold', color='#27AE60')

fig.tight_layout()
fig.savefig('plots/price_tuning_scatter.png', dpi=180, bbox_inches='tight')
plt.close(fig)
print("\n✓ plots/price_tuning_scatter.png")
