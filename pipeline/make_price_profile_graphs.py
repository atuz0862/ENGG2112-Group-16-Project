"""
make_price_profile_graphs.py - Average normalised price profile before/after tuning.
Shows mean actual vs predicted RRP_NORM by hour of day across all 2022 test days.

BEFORE: prototype (50 trees, depth=3, lr=0.1)
AFTER:  optimised (300 trees, depth=4, lr=0.03, L2 reg, subsampling)

Usage:
    python -m pipeline.make_price_profile_graphs
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
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

# ── TRAIN BOTH MODELS ─────────────────────────────────────────────────────────
print("Training prototype model...")
m_proto = XGBRegressor(
    n_estimators=50, max_depth=3, learning_rate=0.1,
    subsample=1.0, reg_lambda=0.0, verbosity=0,
)
m_proto.fit(train[PRICE_FEATURES], train[PRICE_TARGET])
pred_proto = m_proto.predict(test[PRICE_FEATURES])

print("Training optimised model...")
m_opt = XGBRegressor(
    n_estimators=300, max_depth=4, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0, verbosity=0,
)
m_opt.fit(train[PRICE_FEATURES], train[PRICE_TARGET])
pred_opt = m_opt.predict(test[PRICE_FEATURES])

# ── BUILD HOURLY AVERAGE PROFILES ─────────────────────────────────────────────
test = test.copy()
test['PRED_PROTO'] = pred_proto
test['PRED_OPT']   = pred_opt

actual_profile = test.groupby('HOUR_OF_DAY')[PRICE_TARGET].mean()
proto_profile  = test.groupby('HOUR_OF_DAY')['PRED_PROTO'].agg(['mean', 'std'])
opt_profile    = test.groupby('HOUR_OF_DAY')['PRED_OPT'].agg(['mean', 'std'])

hours = actual_profile.index

mae_proto      = mean_absolute_error(test[PRICE_TARGET], pred_proto)
mae_opt        = mean_absolute_error(test[PRICE_TARGET], pred_opt)
shape_mae_proto = mean_absolute_error(actual_profile, proto_profile['mean'])
shape_mae_opt   = mean_absolute_error(actual_profile, opt_profile['mean'])

print(f"Prototype MAE: {mae_proto:.4f}  Shape MAE: {shape_mae_proto:.3f}")
print(f"Optimised MAE: {mae_opt:.4f}  Shape MAE: {shape_mae_opt:.3f}")

# ── PLOT FUNCTION ─────────────────────────────────────────────────────────────
def plot_profile(ax, actual, pred_mean, pred_std, mae, shape_mae,
                 model_label, model_colour, title):
    ax.plot(hours, actual, color='black', linewidth=2.2, label='Actual', zorder=5)
    ax.plot(hours, pred_mean, color=model_colour, linewidth=2,
            label=f'XGBoost (Shape MAE = {shape_mae:.3f})', zorder=4)
    ax.fill_between(hours,
                    pred_mean - pred_std * 0.5,
                    pred_mean + pred_std * 0.5,
                    color=model_colour, alpha=0.15)
    ax.set_title(
        f'Average Normalised Price Profile — Full Year 2022 (Unseen Test Data)\n'
        f'Model: {model_label}   |   Overall MAE = {mae:.4f}',
        fontsize=10, fontweight='bold', pad=8
    )
    ax.set_xlabel('Hour of Day', fontsize=10)
    ax.set_ylabel('Mean Normalised RRP', fontsize=10)
    ax.set_xlim(0, 23)
    ax.set_ylim(0.1, 0.95)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    ax.spines[['top', 'right']].set_visible(False)

# ── GRAPH 1: BEFORE ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
plot_profile(ax, actual_profile, proto_profile['mean'], proto_profile['std'],
             mae_proto, shape_mae_proto, 'Prototype', '#888888',
             'Before Hyperparameter Tuning')
fig.tight_layout()
fig.savefig('plots/price_profile_before.png', dpi=180, bbox_inches='tight')
plt.close(fig)
print("✓ plots/price_profile_before.png")

# ── GRAPH 2: AFTER ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
plot_profile(ax, actual_profile, opt_profile['mean'], opt_profile['std'],
             mae_opt, shape_mae_opt, 'Optimised', '#3498DB',
             'After Hyperparameter Tuning')
fig.tight_layout()
fig.savefig('plots/price_profile_after.png', dpi=180, bbox_inches='tight')
plt.close(fig)
print("✓ plots/price_profile_after.png")
