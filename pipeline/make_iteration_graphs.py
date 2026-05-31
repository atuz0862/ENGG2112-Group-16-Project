"""
make_iteration_graphs.py - Shows model development iterations.
Demonstrates why each change improved results.
Does NOT modify any existing model files.

Outputs to plots/:
  1. dispatch_v1_vs_v2.png     — unnormalised vs normalised dispatch thresholds
  2. model_comparison_solar.png — XGBoost vs Linear vs Neural Network (solar)
  3. model_comparison_price.png — XGBoost vs Linear vs Neural Network (price)
  4. feature_count_solar.png   — 10 features vs 5 features solar MAE
  5. lag_feature_impact.png    — price model with vs without key lag features

Usage:
    python -m pipeline.make_iteration_graphs
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from models.solar_model import SOLAR_FEATURES, TARGET as SOLAR_TARGET
from models.price_model import engineer_price_features, PRICE_FEATURES, PRICE_TARGET
from pipeline.dispatch_strategy import backtest

warnings.filterwarnings('ignore')
os.makedirs('plots', exist_ok=True)

# ── COLOURS ───────────────────────────────────────────────────────────────────
C_XGB    = '#E74C3C'
C_LINEAR = '#3498DB'
C_NN     = '#9B59B6'
C_GOOD   = '#27AE60'
C_BAD    = '#E74C3C'

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv('data/combined_engineered.csv')
df = engineer_price_features(df)
df = df.dropna(subset=SOLAR_FEATURES + PRICE_FEATURES + [SOLAR_TARGET, PRICE_TARGET])
df = df.reset_index(drop=True)

train = df[df['YEAR'] <= 2021].reset_index(drop=True)
test  = df[df['YEAR'] == 2022].reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 1 — Dispatch V1 (unnormalised thresholds) vs V2 (normalised thresholds)
# ─────────────────────────────────────────────────────────────────────────────
print("\nGraph 1: Dispatch V1 vs V2...")

# Train both models (same as holdout_backtest)
solar_model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                            subsample=0.8, reg_lambda=1.0, verbosity=0)
price_model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.03,
                            subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0, verbosity=0)
solar_model.fit(train[SOLAR_FEATURES], train[SOLAR_TARGET])
price_model.fit(train[PRICE_FEATURES], train[PRICE_TARGET])

test = test.copy()
test['SOLAR_PRED']      = solar_model.predict(test[SOLAR_FEATURES]).clip(min=0)
test['PRICE_NORM_PRED'] = price_model.predict(test[PRICE_FEATURES])

# V1 — wrong: applied $/MWh thresholds to a 0-1 normalised prediction
# PRICE_NORM_PRED is 0-1, so comparing to 100 $/MWh means SELL never fires
# and CHARGE_THRESHOLD=30 means charge always fires → battery always stores, never sells
def dispatch_v1(df):
    """Prototype 1: confused scale — $/MWh thresholds on 0-1 normalised predictions."""
    import numpy as np
    SELL_THRESHOLD_DOLLAR   = 100.0   # $/MWh — never triggers on 0-1 scale
    CHARGE_THRESHOLD_DOLLAR = 30.0    # $/MWh — always triggers on 0-1 scale
    SOLAR_MIN = 50.0

    conditions = [
        df['PRICE_NORM_PRED'] >= SELL_THRESHOLD_DOLLAR,
        (df['SOLAR_PRED'] >= SOLAR_MIN) & (df['PRICE_NORM_PRED'] <= CHARGE_THRESHOLD_DOLLAR),
        df['SOLAR_PRED'] >= SOLAR_MIN,
    ]
    choices = ['SELL', 'STORE', 'USE']
    result = df.copy()
    result['DECISION'] = pd.Series(
        np.select(conditions, choices, default='HOLD'), index=df.index)
    return result

def dispatch_v2(df):
    """Prototype 2: correct — normalised thresholds on normalised predictions."""
    import numpy as np
    SELL_THRESHOLD   = 0.55
    CHARGE_THRESHOLD = 0.30
    SOLAR_MIN = 50.0

    conditions = [
        df['PRICE_NORM_PRED'] >= SELL_THRESHOLD,
        (df['SOLAR_PRED'] >= SOLAR_MIN) & (df['PRICE_NORM_PRED'] <= CHARGE_THRESHOLD),
        df['SOLAR_PRED'] >= SOLAR_MIN,
    ]
    choices = ['SELL', 'STORE', 'USE']
    result = df.copy()
    result['DECISION'] = pd.Series(
        np.select(conditions, choices, default='HOLD'), index=df.index)
    return result

def dispatch_dumb(df):
    result = df.copy()
    result['DECISION'] = 'HOLD'
    result.loc[df['HOUR_OF_DAY'].isin(range(10, 15)), 'DECISION'] = 'STORE'
    result.loc[df['HOUR_OF_DAY'].isin(range(16, 21)), 'DECISION'] = 'SELL'
    return result

v1_result   = backtest(dispatch_v1(test))
v2_result   = backtest(dispatch_v2(test))
dumb_result = backtest(dispatch_dumb(test))

v1_net   = v1_result['net_benefit_$']
v2_net   = v2_result['net_benefit_$']
dumb_net = dumb_result['net_benefit_$']

print(f"  V1 (wrong scale):     ${v1_net:.2f}")
print(f"  Dumb baseline:        ${dumb_net:.2f}")
print(f"  V2 (correct scale):   ${v2_net:.2f}")

labels = ['V1: ML\n($/MWh thresholds\non 0–1 predictions)',
          'Naive Baseline\n(time-of-day)',
          'V2: ML\n(normalised thresholds\non 0–1 predictions)']
values = [v1_net, dumb_net, v2_net]
colors = [C_BAD, '#7F8C8D', C_GOOD]

fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle("Dispatch Strategy Iteration — Net Benefit (2022 Out-of-Sample)",
             fontsize=13, fontweight='bold')

bars = ax.bar(labels, values, color=colors, alpha=0.85,
              edgecolor='white', linewidth=0.5, width=0.5)

for bar, val in zip(bars, values):
    y = val + (15 if val >= 0 else -40)
    ax.text(bar.get_x() + bar.get_width()/2, y,
            f'${val:,.0f}', ha='center', va='bottom' if val >= 0 else 'top',
            fontsize=12, fontweight='bold', color='#2C3E50')

ax.axhline(0, color='black', linewidth=0.8, linestyle='-')
ax.set_ylabel('Net Benefit ($)', fontsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.grid(axis='y', alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)

# Annotation arrows
ax.annotate('Bug: predictions are 0–1\nbut thresholds are $/MWh\n→ SELL never fires',
            xy=(0, v1_net), xytext=(0.3, v1_net - 150),
            fontsize=9, color='#C0392B',
            arrowprops=dict(arrowstyle='->', color='#C0392B'))
ax.annotate('Fix: use normalised\nthresholds (0–1 scale)',
            xy=(2, v2_net), xytext=(1.6, v2_net + 120),
            fontsize=9, color='#27AE60',
            arrowprops=dict(arrowstyle='->', color='#27AE60'))

fig.tight_layout()
fig.savefig('plots/dispatch_v1_vs_v2.png', dpi=180, bbox_inches='tight')
plt.close(fig)
print("  ✓ plots/dispatch_v1_vs_v2.png")

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 2 & 3 — XGBoost vs Linear Regression vs Neural Network
# ─────────────────────────────────────────────────────────────────────────────
print("\nGraph 2 & 3: Model comparison (solar + price)...")

scaler_solar = StandardScaler()
X_train_solar = scaler_solar.fit_transform(train[SOLAR_FEATURES])
X_test_solar  = scaler_solar.transform(test[SOLAR_FEATURES])

scaler_price = StandardScaler()
X_train_price = scaler_price.fit_transform(train[PRICE_FEATURES])
X_test_price  = scaler_price.transform(test[PRICE_FEATURES])

# Solar models
print("  Training solar models...")
xgb_solar = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                          subsample=0.8, reg_lambda=1.0, verbosity=0)
xgb_solar.fit(train[SOLAR_FEATURES], train[SOLAR_TARGET])

lin_solar = Ridge(alpha=1.0)
lin_solar.fit(X_train_solar, train[SOLAR_TARGET])

nn_solar = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200,
                         learning_rate_init=0.001, random_state=42,
                         early_stopping=True, validation_fraction=0.1)
nn_solar.fit(X_train_solar, train[SOLAR_TARGET])

# MAE on daylight only (>50 W/m²)
daylight_mask = test[SOLAR_TARGET] > 50
solar_mae = {
    'XGBoost\n(final model)':          mean_absolute_error(test.loc[daylight_mask, SOLAR_TARGET],
                                         np.clip(xgb_solar.predict(test[SOLAR_FEATURES])[daylight_mask], 0, None)),
    'Linear\nRegression':              mean_absolute_error(test.loc[daylight_mask, SOLAR_TARGET],
                                         np.clip(lin_solar.predict(X_test_solar)[daylight_mask], 0, None)),
    'Neural\nNetwork\n(MLP)':          mean_absolute_error(test.loc[daylight_mask, SOLAR_TARGET],
                                         np.clip(nn_solar.predict(X_test_solar)[daylight_mask], 0, None)),
}
print(f"  Solar MAEs: { {k.replace(chr(10),' '): round(v,2) for k,v in solar_mae.items()} }")

# Price models
print("  Training price models...")
xgb_price = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.03,
                          subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0, verbosity=0)
xgb_price.fit(train[PRICE_FEATURES], train[PRICE_TARGET])

lin_price = Ridge(alpha=1.0)
lin_price.fit(X_train_price, train[PRICE_TARGET])

nn_price = MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=300,
                         learning_rate_init=0.001, random_state=42,
                         early_stopping=True, validation_fraction=0.1)
nn_price.fit(X_train_price, train[PRICE_TARGET])

price_mae = {
    'XGBoost\n(final model)': mean_absolute_error(test[PRICE_TARGET], xgb_price.predict(test[PRICE_FEATURES])),
    'Linear\nRegression':     mean_absolute_error(test[PRICE_TARGET], lin_price.predict(X_test_price)),
    'Neural\nNetwork\n(MLP)': mean_absolute_error(test[PRICE_TARGET], nn_price.predict(X_test_price)),
}
print(f"  Price MAEs: { {k.replace(chr(10),' '): round(v,4) for k,v in price_mae.items()} }")

def model_comparison_chart(mae_dict, title, ylabel, filename, lower_is_better=True):
    labels = list(mae_dict.keys())
    values = list(mae_dict.values())
    best   = min(values) if lower_is_better else max(values)
    colors = [C_XGB if v == best else C_LINEAR if 'Linear' in l else C_NN
              for l, v in zip(labels, values)]

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    bars = ax.bar(labels, values, color=colors, alpha=0.85,
                  edgecolor='white', width=0.45)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                f'{val:.4f}' if val < 1 else f'{val:.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold', color='#2C3E50')

    best_idx = values.index(best)
    bars[best_idx].set_edgecolor('#2C3E50')
    bars[best_idx].set_linewidth(2)
    ax.text(bars[best_idx].get_x() + bars[best_idx].get_width()/2,
            best * 0.5, 'BEST', ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')

    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(0, max(values) * 1.3)
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

    handles = [plt.matplotlib.patches.Patch(color=C_XGB, label='XGBoost'),
               plt.matplotlib.patches.Patch(color=C_LINEAR, label='Linear Regression'),
               plt.matplotlib.patches.Patch(color=C_NN, label='Neural Network (MLP)')]
    ax.legend(handles=handles, fontsize=10)

    fig.tight_layout()
    fig.savefig(f'plots/{filename}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ plots/{filename}")

model_comparison_chart(solar_mae,
    "Solar Model — Algorithm Comparison\n(MAE on Daylight Hours, 2022 Out-of-Sample)",
    "MAE (W/m²)", "model_comparison_solar.png")

model_comparison_chart(price_mae,
    "Price Model — Algorithm Comparison\n(MAE on Normalised Price 0–1, 2022 Out-of-Sample)",
    "MAE (RRP_NORM)", "model_comparison_price.png")

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 4 — Feature Count Impact (Solar: 10 features vs 5 features)
# ─────────────────────────────────────────────────────────────────────────────
print("\nGraph 4: Feature count impact (solar)...")

ALL_SOLAR_FEATURES = [
    'CLRSKY_IRRADIANCE', 'CLEARNESS_IDX',
    'HOUR_OF_DAY', 'IS_SOLAR_HOUR',
    'SOLAR_LAG_1H', 'SOLAR_LAG_24H', 'SOLAR_LAG_168H',
    'SOLAR_ROLLING_6H_MEAN', 'SOLAR_ROLLING_24H_MAX',
    'SOLAR_ROLLING_6H_STD',
]
FINAL_SOLAR_FEATURES = SOLAR_FEATURES  # 5 features from models/solar_model.py

train_10 = train.dropna(subset=ALL_SOLAR_FEATURES + [SOLAR_TARGET]).reset_index(drop=True)
test_10  = test.dropna(subset=ALL_SOLAR_FEATURES + [SOLAR_TARGET]).reset_index(drop=True)

xgb_10 = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                        subsample=0.8, reg_lambda=1.0, verbosity=0)
xgb_10.fit(train_10[ALL_SOLAR_FEATURES], train_10[SOLAR_TARGET])

xgb_5 = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                      subsample=0.8, reg_lambda=1.0, verbosity=0)
xgb_5.fit(train[SOLAR_FEATURES], train[SOLAR_TARGET])

day_mask_10 = test_10[SOLAR_TARGET] > 50
day_mask_5  = test[SOLAR_TARGET] > 50

mae_10 = mean_absolute_error(test_10.loc[day_mask_10, SOLAR_TARGET],
                              np.clip(xgb_10.predict(test_10[ALL_SOLAR_FEATURES])[day_mask_10], 0, None))
mae_5  = mean_absolute_error(test.loc[day_mask_5, SOLAR_TARGET],
                              np.clip(xgb_5.predict(test[SOLAR_FEATURES])[day_mask_5], 0, None))

print(f"  10 features MAE: {mae_10:.2f}  |  5 features MAE: {mae_5:.2f}")

fig, ax = plt.subplots(figsize=(9, 6))
fig.suptitle("Solar Model — Feature Reduction Impact\n(MAE on Daylight Hours, 2022 Out-of-Sample)",
             fontsize=13, fontweight='bold')

feat_labels = ['Prototype\n(10 features)', 'Final Model\n(5 features)']
feat_values = [mae_10, mae_5]
feat_colors = [C_BAD if mae_10 > mae_5 else C_GOOD, C_GOOD if mae_5 < mae_10 else C_BAD]

bars = ax.bar(feat_labels, feat_values, color=feat_colors, alpha=0.85,
              edgecolor='white', width=0.4)
for bar, val in zip(bars, feat_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{val:.2f} W/m²', ha='center', va='bottom',
            fontsize=12, fontweight='bold', color='#2C3E50')

# List the removed features
removed = [f for f in ALL_SOLAR_FEATURES if f not in FINAL_SOLAR_FEATURES]
kept    = list(FINAL_SOLAR_FEATURES)
ax.text(0.97, 0.97,
        'Removed (noisy/redundant):\n' + '\n'.join(f'  ✗ {f}' for f in removed) +
        '\n\nKept (most informative):\n' + '\n'.join(f'  ✓ {f}' for f in kept),
        transform=ax.transAxes, fontsize=8, va='top', ha='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#BDC3C7'),
        family='monospace')

ax.set_ylabel('MAE (W/m²)', fontsize=11)
ax.set_ylim(0, max(feat_values) * 1.5)
ax.grid(axis='y', alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)

fig.tight_layout()
fig.savefig('plots/feature_count_solar.png', dpi=180, bbox_inches='tight')
plt.close(fig)
print("  ✓ plots/feature_count_solar.png")

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 5 — Lag Feature Impact on Price Model
# ─────────────────────────────────────────────────────────────────────────────
print("\nGraph 5: Lag feature impact (price)...")

PRICE_NO_LAGS = [f for f in PRICE_FEATURES
                 if f not in ['RRP_NORM_LAG_1H', 'RRP_NORM_LAG_3H',
                               'RRP_NORM_LAG_24H', 'RRP_NORM_LAG_168H',
                               'ROLL_MEAN_24H', 'ROLL_STD_24H']]
PRICE_SOME_LAGS = [f for f in PRICE_FEATURES
                   if f not in ['RRP_NORM_LAG_1H', 'RRP_NORM_LAG_3H']]

train_nl = train.dropna(subset=PRICE_NO_LAGS + [PRICE_TARGET]).reset_index(drop=True)
test_nl  = test.dropna(subset=PRICE_NO_LAGS + [PRICE_TARGET]).reset_index(drop=True)
train_sl = train.dropna(subset=PRICE_SOME_LAGS + [PRICE_TARGET]).reset_index(drop=True)
test_sl  = test.dropna(subset=PRICE_SOME_LAGS + [PRICE_TARGET]).reset_index(drop=True)

def train_price_xgb(feats, tr, te):
    m = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.03,
                     subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0, verbosity=0)
    m.fit(tr[feats], tr[PRICE_TARGET])
    return mean_absolute_error(te[PRICE_TARGET], m.predict(te[feats]))

mae_no_lag   = train_price_xgb(PRICE_NO_LAGS,   train_nl, test_nl)
mae_some_lag = train_price_xgb(PRICE_SOME_LAGS, train_sl, test_sl)
mae_all_lags = mean_absolute_error(test[PRICE_TARGET], xgb_price.predict(test[PRICE_FEATURES]))

print(f"  No lags:         MAE {mae_no_lag:.4f}")
print(f"  Partial lags:    MAE {mae_some_lag:.4f}")
print(f"  All lags (final):MAE {mae_all_lags:.4f}")

lag_labels = ['No Lag Features\n(time/weather only)',
              'Partial Lags\n(24h + 168h only)',
              'All Lags\n(final model)']
lag_values = [mae_no_lag, mae_some_lag, mae_all_lags]
best_lag = min(lag_values)
lag_colors = [C_GOOD if v == best_lag else ('#F39C12' if v == sorted(lag_values)[1] else C_BAD)
              for v in lag_values]

fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle("Price Model — Impact of Lag Features\n(MAE on Normalised Price 0–1, 2022 Out-of-Sample)",
             fontsize=13, fontweight='bold')

bars = ax.bar(lag_labels, lag_values, color=lag_colors, alpha=0.85,
              edgecolor='white', width=0.45)
for bar, val in zip(bars, lag_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{val:.4f}', ha='center', va='bottom',
            fontsize=12, fontweight='bold', color='#2C3E50')

ax.set_ylabel('MAE (RRP_NORM, lower is better)', fontsize=11)
ax.set_ylim(0, max(lag_values) * 1.3)
ax.grid(axis='y', alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)

ax.text(0.5, 0.85,
        'Key insight: RRP_NORM_LAG_1H has correlation 0.74\nwith target — strongest single predictor',
        transform=ax.transAxes, ha='center', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#EBF5FB', edgecolor='#3498DB'))

fig.tight_layout()
fig.savefig('plots/lag_feature_impact.png', dpi=180, bbox_inches='tight')
plt.close(fig)
print("  ✓ plots/lag_feature_impact.png")

print("\nAll 5 iteration graphs saved to plots/")
