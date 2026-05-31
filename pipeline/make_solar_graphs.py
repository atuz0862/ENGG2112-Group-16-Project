"""
make_solar_graphs.py - Solar model evaluation graphs for presentation.
Outputs 4 PNGs to plots/:
  1. solar_scatter.png      — predicted vs actual scatter
  2. solar_week.png         — time series overlay sample week
  3. solar_residuals.png    — error distribution histogram
  4. solar_mae_monthly.png  — MAE per month bar chart

Usage:
    python -m pipeline.make_solar_graphs
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from models.solar_model import SOLAR_FEATURES, TARGET as SOLAR_TARGET
from models.price_model import engineer_price_features, PRICE_FEATURES, PRICE_TARGET

os.makedirs('plots', exist_ok=True)

# ── LOAD & TRAIN ──────────────────────────────────────────────────────────────
print("Loading data and training solar model on 2018–2021...")
df = pd.read_csv('data/combined_engineered.csv')
df = engineer_price_features(df)
df = df.dropna(subset=SOLAR_FEATURES + PRICE_FEATURES + [SOLAR_TARGET, PRICE_TARGET])
df = df.reset_index(drop=True)
df['HOUR'] = pd.to_datetime(df['HOUR'])

train = df[df['YEAR'] <= 2021].reset_index(drop=True)
test  = df[df['YEAR'] == 2022].reset_index(drop=True)

solar_model = XGBRegressor(
    n_estimators=100, max_depth=4, learning_rate=0.05,
    subsample=0.8, reg_lambda=1.0, verbosity=0,
)
solar_model.fit(train[SOLAR_FEATURES], train[SOLAR_TARGET])
test['SOLAR_PRED'] = solar_model.predict(test[SOLAR_FEATURES])
test['SOLAR_PRED'] = test['SOLAR_PRED'].clip(lower=0)

mae_all      = mean_absolute_error(test[SOLAR_TARGET], test['SOLAR_PRED'])
daylight     = test[test[SOLAR_TARGET] > 50].copy()   # only hours with real solar
mae_daylight = mean_absolute_error(daylight[SOLAR_TARGET], daylight['SOLAR_PRED'])

print(f"2022 Solar MAE (all hours):      {mae_all:.2f} W/m²")
print(f"2022 Solar MAE (daylight only):  {mae_daylight:.2f} W/m²")
print(f"Night hours (trivial zeros): {(test[SOLAR_TARGET]==0).sum():,} of {len(test):,} ({(test[SOLAR_TARGET]==0).mean()*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 1 — Predicted vs Actual Scatter
# ─────────────────────────────────────────────────────────────────────────────
print("Generating graph 1: predicted vs actual scatter...")

# Use daylight hours only — night zeros make the scatter look artificially perfect
sample = daylight.sample(min(3000, len(daylight)), random_state=42)
actual = sample[SOLAR_TARGET].values
pred   = sample['SOLAR_PRED'].values

fig, ax = plt.subplots(figsize=(8, 7))
fig.suptitle("Solar Model — Predicted vs Actual Irradiance\n(Daylight Hours Only, 2022 Out-of-Sample)",
             fontsize=13, fontweight='bold')

ax.scatter(actual, pred, alpha=0.25, s=12, color='#3498DB', linewidths=0)

# Perfect prediction line
max_val = max(actual.max(), pred.max()) * 1.05
ax.plot([0, max_val], [0, max_val], color='#E74C3C', linewidth=1.8,
        linestyle='--', label='Perfect prediction (y = x)', zorder=5)

mae = mean_absolute_error(actual, pred)
ax.text(0.05, 0.92, f'MAE = {mae:.2f} W/m²\n(daylight hours only)',
        transform=ax.transAxes,
        fontsize=11, fontweight='bold', color='#2C3E50',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#BDC3C7'))

ax.set_xlabel('Actual Irradiance (W/m²)', fontsize=11)
ax.set_ylabel('Predicted Irradiance (W/m²)', fontsize=11)
ax.set_xlim(0, max_val)
ax.set_ylim(0, max_val)
ax.legend(fontsize=10)
ax.grid(alpha=0.25)
ax.spines[['top', 'right']].set_visible(False)
ax.set_aspect('equal')

fig.tight_layout()
fig.savefig('plots/solar_scatter.png', dpi=180, bbox_inches='tight')
plt.close(fig)
print("  ✓ plots/solar_scatter.png")

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 2 — Time Series Overlay: Sample Week
# ─────────────────────────────────────────────────────────────────────────────
print("Generating graph 2: time series overlay...")

# Pick a week with interesting weather variation (mix of clear and cloudy)
week_start = pd.Timestamp('2022-03-14')
week_end   = week_start + pd.Timedelta(days=7)
week = test[(test['HOUR'] >= week_start) & (test['HOUR'] < week_end)].copy()

fig, ax = plt.subplots(figsize=(13, 5))
fig.suptitle("Solar Model — Predicted vs Actual Irradiance, Sample Week (Mar 2022)",
             fontsize=13, fontweight='bold')

ax.plot(week['HOUR'], week[SOLAR_TARGET], color='#F39C12', linewidth=2,
        label='Actual irradiance', zorder=3)
ax.plot(week['HOUR'], week['SOLAR_PRED'], color='#2980B9', linewidth=1.8,
        linestyle='--', label='Predicted irradiance', zorder=4, alpha=0.9)

ax.fill_between(week['HOUR'], week[SOLAR_TARGET], week['SOLAR_PRED'],
                alpha=0.15, color='#E74C3C', label='Prediction error')

ax.set_ylabel('Solar Irradiance (W/m²)', fontsize=11)
ax.set_ylim(bottom=0)
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%a\n%d %b'))
ax.xaxis.set_major_locator(plt.matplotlib.dates.DayLocator())
ax.legend(fontsize=10, loc='upper right')
ax.grid(alpha=0.25)
ax.spines[['top', 'right']].set_visible(False)

fig.tight_layout()
fig.savefig('plots/solar_week.png', dpi=180, bbox_inches='tight')
plt.close(fig)
print("  ✓ plots/solar_week.png")

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 3 — Residual Distribution
# ─────────────────────────────────────────────────────────────────────────────
print("Generating graph 3: residual distribution...")

residuals = daylight['SOLAR_PRED'] - daylight[SOLAR_TARGET]

fig, ax = plt.subplots(figsize=(9, 6))
fig.suptitle("Solar Model — Prediction Error Distribution\n(Daylight Hours Only, 2022 Out-of-Sample)",
             fontsize=13, fontweight='bold')

ax.hist(residuals, bins=80, color='#3498DB', alpha=0.75, edgecolor='white',
        linewidth=0.4)
ax.axvline(0, color='#E74C3C', linewidth=2, linestyle='--', label='Zero error')
ax.axvline(residuals.mean(), color='#27AE60', linewidth=1.8,
           label=f'Mean error = {residuals.mean():.2f} W/m²')

ax.set_xlabel('Prediction Error (Predicted − Actual)  W/m²', fontsize=11)
ax.set_ylabel('Number of Hours', fontsize=11)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)

# Annotation box
stats_text = (f'MAE:  {mean_absolute_error(daylight[SOLAR_TARGET], daylight["SOLAR_PRED"]):.2f} W/m²\n'
              f'Std:   {residuals.std():.2f} W/m²\n'
              f'Mean: {residuals.mean():.2f} W/m²')
ax.text(0.97, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
        va='top', ha='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#BDC3C7'))

fig.tight_layout()
fig.savefig('plots/solar_residuals.png', dpi=180, bbox_inches='tight')
plt.close(fig)
print("  ✓ plots/solar_residuals.png")

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 4 — MAE per Month
# ─────────────────────────────────────────────────────────────────────────────
print("Generating graph 4: MAE per month...")

daylight['ABS_ERROR'] = (daylight['SOLAR_PRED'] - daylight[SOLAR_TARGET]).abs()
monthly = daylight.groupby('MONTH')['ABS_ERROR'].mean().reset_index()
month_names = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']
monthly['MONTH_NAME'] = monthly['MONTH'].apply(lambda m: month_names[m-1])

# Colour by season
season_colours = {
    'Summer': '#E74C3C',
    'Autumn': '#E67E22',
    'Winter': '#3498DB',
    'Spring': '#2ECC71',
}
month_season = {12:'Summer',1:'Summer',2:'Summer',
                3:'Autumn',4:'Autumn',5:'Autumn',
                6:'Winter',7:'Winter',8:'Winter',
                9:'Spring',10:'Spring',11:'Spring'}
bar_colours = [season_colours[month_season[m]] for m in monthly['MONTH']]

fig, ax = plt.subplots(figsize=(11, 6))
fig.suptitle("Solar Model — Mean Absolute Error by Month\n(Daylight Hours Only, 2022 Out-of-Sample)",
             fontsize=13, fontweight='bold')

bars = ax.bar(monthly['MONTH_NAME'], monthly['ABS_ERROR'],
              color=bar_colours, alpha=0.85, edgecolor='white', linewidth=0.5)

for bar, val in zip(bars, monthly['ABS_ERROR']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

overall_mae = mean_absolute_error(daylight[SOLAR_TARGET], daylight['SOLAR_PRED'])
ax.axhline(overall_mae, color='#2C3E50', linewidth=1.5, linestyle='--',
           label=f'Overall MAE = {overall_mae:.2f} W/m²')

# Season legend
season_patches = [plt.matplotlib.patches.Patch(color=c, alpha=0.85, label=s)
                  for s, c in season_colours.items()]
handles = season_patches + [plt.Line2D([0],[0], color='#2C3E50',
                             linewidth=1.5, linestyle='--',
                             label=f'Overall MAE = {overall_mae:.2f} W/m²')]
ax.legend(handles=handles, fontsize=10, loc='upper right')

ax.set_ylabel('Mean Absolute Error (W/m²)', fontsize=11)
ax.set_ylim(bottom=0, top=monthly['ABS_ERROR'].max() * 1.25)
ax.grid(axis='y', alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)

fig.tight_layout()
fig.savefig('plots/solar_mae_monthly.png', dpi=180, bbox_inches='tight')
plt.close(fig)
print("  ✓ plots/solar_mae_monthly.png")

print("\nAll 4 solar graphs saved to plots/")
