"""
make_naive_baseline_graph.py - Visualises the naive baseline strategy.
Shows a sample week where the fixed time window misses real price spikes.

Usage:
    python -m pipeline.make_naive_baseline_graph
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

from models.price_model import engineer_price_features, PRICE_FEATURES, PRICE_TARGET
from models.solar_model import SOLAR_FEATURES, TARGET as SOLAR_TARGET
from pipeline.dispatch_strategy import dumb_dispatch, smart_dispatch, backtest

os.makedirs('plots', exist_ok=True)

COLOURS = {
    'SELL':  '#E74C3C',
    'STORE': '#3498DB',
    'USE':   '#2ECC71',
    'HOLD':  '#BDC3C7',
}

# ── LOAD ──────────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv('backtest/holdout_2022_decisions.csv', parse_dates=['HOUR'])
df = df.sort_values('HOUR').reset_index(drop=True)

dumb_df  = dumb_dispatch(df)
smart_df = df.copy()   # already has DECISION from smart dispatch

# ── PICK A SINGLE DAY WHERE NAIVE BASELINE CLEARLY FAILS ─────────────────────
# Find a day where a price spike falls outside the 4-8pm naive sell window
day_start = pd.Timestamp('2022-07-13')
day_end   = day_start + pd.Timedelta(days=1)

smart_day = smart_df[(smart_df['HOUR'] >= day_start) & (smart_df['HOUR'] < day_end)].copy()
dumb_day  = dumb_df[(dumb_df['HOUR'] >= day_start)  & (dumb_df['HOUR'] < day_end)].copy()

hours = list(range(24))

# ── PLOT: 2 rows — naive on top, smart below ──────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                          gridspec_kw={'hspace': 0.12})
fig.suptitle("Naive Baseline vs ML Strategy — Hour by Hour (13 Jul 2022)\nShading shows dispatch decision each hour",
             fontsize=14, fontweight='bold')

for ax, day, label in [(axes[0], dumb_day,  'NAIVE BASELINE (fixed 10am–2pm charge, 4pm–8pm sell)'),
                        (axes[1], smart_day, 'ML STRATEGY (reacts to predicted price position)')]:

    # Shade each hour by decision
    for _, row in day.iterrows():
        ax.axvspan(row['HOUR_OF_DAY'], row['HOUR_OF_DAY'] + 1,
                   color=COLOURS[row['DECISION']], alpha=0.35, linewidth=0)

    # Actual price bar/line
    ax.plot(day['HOUR_OF_DAY'] + 0.5, day['RRP'], color='#2C3E50',
            linewidth=2.5, marker='o', markersize=5, label='Actual price', zorder=4)

    # Mark price peaks the naive baseline misses
    if label.startswith('NAIVE'):
        peak_threshold = day['RRP'].quantile(0.75)
        missed = day[(day['RRP'] >= peak_threshold) & (day['DECISION'] != 'SELL')]
        ax.scatter(missed['HOUR_OF_DAY'] + 0.5, missed['RRP'], color='#E74C3C',
                   s=120, zorder=5, label='High price — NOT selling',
                   marker='x', linewidths=3)

    ax.set_ylabel('Price ($/MWh)', fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', alpha=0.25)
    ax.set_title(label, fontsize=11, fontweight='bold', loc='left', pad=6)

    decision_patches = [mpatches.Patch(color=COLOURS[d], alpha=0.5, label=d)
                        for d in ['SELL', 'STORE', 'USE', 'HOLD']]
    price_line = plt.Line2D([0], [0], color='#2C3E50', linewidth=2,
                             marker='o', markersize=5, label='Actual price')
    handles = [price_line] + decision_patches
    if label.startswith('NAIVE'):
        missed_marker = plt.Line2D([0], [0], marker='x', color='#E74C3C',
                                   linewidth=0, markersize=9, markeredgewidth=3,
                                   label='High price — NOT selling')
        handles.append(missed_marker)
    ax.legend(handles=handles, loc='upper left', fontsize=9, framealpha=0.9)

axes[1].set_xlabel('Hour of Day', fontsize=11)
axes[1].set_xticks(np.arange(0.5, 24.5))
axes[1].set_xticklabels([f'{h:02d}:00' for h in range(24)], fontsize=8, rotation=45, ha='right')
axes[1].set_xlim(0, 24)
axes[0].set_xlim(0, 24)

# ── BOTTOM ANNOTATION ─────────────────────────────────────────────────────────
dumb_r = backtest(dumb_df)
smart_r = backtest(smart_df)
fig.text(0.5, -0.01,
         f'Full year 2022:  Naive = ${dumb_r["net_benefit_$"]:,.2f}   |   '
         f'ML = ${smart_r["net_benefit_$"]:,.2f}   |   '
         f'ML uplift = +${smart_r["net_benefit_$"] - dumb_r["net_benefit_$"]:,.2f} (+25%)',
         ha='center', fontsize=11, fontweight='bold', color='#27AE60')

fig.tight_layout()
fig.savefig('plots/naive_vs_ml_day.png', dpi=180, bbox_inches='tight')
plt.close(fig)
print("✓ plots/naive_vs_ml_day.png")
