"""
make_naive_only_graph.py - Naive baseline hour-by-hour decision graph for one day.

Usage:
    python -m pipeline.make_naive_only_graph
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from pipeline.dispatch_strategy import dumb_dispatch

os.makedirs('plots', exist_ok=True)

COLOURS = {
    'SELL':  '#E74C3C',
    'STORE': '#3498DB',
    'USE':   '#2ECC71',
    'HOLD':  '#BDC3C7',
}

df = pd.read_csv('backtest/holdout_2022_decisions.csv', parse_dates=['HOUR'])
df = df.sort_values('HOUR').reset_index(drop=True)
dumb_df = dumb_dispatch(df)

day_start = pd.Timestamp('2022-07-13')
day_end   = day_start + pd.Timedelta(days=1)
day = dumb_df[(dumb_df['HOUR'] >= day_start) & (dumb_df['HOUR'] < day_end)].copy()

fig, ax = plt.subplots(figsize=(14, 6))
fig.suptitle("Naive Baseline — Hour by Hour Dispatch Decisions (13 Jul 2022)",
             fontsize=14, fontweight='bold')

# Shade each hour by decision
for _, row in day.iterrows():
    ax.axvspan(row['HOUR_OF_DAY'], row['HOUR_OF_DAY'] + 1,
               color=COLOURS[row['DECISION']], alpha=0.35, linewidth=0)

# Price line
ax.plot(day['HOUR_OF_DAY'] + 0.5, day['RRP'], color='#2C3E50',
        linewidth=2.5, marker='o', markersize=6, label='Actual price ($/MWh)', zorder=4)

# Annotate the fixed windows
ax.axvspan(10, 15, color='none', linewidth=2,
           edgecolor='#3498DB', linestyle='--', zorder=5,
           label='Fixed STORE window (10am–2pm)')
ax.axvspan(16, 21, color='none', linewidth=2,
           edgecolor='#E74C3C', linestyle='--', zorder=5,
           label='Fixed SELL window (4pm–8pm)')

ax.set_xlabel('Hour of Day', fontsize=12)
ax.set_ylabel('Price ($/MWh)', fontsize=12)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.set_ylim(bottom=0)
ax.set_xlim(0, 24)
ax.set_xticks(np.arange(0.5, 24.5))
ax.set_xticklabels([f'{h:02d}:00' for h in range(24)], fontsize=9, rotation=45, ha='right')
ax.grid(axis='y', alpha=0.25)
ax.spines[['top', 'right']].set_visible(False)

decision_patches = [mpatches.Patch(color=COLOURS[d], alpha=0.5, label=d)
                    for d in ['SELL', 'STORE', 'USE', 'HOLD']]
price_line = plt.Line2D([0], [0], color='#2C3E50', linewidth=2.5,
                         marker='o', markersize=6, label='Actual price')
sell_box   = mpatches.Patch(fill=False, edgecolor='#E74C3C',
                              linestyle='--', linewidth=2, label='Fixed SELL window (4pm–8pm)')
store_box  = mpatches.Patch(fill=False, edgecolor='#3498DB',
                              linestyle='--', linewidth=2, label='Fixed STORE window (10am–2pm)')

ax.legend(handles=[price_line] + decision_patches + [store_box, sell_box],
          fontsize=10, loc='upper right', framealpha=0.9)

fig.tight_layout()
fig.savefig('plots/naive_baseline_day.png', dpi=180, bbox_inches='tight')
plt.close(fig)
print("✓ plots/naive_baseline_day.png")
