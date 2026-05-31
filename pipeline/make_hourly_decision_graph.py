"""
make_hourly_decision_graph.py - Stacked bar showing decision distribution by hour of day.

Usage:
    python -m pipeline.make_hourly_decision_graph
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

os.makedirs('plots', exist_ok=True)

COLOURS = {
    'SELL':  '#E74C3C',
    'STORE': '#3498DB',
    'USE':   '#2ECC71',
    'HOLD':  '#BDC3C7',
}

df = pd.read_csv('backtest/holdout_2022_decisions.csv', parse_dates=['HOUR'])
df = df.sort_values('HOUR').reset_index(drop=True)

# Count decisions per hour of day
pivot = df.groupby(['HOUR_OF_DAY', 'DECISION']).size().unstack(fill_value=0)
for d in ['SELL', 'STORE', 'USE', 'HOLD']:
    if d not in pivot.columns:
        pivot[d] = 0
pivot = pivot[['SELL', 'STORE', 'USE', 'HOLD']]

# Convert to percentage
pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(14, 6))
fig.suptitle("ML Strategy — Decision Distribution by Hour of Day (2022)",
             fontsize=14, fontweight='bold')

bottom = np.zeros(24)
for decision in ['SELL', 'STORE', 'USE', 'HOLD']:
    ax.bar(pivot_pct.index, pivot_pct[decision], bottom=bottom,
           color=COLOURS[decision], label=decision, width=0.8, alpha=0.88)
    bottom += pivot_pct[decision].values

ax.set_xlabel('Hour of Day', fontsize=12)
ax.set_ylabel('% of Days', fontsize=12)
ax.set_xticks(range(24))
ax.set_xticklabels([f'{h:02d}:00' for h in range(24)], rotation=45, ha='right', fontsize=9)
ax.set_ylim(0, 100)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
ax.grid(axis='y', alpha=0.25)
ax.spines[['top', 'right']].set_visible(False)

patches = [mpatches.Patch(color=COLOURS[d], alpha=0.88, label=d)
           for d in ['SELL', 'STORE', 'USE', 'HOLD']]
ax.legend(handles=patches, fontsize=11, loc='upper left')

fig.tight_layout()
fig.savefig('plots/hourly_decisions.png', dpi=180, bbox_inches='tight')
plt.close(fig)
print("✓ plots/hourly_decisions.png")
