"""
make_presentation_graphs.py - Generates presentation-quality PNG graphs.
Outputs 4 files to plots/ folder:
  1. dispatch_week.png    — one sample week: price + decisions + solar
  2. strategy_bar.png    — ML vs Dumb net benefit comparison
  3. decisions_pie.png   — SELL/STORE/USE/HOLD breakdown (2022)
  4. cumulative_gain.png — cumulative ML vs Dumb benefit over 2022

Usage:
    python -m pipeline.make_presentation_graphs
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

os.makedirs('plots', exist_ok=True)

# ── COLOURS (consistent palette) ─────────────────────────────────────────────
COLOURS = {
    'SELL':  '#E74C3C',   # red
    'STORE': '#3498DB',   # blue
    'USE':   '#2ECC71',   # green
    'HOLD':  '#BDC3C7',   # light grey
}
SOLAR_COL  = '#F39C12'   # amber
PRICE_COL  = '#2C3E50'   # dark navy

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print("Loading holdout decisions...")
df = pd.read_csv('backtest/holdout_2022_decisions.csv', parse_dates=['HOUR'])
df = df.sort_values('HOUR').reset_index(drop=True)

# Reconstruct dumb dispatch for comparison plots
from pipeline.dispatch_strategy import dumb_dispatch, backtest, smart_dispatch

dumb_df = dumb_dispatch(df)
dumb_result = backtest(dumb_df)
smart_result = backtest(df)   # df already has DECISION from smart dispatch

print(f"Smart: ${smart_result['net_benefit_$']:.2f}  |  Dumb: ${dumb_result['net_benefit_$']:.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 1 — Sample Week: Price + Decisions + Solar
# ─────────────────────────────────────────────────────────────────────────────
print("Generating graph 1: dispatch week...")

# Pick the first full week of February (interesting mix of decisions)
week_start = pd.Timestamp('2022-02-07')
week_end   = week_start + pd.Timedelta(days=7)
week = df[(df['HOUR'] >= week_start) & (df['HOUR'] < week_end)].copy()

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                          gridspec_kw={'height_ratios': [2.5, 1]})
fig.suptitle("Battery Dispatch Decisions — Sample Week (Feb 2022)",
             fontsize=15, fontweight='bold', y=0.98)

ax_price = axes[0]
ax_solar = axes[1]

# — shade background by decision ——————————————————————————————————————————————
for i, row in week.iterrows():
    ax_price.axvspan(row['HOUR'], row['HOUR'] + pd.Timedelta(hours=1),
                     color=COLOURS[row['DECISION']], alpha=0.25, linewidth=0)
    ax_solar.axvspan(row['HOUR'], row['HOUR'] + pd.Timedelta(hours=1),
                     color=COLOURS[row['DECISION']], alpha=0.25, linewidth=0)

# — actual price line —————————————————————————————————————————————————————————
ax_price.plot(week['HOUR'], week['RRP'], color=PRICE_COL, linewidth=1.8,
              label='Actual price (RRP)', zorder=3)
ax_price.set_ylabel('Electricity Price ($/MWh)', fontsize=11)
ax_price.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax_price.set_ylim(bottom=0)
ax_price.grid(axis='y', alpha=0.3)

# — solar irradiance fill ———————————————————————————————————————————————————
ax_solar.fill_between(week['HOUR'], week['SOLAR_PRED'], color=SOLAR_COL, alpha=0.7,
                       label='Predicted solar (W/m²)')
ax_solar.set_ylabel('Solar (W/m²)', fontsize=11)
ax_solar.set_ylim(bottom=0)
ax_solar.grid(axis='y', alpha=0.3)

# — legend ————————————————————————————————————————————————————————————————————
decision_patches = [mpatches.Patch(color=COLOURS[d], alpha=0.6, label=d)
                    for d in ['SELL', 'STORE', 'USE', 'HOLD']]
price_line = plt.Line2D([0], [0], color=PRICE_COL, linewidth=1.8, label='Actual price')
solar_patch = mpatches.Patch(color=SOLAR_COL, alpha=0.7, label='Predicted solar')

ax_price.legend(handles=[price_line] + decision_patches,
                loc='upper right', fontsize=9, framealpha=0.9)
ax_solar.legend(handles=[solar_patch], loc='upper right', fontsize=9)

ax_solar.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%a\n%d %b'))
fig.autofmt_xdate(rotation=0, ha='center')
fig.tight_layout()
fig.savefig('plots/dispatch_week.png', dpi=180, bbox_inches='tight')
plt.close(fig)
print("  ✓ plots/dispatch_week.png")

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 2 — ML vs Dumb Net Benefit Bar Chart
# ─────────────────────────────────────────────────────────────────────────────
print("Generating graph 2: strategy comparison bar...")

fig, ax = plt.subplots(figsize=(9, 6))
fig.suptitle("ML Strategy vs Naive Baseline — 2022 Out-of-Sample\n(Trained on 2018–2021, Never Seen 2022)",
             fontsize=13, fontweight='bold')

labels  = ['ML Strategy\n(XGBoost)', 'Naive Baseline\n(Time-of-Day)']
rev     = [smart_result['total_revenue_$'], dumb_result['total_revenue_$']]
sav     = [smart_result['total_savings_$'], dumb_result['total_savings_$']]

x = np.arange(len(labels))
w = 0.35

bars1 = ax.bar(x - w/2, rev, w, label='Revenue (SELL)', color='#E74C3C', alpha=0.85)
bars2 = ax.bar(x + w/2, sav, w, label='Savings (USE)',  color='#2ECC71', alpha=0.85)

# Net benefit annotation
for i, (r, s) in enumerate(zip(rev, sav)):
    net = r + s
    ax.annotate(f'Net: ${net:,.0f}',
                xy=(x[i], max(r, s) + 5),
                ha='center', va='bottom', fontsize=11, fontweight='bold',
                color='#2C3E50')

uplift = smart_result['net_benefit_$'] - dumb_result['net_benefit_$']
ax.set_title(f'ML uplift: +${uplift:,.2f} over naive baseline', fontsize=11,
             color='#27AE60', pad=8)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.set_ylabel('Dollar Value ($)', fontsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.legend(fontsize=10)
ax.set_ylim(bottom=0, top=max(rev + sav) * 1.25)
ax.grid(axis='y', alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)

fig.tight_layout()
fig.savefig('plots/strategy_bar.png', dpi=180, bbox_inches='tight')
plt.close(fig)
print("  ✓ plots/strategy_bar.png")

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 3 — Decision Breakdown Donut Chart
# ─────────────────────────────────────────────────────────────────────────────
print("Generating graph 3: decision breakdown donut...")

counts = df['DECISION'].value_counts()
order  = ['SELL', 'STORE', 'USE', 'HOLD']
sizes  = [counts.get(d, 0) for d in order]
colors = [COLOURS[d] for d in order]

fig, ax = plt.subplots(figsize=(8, 7))
fig.suptitle("ML Decision Breakdown — 2022 (8,760 Hours)",
             fontsize=13, fontweight='bold')

wedges, texts, autotexts = ax.pie(
    sizes, labels=None, colors=colors, autopct='%1.1f%%',
    startangle=140, pctdistance=0.75,
    wedgeprops=dict(width=0.55, edgecolor='white', linewidth=2),
)
for at in autotexts:
    at.set_fontsize(12)
    at.set_fontweight('bold')
    at.set_color('white')

legend_labels = [f"{d}  ({counts.get(d, 0):,} hrs)" for d in order]
ax.legend(wedges, legend_labels, title="Decision", loc='lower center',
          bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=11,
          title_fontsize=11, framealpha=0.9)

# Centre label
ax.text(0, 0, '2022\nOut-of-Sample', ha='center', va='center',
        fontsize=12, fontweight='bold', color='#2C3E50')

fig.tight_layout()
fig.savefig('plots/decisions_pie.png', dpi=180, bbox_inches='tight')
plt.close(fig)
print("  ✓ plots/decisions_pie.png")

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 4 — Cumulative Benefit: ML vs Dumb over 2022
# ─────────────────────────────────────────────────────────────────────────────
print("Generating graph 4: cumulative gain over 2022...")

BATTERY_CAPACITY_KWH = 10.0
CHARGE_RATE_KW       =  5.0
PANEL_AREA_M2        = 20.0
PANEL_EFFICIENCY     =  0.20

def cumulative_benefit(decisions_df, actual_price_col='RRP'):
    battery_kwh = 0.0
    cum = []
    running = 0.0
    for _, row in decisions_df.iterrows():
        price    = row[actual_price_col] / 1000.0
        decision = row['DECISION']
        solar_w  = row['SOLAR_PRED']
        solar_kwh = (solar_w * PANEL_AREA_M2 * PANEL_EFFICIENCY) / 1000.0

        hour_benefit = 0.0
        if decision == 'STORE':
            charge = min(solar_kwh, CHARGE_RATE_KW, BATTERY_CAPACITY_KWH - battery_kwh)
            battery_kwh += charge
        elif decision == 'SELL':
            discharge = min(CHARGE_RATE_KW, battery_kwh)
            battery_kwh -= discharge
            hour_benefit = discharge * price
        elif decision == 'USE':
            offset = min(solar_kwh, 1.0)
            hour_benefit = offset * price

        running += hour_benefit
        cum.append(running)
    return cum

smart_cum = cumulative_benefit(df)
dumb_cum  = cumulative_benefit(dumb_df)

fig, ax = plt.subplots(figsize=(13, 6))
fig.suptitle("Cumulative Financial Benefit — ML vs Naive Baseline (2022)",
             fontsize=13, fontweight='bold')

ax.plot(df['HOUR'], smart_cum, color='#E74C3C', linewidth=2,
        label=f"ML Strategy  (final: ${smart_cum[-1]:,.0f})")
ax.plot(df['HOUR'], dumb_cum,  color='#7F8C8D', linewidth=1.8,
        linestyle='--', label=f"Naive Baseline  (final: ${dumb_cum[-1]:,.0f})")

ax.fill_between(df['HOUR'], smart_cum, dumb_cum,
                where=[s >= d for s, d in zip(smart_cum, dumb_cum)],
                alpha=0.12, color='#27AE60', label='ML ahead')
ax.fill_between(df['HOUR'], smart_cum, dumb_cum,
                where=[s < d for s, d in zip(smart_cum, dumb_cum)],
                alpha=0.12, color='#E74C3C', label='ML behind')

uplift = smart_cum[-1] - dumb_cum[-1]
ax.set_title(f'ML uplift over full year: +${uplift:,.2f}', fontsize=11,
             color='#27AE60', pad=6)
ax.set_ylabel('Cumulative Benefit ($)', fontsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))
ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
ax.legend(fontsize=10, loc='upper left')
ax.grid(alpha=0.25)
ax.spines[['top', 'right']].set_visible(False)

fig.tight_layout()
fig.savefig('plots/cumulative_gain.png', dpi=180, bbox_inches='tight')
plt.close(fig)
print("  ✓ plots/cumulative_gain.png")

print("\nAll 4 graphs saved to plots/")
