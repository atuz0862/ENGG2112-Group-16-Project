"""
make_decision_accuracy.py - Confusion matrix and classification metrics
for the combined dispatch decision.

Ground truth: was this hour actually in the top 45% of that day's prices?
Prediction:   did the ML model decide to SELL?

Usage:
    python -m pipeline.make_decision_accuracy
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import (confusion_matrix, precision_score,
                              recall_score, f1_score, accuracy_score)
import seaborn as sns

from pipeline.dispatch_strategy import dumb_dispatch

os.makedirs('plots', exist_ok=True)

# ── LOAD ──────────────────────────────────────────────────────────────────────
print("Loading holdout decisions...")
df = pd.read_csv('backtest/holdout_2022_decisions.csv', parse_dates=['HOUR'])
df = df.sort_values('HOUR').reset_index(drop=True)
dumb_df = dumb_dispatch(df)

# ── GROUND TRUTH: actual price rank within each day ───────────────────────────
df['DATE'] = df['HOUR'].dt.date
day_stats  = df.groupby('DATE')['RRP'].transform(lambda x:
             (x - x.min()) / max(x.max() - x.min(), 1e-9))
df['ACTUAL_PRICE_RANK'] = day_stats

# Ground truth label: 1 = actually expensive (top 45% of day), 0 = not
SELL_THRESHOLD = 0.55
df['ACTUALLY_EXPENSIVE']   = (df['ACTUAL_PRICE_RANK'] >= SELL_THRESHOLD).astype(int)
dumb_df['ACTUALLY_EXPENSIVE'] = df['ACTUALLY_EXPENSIVE']

# Prediction label: 1 = model decided to SELL, 0 = didn't
df['PRED_SELL']      = (df['DECISION'] == 'SELL').astype(int)
dumb_df['PRED_SELL'] = (dumb_df['DECISION'] == 'SELL').astype(int)

y_true_ml   = df['ACTUALLY_EXPENSIVE']
y_pred_ml   = df['PRED_SELL']
y_true_dumb = dumb_df['ACTUALLY_EXPENSIVE']
y_pred_dumb = dumb_df['PRED_SELL']

def print_metrics(label, y_true, y_pred):
    print(f"\n{label}")
    print(f"  Accuracy:  {accuracy_score(y_true, y_pred):.3f}")
    print(f"  Precision: {precision_score(y_true, y_pred):.3f}")
    print(f"  Recall:    {recall_score(y_true, y_pred):.3f}")
    print(f"  F1:        {f1_score(y_true, y_pred):.3f}")

print_metrics("ML Strategy",      y_true_ml,   y_pred_ml)
print_metrics("Naive Baseline",   y_true_dumb, y_pred_dumb)

# ── PLOT: side-by-side confusion matrices ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle("SELL Decision Accuracy — ML Strategy vs Naive Baseline\n"
             "Ground truth: was this hour actually in the top 45% of that day's prices?",
             fontsize=13, fontweight='bold')

for ax, y_true, y_pred, title, colour in [
    (axes[0], y_true_dumb, y_pred_dumb, 'Naive Baseline', 'Blues'),
    (axes[1], y_true_ml,   y_pred_ml,   'ML Strategy',    'Greens'),
]:
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap=colour, ax=ax,
                linewidths=1, linecolor='white',
                annot_kws={'size': 16, 'weight': 'bold'},
                cbar=False)

    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_xticklabels(['Not Sell', 'SELL'], fontsize=11)
    ax.set_yticklabels(['Not Expensive', 'Expensive'], fontsize=11, rotation=0)

    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    acc  = accuracy_score(y_true, y_pred)

    ax.set_title(f'{title}\n'
                 f'Precision: {prec:.2f}  |  Recall: {rec:.2f}  |  '
                 f'F1: {f1:.2f}  |  Accuracy: {acc:.2f}',
                 fontsize=11, fontweight='bold', pad=12)

fig.tight_layout()
fig.savefig('plots/decision_confusion_matrix.png', dpi=180, bbox_inches='tight')
plt.close(fig)
print("\n✓ plots/decision_confusion_matrix.png")
