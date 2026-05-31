"""
dispatch_strategy.py - Battery dispatch decision logic
Combines solar and price forecasts to produce hourly SELL/STORE/USE/HOLD decisions.

Expected input DataFrame columns:
    SOLAR_PRED   — predicted solar irradiance (W/m²), from solar_model.py
    PRICE_PRED   — predicted spot price ($/MWh), from teammate's price model
    HOUR_OF_DAY  — hour (0–23), used by dumb baseline

Output:
    DECISION column with values: SELL | STORE | USE | HOLD
"""

import pandas as pd

# ── THRESHOLDS (normalised 0–1 scale) ────────────────────────────────────────
# PRICE_NORM_PRED = 0 means cheapest hour of the day, 1 means most expensive
SELL_THRESHOLD   = 0.55   # sell when price is in top 45% of today's prices
CHARGE_THRESHOLD = 0.30   # store when price is in bottom 30% of today's prices
SOLAR_MIN        = 50.0   # W/m²  — minimum solar generation to act on

# ── SMART STRATEGY ────────────────────────────────────────────────────────────
def smart_dispatch(df):
    """
    Rule-based dispatch using solar and normalised price forecasts.

    Uses PRICE_NORM_PRED (0–1 scale) directly — 0 = cheapest hour of day,
    1 = most expensive hour of day. Avoids fixed $/MWh thresholds that break
    across years with different absolute price levels.

    Priority order:
      1. SELL  — hour predicted to be expensive (top 45% of today)
      2. STORE — solar generating + hour predicted to be cheap (bottom 30%)
      3. USE   — solar generating, consume directly
      4. HOLD  — no solar, not an expensive hour
    """
    conditions = [
        df['PRICE_NORM_PRED'] >= SELL_THRESHOLD,                                          # SELL
        (df['SOLAR_PRED'] >= SOLAR_MIN) & (df['PRICE_NORM_PRED'] <= CHARGE_THRESHOLD),   # STORE
        df['SOLAR_PRED'] >= SOLAR_MIN,                                                     # USE
    ]
    choices = ['SELL', 'STORE', 'USE']

    result = df.copy()
    result['DECISION'] = pd.Series(
        __import__('numpy').select(conditions, choices, default='HOLD'),
        index=df.index,
    )
    return result


# ── DUMB BASELINE ─────────────────────────────────────────────────────────────
CHARGE_HOURS = list(range(10, 15))   # 10am–2pm
SELL_HOURS   = list(range(16, 21))   # 4pm–8pm

def dumb_dispatch(df):
    """
    Time-of-day rule: charge 10am–2pm, sell 4–8pm, hold otherwise.
    No forecast inputs needed — only HOUR_OF_DAY.
    """
    result = df.copy()
    result['DECISION'] = 'HOLD'
    result.loc[df['HOUR_OF_DAY'].isin(CHARGE_HOURS), 'DECISION'] = 'STORE'
    result.loc[df['HOUR_OF_DAY'].isin(SELL_HOURS),   'DECISION'] = 'SELL'
    return result


# ── BACKTEST ──────────────────────────────────────────────────────────────────
BATTERY_CAPACITY_KWH = 10.0   # kWh
CHARGE_RATE_KW       =  5.0   # kW — max charge/discharge per hour
PANEL_AREA_M2        = 20.0   # m² — solar panel area
PANEL_EFFICIENCY     =  0.20  # 20% conversion efficiency

def backtest(df_with_decisions, actual_price_col='RRP'):
   
    battery_kwh  = 0.0
    total_revenue  = 0.0
    total_savings  = 0.0
    baseline_cost  = 0.0

    for _, row in df_with_decisions.iterrows():
        price     = row[actual_price_col] / 1000.0   # $/MWh → $/kWh
        decision  = row['DECISION']
        solar_w   = row['SOLAR_PRED']

        # Solar energy generated this hour (kWh)
        solar_kwh = (solar_w * PANEL_AREA_M2 * PANEL_EFFICIENCY) / 1000.0

        if decision == 'STORE':
            charge = min(solar_kwh, CHARGE_RATE_KW, BATTERY_CAPACITY_KWH - battery_kwh)
            battery_kwh += charge

        elif decision == 'SELL':
            discharge = min(CHARGE_RATE_KW, battery_kwh)
            battery_kwh  -= discharge
            total_revenue += discharge * price

        elif decision == 'USE':
            # Solar directly offsets 1 kWh grid purchase
            offset = min(solar_kwh, 1.0)
            total_savings += offset * price

        # Baseline always buys 1 kWh from grid each hour
        baseline_cost += 1.0 * price

    net_benefit = total_revenue + total_savings - 0.0   # storage is free (solar)

    return {
        'total_revenue_$':  round(total_revenue, 2),
        'total_savings_$':  round(total_savings, 2),
        'net_benefit_$':    round(net_benefit, 2),
        'baseline_cost_$':  round(baseline_cost, 2),
    }


# ── QUICK COMPARISON ──────────────────────────────────────────────────────────
def compare_strategies(df):
    """
    Run smart vs dumb dispatch on the same DataFrame and print a summary.
    Requires columns: SOLAR_PRED, PRICE_PRED, HOUR_OF_DAY, RRP
    """
    smart  = smart_dispatch(df)
    dumb   = dumb_dispatch(df)

    smart_result = backtest(smart)
    dumb_result  = backtest(dumb)

    print("\n" + "=" * 50)
    print("DISPATCH STRATEGY COMPARISON")
    print("=" * 50)
    for label, result in [("Smart (ML)", smart_result), ("Dumb (time-of-day)", dumb_result)]:
        print(f"\n  {label}")
        print(f"    Revenue from sales:  ${result['total_revenue_$']:>8.2f}")
        print(f"    Savings from USE:    ${result['total_savings_$']:>8.2f}")
        print(f"    Net benefit:         ${result['net_benefit_$']:>8.2f}")

    uplift = smart_result['net_benefit_$'] - dumb_result['net_benefit_$']
    print(f"\n  ML uplift over dumb baseline: ${uplift:.2f}")
    return smart_result, dumb_result
