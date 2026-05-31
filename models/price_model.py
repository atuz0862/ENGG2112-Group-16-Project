"""
price_model.py - Electricity Price Forecasting Model
Predicts normalised daily price shape (RRP_NORM: 0 = cheapest hour, 1 = most expensive).
Based on teammate's notebook: price_model_newattempt2.ipynb
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from pipeline.walk_forward_validation import walk_forward_validate, summarise

# ── FEATURES & TARGET ─────────────────────────────────────────────────────────
PRICE_FEATURES = [
    'HOUR_OF_DAY_sin', 'HOUR_OF_DAY_cos',   # cyclical hour encoding
    'DAY_OF_WEEK_sin', 'DAY_OF_WEEK_cos',   # cyclical day encoding
    'SEASON_Spring', 'SEASON_Summer', 'SEASON_Winter',   # season dummies (Autumn = reference)
    'IS_WEEKEND',
    'TOTALDEMAND',
    'TEMP_2M',
    'ALLSKY_IRRADIANCE',
    'RRP_NORM_LAG_1H',     # normalised price 1 hour ago (corr=0.74, strongest signal)
    'RRP_NORM_LAG_3H',     # normalised price 3 hours ago (corr=0.35)
    'RRP_NORM_LAG_24H',    # normalised price same hour yesterday
    'RRP_NORM_LAG_168H',   # normalised price same hour last week
    'ROLL_MEAN_24H',       # rolling 24h mean of normalised price
    'ROLL_STD_24H',        # rolling 24h std of normalised price
]

PRICE_TARGET = 'RRP_NORM'


# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────
def engineer_price_features(df):
    """
    Applies teammate's preprocessing pipeline to the combined dataset.
    Adds price-specific engineered columns without removing solar columns.

    Parameters
    ----------
    df : pd.DataFrame — combined_engineered.csv loaded as DataFrame

    Returns
    -------
    DataFrame with additional price feature columns and day_min/day_range
    for de-normalisation later.
    """
    df = df.copy()

    # Parse HOUR to datetime
    df['HOUR'] = pd.to_datetime(df['HOUR'], dayfirst=True)

    # ── Night irradiance correction ───────────────────────────────────────────
    # Hours 0–4 and 22–23 have no sunlight in Sydney — zero any non-zero values
    night_mask = df['HOUR_OF_DAY'].isin(list(range(0, 5)) + [22, 23])
    df.loc[night_mask, 'ALLSKY_IRRADIANCE'] = 0.0
    df.loc[night_mask, 'CLRSKY_IRRADIANCE'] = 0.0
    df.loc[night_mask, 'CLEARNESS_IDX']     = 0.0

    # ── Daily RRP normalisation ───────────────────────────────────────────────
    # RRP_NORM = (RRP - daily_min) / daily_range → 0 = cheapest, 1 = most expensive
    df['DATE'] = df['HOUR'].dt.date
    day_stats = df.groupby('DATE')['RRP'].agg(day_min='min', day_max='max')
    day_stats['day_range'] = day_stats['day_max'] - day_stats['day_min']
    df = df.join(day_stats, on='DATE')
    df[PRICE_TARGET] = (df['RRP'] - df['day_min']) / df['day_range'].replace(0, np.nan)

    # ── Cyclical encoding ─────────────────────────────────────────────────────
    # Maps integer hours/days onto a circle so 23→0 is one step, not 23 steps
    df['HOUR_OF_DAY_sin'] = np.sin(2 * np.pi * df['HOUR_OF_DAY'] / 24)
    df['HOUR_OF_DAY_cos'] = np.cos(2 * np.pi * df['HOUR_OF_DAY'] / 24)
    df['DAY_OF_WEEK_sin'] = np.sin(2 * np.pi * df['DAY_OF_WEEK'] / 7)
    df['DAY_OF_WEEK_cos'] = np.cos(2 * np.pi * df['DAY_OF_WEEK'] / 7)

    # ── Season dummies (Autumn = reference category, dropped) ────────────────
    season_dummies = pd.get_dummies(df['SEASON'], prefix='SEASON', dtype=int)
    for col in ['SEASON_Spring', 'SEASON_Summer', 'SEASON_Winter']:
        df[col] = season_dummies.get(col, 0)

    # ── Normalised price lag features ─────────────────────────────────────────
    df['RRP_NORM_LAG_1H']   = df[PRICE_TARGET].shift(1)
    df['RRP_NORM_LAG_3H']   = df[PRICE_TARGET].shift(3)
    df['RRP_NORM_LAG_24H']  = df[PRICE_TARGET].shift(24)
    df['RRP_NORM_LAG_168H'] = df[PRICE_TARGET].shift(168)

    # ── Rolling features on normalised price ──────────────────────────────────
    # shift(1) ensures the current hour is excluded from its own rolling window
    shifted = df[PRICE_TARGET].shift(1)
    df['ROLL_MEAN_24H'] = shifted.rolling(24, min_periods=12).mean()
    df['ROLL_STD_24H']  = shifted.rolling(24, min_periods=12).std()

    # ── Drop rows with NaN in price features ──────────────────────────────────
    df = df.dropna(subset=[PRICE_TARGET, 'RRP_NORM_LAG_1H', 'RRP_NORM_LAG_24H',
                           'RRP_NORM_LAG_168H', 'ROLL_MEAN_24H']).reset_index(drop=True)

    return df


if __name__ == "__main__":
    from models.solar_model import SOLAR_FEATURES, TARGET as SOLAR_TARGET

    print("Loading data...")
    df = pd.read_csv('data/combined_engineered.csv')
    df = engineer_price_features(df)

    df = df.dropna(subset=PRICE_FEATURES + SOLAR_FEATURES + [PRICE_TARGET, SOLAR_TARGET])
    print(f"Dataset shape after engineering: {df.shape}")

    price_model = XGBRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.8, reg_lambda=1.0, verbosity=0,
    )

    print("\nRunning walk-forward validation...")
    results = walk_forward_validate(df, PRICE_FEATURES, PRICE_TARGET, price_model)
    summarise(results, label="Price Model (RRP_NORM)")

    results.to_csv('backtest/price_wfv_results.csv', index=False)
    print("✓ Results saved to backtest/price_wfv_results.csv")
