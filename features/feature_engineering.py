"""
feature_engineering.py - Solar Generation Feature Engineering
Adds lag features, rolling stats, and calendar flags for solar model
"""

import pandas as pd
import numpy as np

def engineer_solar_features(df):
    """
    Takes raw merged dataset and adds solar-specific features.
    
    Args:
        df: raw DataFrame from merged_price_solar_rates.csv
    
    Returns:
        DataFrame with new solar features added
    """
    df = df.copy()

    # ── 1. LAG FEATURES ──────────────────────────────────────────────────────
    # What was solar doing before this hour?
    df['SOLAR_LAG_1H']   = df['ALLSKY_IRRADIANCE'].shift(1)   # 1 hour ago
    df['SOLAR_LAG_24H']  = df['ALLSKY_IRRADIANCE'].shift(24)  # same time yesterday
    df['SOLAR_LAG_168H'] = df['ALLSKY_IRRADIANCE'].shift(168) # same time last week

    # ── 2. ROLLING STATISTICS ─────────────────────────────────────────────────
    # What has solar been doing recently?
    df['SOLAR_ROLLING_6H_MEAN']  = df['ALLSKY_IRRADIANCE'].rolling(6).mean()
    df['SOLAR_ROLLING_24H_MAX']  = df['ALLSKY_IRRADIANCE'].rolling(24).max()
    df['SOLAR_ROLLING_24H_MEAN'] = df['ALLSKY_IRRADIANCE'].rolling(24).mean()
    df['SOLAR_ROLLING_6H_STD']   = df['ALLSKY_IRRADIANCE'].rolling(6).std()

    # ── 3. CALENDAR FLAGS ─────────────────────────────────────────────────────
    # Is it a time when solar generation is expected?
    df['IS_SOLAR_HOUR'] = df['HOUR_OF_DAY'].between(9, 15).astype(int)  # 9am-3pm
    df['IS_DAWN_DUSK']  = df['HOUR_OF_DAY'].isin([7, 8, 16, 17]).astype(int)
    df['IS_NIGHT']      = (df['ALLSKY_IRRADIANCE'] == 0).astype(int)

    # ── 4. DROP NULLS FROM LAGS/ROLLING ──────────────────────────────────────
    df = df.dropna(subset=[
        'SOLAR_LAG_1H',
        'SOLAR_LAG_24H', 
        'SOLAR_LAG_168H',
        'SOLAR_ROLLING_6H_MEAN',
        'SOLAR_ROLLING_24H_MAX',
    ])

    return df


if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_csv('data/merged_price_solar_rates.csv')
    print(f"Before engineering: {df.shape}")

    df_engineered = engineer_solar_features(df)
    print(f"After engineering:  {df_engineered.shape}")

    print("\nNew features added:")
    new_features = [
        'SOLAR_LAG_1H', 'SOLAR_LAG_24H', 'SOLAR_LAG_168H',
        'SOLAR_ROLLING_6H_MEAN', 'SOLAR_ROLLING_24H_MAX',
        'SOLAR_ROLLING_24H_MEAN', 'SOLAR_ROLLING_6H_STD',
        'IS_SOLAR_HOUR', 'IS_DAWN_DUSK', 'IS_NIGHT',
    ]
    for feat in new_features:
        print(f"  {feat:<30} mean: {df_engineered[feat].mean():.3f}")

    # Save engineered dataset
    df_engineered.to_csv('data/solar_engineered.csv', index=False)
    print("\n✓ Saved to data/solar_engineered.csv")
