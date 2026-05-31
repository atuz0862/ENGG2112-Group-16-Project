# ENGG2112 — Solar Arbitrage Pipeline: Code Guide

## How to Run

```bash
# Full end-to-end pipeline (trains models, evaluates on 2022, prints all results)
python -m pipeline.holdout_backtest

# Interactive dashboard
streamlit run app.py
```

---

## Project Structure

```
Multi/
├── data/
│   └── combined_engineered.csv       # Main dataset (47,760 rows, 2018–2023)
├── features/
│   └── feature_engineering.py        # Solar feature engineering
├── models/
│   ├── solar_model.py                # Solar irradiance forecasting model
│   └── price_model.py                # Electricity price forecasting model
├── pipeline/
│   ├── holdout_backtest.py           # Main runner — train on 2018–2021, test on 2022
│   ├── run_pipeline.py               # Full pipeline runner (all years)
│   ├── dispatch_strategy.py          # Battery dispatch logic + backtest simulation
│   └── walk_forward_validation.py    # Walk-forward validation framework
├── backtest/
│   └── holdout_2022_decisions.csv    # Saved hourly decisions for 2022
└── app.py                            # Streamlit dashboard
```

---

## File Descriptions

### `features/feature_engineering.py`
Adds solar-specific engineered columns to the raw dataset.

**What it does:**
- Creates lag features: `SOLAR_LAG_1H`, `SOLAR_LAG_24H`, `SOLAR_LAG_168H`
- Creates rolling statistics: `SOLAR_ROLLING_6H_MEAN`, `SOLAR_ROLLING_6H_STD`, `SOLAR_ROLLING_24H_MAX`
- Creates calendar flags: `IS_SOLAR_HOUR`, `IS_DAWN_DUSK`, `IS_NIGHT`
- Drops first 168 rows where 168-hour lag is undefined

**Input:** Raw CSV with `ALLSKY_IRRADIANCE` column
**Output:** DataFrame with additional engineered solar columns

---

### `models/solar_model.py`
XGBoost model that predicts solar irradiance one hour ahead.

**Key variables:**
- `SOLAR_FEATURES` — list of 5 input features used by the model
- `TARGET` — `ALLSKY_IRRADIANCE` (W/m²), what the model predicts

**What it does when run directly (`python -m models.solar_model`):**
- Loads `data/combined_engineered.csv`
- Runs walk-forward validation across all years
- Prints MAE and RMSE
- Saves results to `backtest/solar_wfv_results.csv`

**Key function:** `engineer_solar_features(df)` — not present here but called via `feature_engineering.py`

**Final 5 features and why:**
| Feature | Importance | Role |
|---|---|---|
| `CLRSKY_IRRADIANCE` | 51.6% | Theoretical max solar if no clouds |
| `SOLAR_LAG_1H` | 35.9% | Irradiance one hour ago |
| `SOLAR_ROLLING_6H_STD` | 6.4% | Cloud patchiness indicator |
| `CLEARNESS_IDX` | 4.8% | Actual/clear-sky ratio — encodes cloud cover |
| `HOUR_OF_DAY` | 1.2% | Position in daily arc |

**Hyperparameters:**
```python
n_estimators=100   # number of trees
max_depth=4        # max depth per tree (prevents overfitting)
learning_rate=0.05 # step size per tree
subsample=0.8      # 80% row sampling per tree
reg_lambda=1.0     # L2 regularisation
```

---

### `models/price_model.py`
XGBoost model that predicts the relative price position of each hour within its day.

**Key variables:**
- `PRICE_FEATURES` — list of 17 input features
- `PRICE_TARGET` — `RRP_NORM` (normalised 0–1 daily price, 0=cheapest, 1=most expensive)

**Key function: `engineer_price_features(df)`**
Called before training or predicting. Adds:
- Daily min-max normalisation of RRP → `RRP_NORM`
- Cyclical sin/cos encoding of hour and day of week
- Season dummy variables (Spring, Summer, Winter; Autumn = reference)
- Lag features: `RRP_NORM_LAG_1H`, `RRP_NORM_LAG_3H`, `RRP_NORM_LAG_24H`, `RRP_NORM_LAG_168H`
- Rolling features: `ROLL_MEAN_24H`, `ROLL_STD_24H`
- Night irradiance correction (hours 0–4 and 22–23 set to zero)

**Why normalise the target:**
Raw RRP has extreme spikes (max $14,700/MWh). Normalising within each day removes spike distortion and makes thresholds robust across years with different absolute price levels.

**Hyperparameters:**
```python
n_estimators=300      # more trees than solar — prices have more complex patterns
max_depth=4           # same depth cap
learning_rate=0.03    # slower learning rate paired with more trees
subsample=0.8         # row sampling
colsample_bytree=0.7  # 70% feature sampling per tree
reg_lambda=1.0        # L2 regularisation
```

---

### `pipeline/walk_forward_validation.py`
Reusable walk-forward validation framework for time series models.

**Why not k-fold cross-validation:**
Standard k-fold randomly shuffles data, creating look-ahead bias in time series — the model would train on future data and test on past data. WFV always trains on the past and tests on the immediate future, matching real deployment conditions.

**Key function: `walk_forward_validate(df, features, target, model)`**
```
Parameters:
  df       — full DataFrame sorted by time
  features — list of feature column names
  target   — target column name
  model    — any sklearn-compatible model (XGBRegressor, etc.)
  W_train  — training window size (default 500 hours)
  W_test   — test window size (default 100 hours)
  purge    — gap between train end and test start (default 7 hours)

Returns:
  DataFrame with columns: fold, actual, predicted, absolute_error
```

**Fold structure:**
```
Fold 1: train[0:500]   → test[507:607]
Fold 2: train[100:600] → test[607:707]
...
Total: ~472 folds
```

**The purge gap:** 7 hours are removed between training and test windows to prevent lag features (e.g. `SOLAR_LAG_1H`) from leaking information across the boundary.

**Key function: `summarise(results, label)`**
Prints mean MAE, std MAE, and mean RMSE across all folds.

---

### `pipeline/dispatch_strategy.py`
Battery dispatch decision engine and backtest simulation.

**Key constants:**
```python
SELL_THRESHOLD   = 0.55   # sell when predicted price >= top 45% of today
CHARGE_THRESHOLD = 0.30   # store when predicted price <= bottom 30% of today
SOLAR_MIN        = 50.0   # W/m² — minimum solar to act on

BATTERY_CAPACITY_KWH = 10.0
CHARGE_RATE_KW       = 5.0
PANEL_AREA_M2        = 20.0
PANEL_EFFICIENCY     = 0.20
```

**`smart_dispatch(df)`**
Takes a DataFrame with `SOLAR_PRED` and `PRICE_NORM_PRED` columns.
Returns the same DataFrame with a `DECISION` column added.

Decision priority (evaluated in order):
1. `SELL` — `PRICE_NORM_PRED >= 0.55` (discharge battery at expensive hour)
2. `STORE` — `SOLAR_PRED >= 50` AND `PRICE_NORM_PRED <= 0.30` (charge during cheap solar hours)
3. `USE` — `SOLAR_PRED >= 50` (offset household consumption directly from solar)
4. `HOLD` — default (no solar, not an expensive hour)

**`dumb_dispatch(df)`**
Naive time-of-day baseline. No forecast inputs needed.
- STORE: hours 10–14 (10am–2pm)
- SELL: hours 16–20 (4pm–8pm)
- HOLD: everything else

**`backtest(df_with_decisions)`**
Simulates the battery hour by hour using actual RRP prices.
Returns a dict with `total_revenue_$`, `total_savings_$`, `net_benefit_$`, `baseline_cost_$`.

Per-hour logic:
- `STORE`: `charge = min(solar_kwh, 5kW, remaining_capacity)`
- `SELL`: `discharge = min(5kW, battery_kwh)` → `revenue += discharge × price`
- `USE`: `offset = min(solar_kwh, 1kWh)` → `savings += offset × price`

**`compare_strategies(df)`**
Runs both smart and dumb dispatch on the same DataFrame and prints a comparison summary.

---

### `pipeline/holdout_backtest.py`
The main evaluation script. Run this to see the full model results.

**What it does step by step:**
1. Loads `data/combined_engineered.csv`
2. Applies `engineer_price_features()` to add price-specific columns
3. Splits data: train = 2018–2021, test = 2022 (never seen during training)
4. Trains solar model on train set
5. Trains price model on train set
6. Predicts on 2022 test set
7. Calculates MAE and RMSE for both models
8. Runs smart and dumb dispatch on 2022 predictions
9. Prints net benefit comparison and decision breakdown
10. Saves `backtest/holdout_2022_decisions.csv`

**Expected output:**
```
Solar model — MAE: 5.58 W/m²
Price model — MAE: 0.1222

Smart (ML):   Net benefit: $944.00
Dumb (naive): Net benefit: $756.85
ML uplift:    +$187.15
```

---

### `pipeline/run_pipeline.py`
Runs the full pipeline on all available data (2018–2023) using in-sample predictions.

**Difference from holdout_backtest.py:**
`run_pipeline.py` trains and tests on the full dataset (in-sample). `holdout_backtest.py` is the honest evaluation — it trains on 2018–2021 and tests on completely unseen 2022 data.

---

### `app.py`
Streamlit interactive dashboard with 4 tabs.

**Tab 1 — 2022 Holdout Proof of Concept:**
Loads `backtest/holdout_2022_decisions.csv` and shows ML vs naive results on unseen 2022 data.

**Tab 2 — Full Pipeline (2018–2023):**
Trains models on all data and shows in-sample results with interactive date range selection.

**Tab 3 — Model Performance:**
Shows accuracy metrics table (MAE, RMSE, R², sMAPE) for both models. Walk-forward validation can be run on demand via button.

**Tab 4 — Hourly Explorer:**
Pick any date to see hour-by-hour decisions with actual price, solar forecast, and price rank forecast overlaid. Also shows decision distribution by hour of day across all of 2022.

**Run with:**
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`
