# sarima_forecasting.py

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1 ──────────────────────────────────────────────────────────────────────────────
# Load data (same path as ML script)
path = "./data/renewables/dataset.csv"
df = pd.read_csv(path, comment="#")
df["time"] = pd.to_datetime(df["time"])
df.set_index("time", inplace=True)

TARGET = "electricity"
series = df[TARGET]

# 2 ──────────────────────────────────────────────────────────────────────────────
# Train / Validation / Test (with 7‑day forecast window & 168 h lag buffer)
max_lag = 168  # hours
forecast_start = pd.Timestamp("2024-01-01")
test_start = forecast_start - pd.Timedelta(hours=max_lag)
test_end   = forecast_start + pd.Timedelta(days=7)

train_series = series.loc[:"2020-12-31"]
valid_series = series.loc["2021-01-01":"2023-12-24"]
test_series  = series.loc[test_start:test_end - pd.Timedelta(hours=1)]

forecast_mask   = test_series.index >= forecast_start
y_test_forecast = test_series[forecast_mask]

print(f"Train size: {train_series.shape[0]}")
print(f"Valid size: {valid_series.shape[0]}")
print(f"Test  size: {y_test_forecast.shape[0]}  (forecast horizon)")

# 3 ──────────────────────────────────────────────────────────────────────────────
# Stationarity check (ADF) & differencing suggestion

def adf_report(x, title=""):
    result = adfuller(x, autolag="AIC")
    print(f"\nADF Test {title}:")
    print(f"  Test statistic : {result[0]:.3f}")
    print(f"  p‑value        : {result[1]:.4f}")
    print(f"  Lags used      : {result[2]}")
    print(f"  N obs          : {result[3]}")
    for k, v in result[4].items():
        print(f"    {k}: {v:.3f}")

adf_report(train_series, "(level)")

# First & 24‑hour seasonal differences
train_diff       = train_series.diff().dropna()
train_seas_diff  = train_series.diff(24).dropna()

adf_report(train_diff, "(first diff)")
adf_report(train_seas_diff, "(24h seasonal diff)")

# 4 ──────────────────────────────────────────────────────────────────────────────
# Quick ACF / PACF so you can eyeball p, q, P, Q
fig, axes = plt.subplots(2, 2, figsize=(12, 6))
plot_acf(train_seas_diff, lags=48, ax=axes[0, 0])
axes[0, 0].set_title("ACF (24h seasonal diff)")
plot_pacf(train_seas_diff, lags=48, ax=axes[0, 1])
axes[0, 1].set_title("PACF (24h seasonal diff)")
plot_acf(train_diff, lags=48, ax=axes[1, 0])
axes[1, 0].set_title("ACF (first diff)")
plot_pacf(train_diff, lags=48, ax=axes[1, 1])
axes[1, 1].set_title("PACF (first diff)")
plt.tight_layout()
plt.show()

# 5 ──────────────────────────────────────────────────────────────────────────────
# Baseline SARIMA model (simple & fast)
order          = (1, 1, 1)
seasonal_order = (1, 1, 1, 24)  # daily seasonality for hourly data

print(f"\nFitting baseline SARIMA{order}×{seasonal_order} …")
model_base = SARIMAX(train_series,
                     order=order,
                     seasonal_order=seasonal_order,
                     enforce_stationarity=False,
                     enforce_invertibility=False)
results_base = model_base.fit(disp=False)
print(results_base.summary())

# 6 ──────────────────────────────────────────────────────────────────────────────
# Validation prediction (one‑step ahead, no dynamics)
pred_valid   = results_base.get_prediction(start=valid_series.index[0],
                                           end=valid_series.index[-1],
                                           dynamic=False)
y_valid_pred = pred_valid.predicted_mean
y_valid      = valid_series

# Metrics helper

def metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return rmse, mae, r2

val_rmse, val_mae, val_r2 = metrics(y_valid, y_valid_pred)
print(f"\nValidation metrics → RMSE: {val_rmse:.3f}  MAE: {val_mae:.3f}  R2: {val_r2:.3f}")

# 7 ──────────────────────────────────────────────────────────────────────────────
# Re‑fit on train+valid, then forecast 7‑day horizon beginning 2024‑01‑01
train_valid_series = series.loc[:"2023-12-24"]
model_final = SARIMAX(train_valid_series,
                      order=order,
                      seasonal_order=seasonal_order,
                      enforce_stationarity=False,
                      enforce_invertibility=False)
results_final = model_final.fit(disp=False)

n_steps      = y_test_forecast.shape[0]
forecast_res = results_final.get_forecast(steps=n_steps)
y_test_pred  = forecast_res.predicted_mean
# Align indices for nice plotting / metrics
y_test_pred.index = y_test_forecast.index

# 8 ──────────────────────────────────────────────────────────────────────────────
# Test metrics
te_rmse, te_mae, te_r2 = metrics(y_test_forecast, y_test_pred)
print(f"\nTEST metrics → RMSE: {te_rmse:.3f}  MAE: {te_mae:.3f}  R2: {te_r2:.3f}")

# 9 ──────────────────────────────────────────────────────────────────────────────
# Plots (mirror ML script)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6),
                               gridspec_kw={"width_ratios": [2, 1], "height_ratios": [1]})

# Left: first 48 h horizon
n_hours_plot = 48
ax1.plot(y_test_forecast.iloc[:n_hours_plot], label="Actual", linewidth=2, color="black")
ax1.plot(y_test_pred.iloc[:n_hours_plot],    label="Predicted", alpha=0.7, color="blue")
ax1.set_title("Operational Forecast – SARIMA (first 2 days from 2024‑01‑01)")
ax1.set_xlabel("Hour")
ax1.set_ylabel("Electricity (kW)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right: scatter
ax2.scatter(y_test_forecast, y_test_pred, s=10, alpha=0.5, color="blue")
ax2.plot([y_test_forecast.min(), y_test_forecast.max()],
         [y_test_forecast.min(), y_test_forecast.max()], "r--", lw=2)
ax2.set_xlabel("Actual")
ax2.set_ylabel("Predicted")
ax2.set_title("Predicted vs Actual")
ax2.grid(True, alpha=0.3)
ax2.set_aspect("equal", adjustable="box")

plt.tight_layout()
plt.show()

print("\nOVERALL TEST METRICS:")
print(f"  RMSE: {te_rmse:.3f}")
print(f"  MAE : {te_mae:.3f}")
print(f"  R²  : {te_r2:.3f}")

# 10 ──────────────────────────────────────────────────────────────────────────────
# Daily metrics table (7‑day horizon)
forecast_df = y_test_forecast.to_frame(name="y_true")
forecast_df["y_pred"] = y_test_pred
forecast_df["date"]   = forecast_df.index.date

daily_stats = (forecast_df.groupby("date")
               .apply(lambda g: pd.Series({
                   "MAE" : mean_absolute_error(g["y_true"], g["y_pred"]),
                   "RMSE": np.sqrt(mean_squared_error(g["y_true"], g["y_pred"])),
                   "R2"  : r2_score(g["y_true"], g["y_pred"])
               })))

print("\nDaily Metrics Table (7‑Day Forecast Period):")
print("=" * 80)
print(f"{'Date':<12}{'MAE':<10}{'RMSE':<10}{'R2':<10}")
print("-" * 80)
for idx, row in daily_stats.iterrows():
    print(f"{str(idx):<12}{row['MAE']:<10.3f}{row['RMSE']:<10.3f}{row['R2']:<10.3f}")
print("=" * 80)

# 11 ──────────────────────────────────────────────────────────────────────────────
# Tiny grid‑search block 
"""
# import itertools
# p = d = q = range(0, 2)
# P = D = Q = range(0, 2)
# seasonal_period = 24
# best_aic = np.inf
# best_cfg = None
# for combo in itertools.product(p, d, q, P, D, Q):
#     order_ = combo[:3]
#     seas_order_ = combo[3:] + (seasonal_period,)
#     try:
#         res = SARIMAX(train_valid_series, order=order_, seasonal_order=seas_order_,
#                       enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
#         if res.aic < best_aic:
#             best_aic = res.aic
#             best_cfg = (order_, seas_order_)
#     except:
#         continue
# print(f"Best config by AIC → {best_cfg}  (AIC={best_aic:.2f})")
"""
#