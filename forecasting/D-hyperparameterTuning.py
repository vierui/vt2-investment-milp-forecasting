#%%
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#%%
# --- 1. Load data ---
csv_path = "./data/renewables/dataset.csv"
df = pd.read_csv(csv_path, comment="#")
df["time"] = pd.to_datetime(df["time"]).dt.tz_localize("UTC")

#%%
# --- 2. Define time split ---
forecast_start = pd.Timestamp("2024-01-01", tz="UTC")
train_cutoff = pd.Timestamp("2021-01-01", tz="UTC")
valid_cutoff = pd.Timestamp("2023-12-24", tz="UTC")
test_end = pd.Timestamp("2024-01-08", tz="UTC")

#%%
# --- 3. Feature engineering ---
df = df.copy()
df = df.sort_values("time").reset_index(drop=True)
df["hour"] = df["time"].dt.hour
df["month"] = df["time"].dt.month
df["dayofyear"] = df["time"].dt.dayofyear
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)

lags = [1, 2, 3, 6, 12, 24, 48, 168]
for lag in lags:
    df[f"electricity_lag{lag}"] = df["electricity"].shift(lag)
for lag in [6, 12, 24, 48, 168]:
    df[f"electricity_rollmean{lag}"] = df["electricity"].rolling(lag, min_periods=1).mean().shift(1)

#%%
# --- 4. Train-test split ---
train_df = df[df["time"] < forecast_start].copy()
test_df = df[(df["time"] >= forecast_start) & (df["time"] < forecast_start + pd.Timedelta(hours=168))].copy()

required_feats = [
    'electricity_lag1', 'electricity_lag24', 'hour', 'electricity_lag48',
    'electricity_lag3', 'electricity_lag168', 'electricity_lag6', 'electricity_lag2',
    'dayofyear', 'electricity_rollmean6', 'electricity_rollmean24', 'electricity_rollmean12',
    'electricity_rollmean48', 'month', 'hour_sin', 'electricity_rollmean168', 'electricity'
]
train_df = train_df.dropna(subset=required_feats)

feature_cols = [
    'electricity_lag1', 'electricity_lag24', 'hour', 'electricity_lag48',
    'electricity_lag3', 'electricity_lag168', 'electricity_lag6', 'electricity_lag2',
    'dayofyear', 'electricity_rollmean6', 'electricity_rollmean24', 'electricity_rollmean12',
    'electricity_rollmean48', 'month', 'hour_sin', 'electricity_rollmean168'
]
target_col = "electricity"
X_train = train_df[feature_cols].values
y_train = train_df[target_col].values

#%%
# --- 5. Bayesian optimization for XGBoost hyperparameters ---
from skopt import BayesSearchCV
from sklearn.model_selection import TimeSeriesSplit

search_spaces = {
    'n_estimators': (100, 800),
    'max_depth': (3, 16),
    'learning_rate': (0.01, 0.2, 'log-uniform'),
    'subsample': (0.3, 1.0, 'uniform'),
    'colsample_bytree': (0.6, 1.0, 'uniform'),
    'min_child_weight': (1, 20)
}
tscv = TimeSeriesSplit(n_splits=3)
opt = BayesSearchCV(
    XGBRegressor(
        random_state=42, objective="reg:squarederror", n_jobs=1
    ),
    search_spaces,
    n_iter=20,
    scoring="neg_mean_absolute_error",
    cv=tscv,
    n_jobs=1,
    verbose=2,
    random_state=42
)

print("Bayesian optimization for XGBoost hyperparameters in progress...")
start_time = time.time()
opt.fit(X_train, y_train)
end_time = time.time()
best_params = opt.best_params_
print("\nBest hyperparameters (XGBoost):")
for k, v in best_params.items():
    print(f"  {k}: {v}")

final_model = XGBRegressor(
    **best_params,
    random_state=42,
    objective="reg:squarederror",
    n_jobs=1,
)
final_model.fit(X_train, y_train)

#%%
# --- 6. Operational Forecasting (using only selected features) ---
context = train_df.copy().reset_index(drop=True)
preds = []
true_vals = []
times = []

for h in range(168):
    curr_time = forecast_start + pd.Timedelta(hours=h)
    feat_dict = {}
    for feat in feature_cols:
        if feat == 'hour':
            feat_dict[feat] = curr_time.hour
        elif feat == 'month':
            feat_dict[feat] = curr_time.month
        elif feat == 'dayofyear':
            feat_dict[feat] = curr_time.dayofyear
        elif feat == 'hour_sin':
            feat_dict[feat] = np.sin(2 * np.pi * curr_time.hour / 24)
        elif feat.startswith('electricity_lag'):
            lag = int(feat.split('lag')[1])
            feat_dict[feat] = context[target_col].iloc[-lag] if len(context) >= lag else 0.0
        elif feat.startswith('electricity_rollmean'):
            lag = int(feat.split('rollmean')[1])
            if len(context) >= lag:
                roll_val = context[target_col].iloc[-lag:].mean()
            else:
                roll_val = context[target_col].mean()
            feat_dict[feat] = roll_val
    feat = np.array([[feat_dict[c] for c in feature_cols]])
    y_pred = final_model.predict(feat)[0]
    preds.append(y_pred)
    match = test_df[test_df["time"] == curr_time]
    if not match.empty:
        true_val = match[target_col].values[0]
    else:
        true_val = np.nan
    true_vals.append(true_val)
    times.append(curr_time)
    # Update context
    new_row = feat_dict.copy()
    new_row["time"] = curr_time
    new_row[target_col] = y_pred
    context = pd.concat([context, pd.DataFrame([new_row])], ignore_index=True)

preds = np.array(preds)
true_vals = np.array(true_vals)
times = np.array(times)

#%%
# --- 7. Metrics ---
mask_valid = ~np.isnan(true_vals)
y_true = true_vals[mask_valid]
y_pred = preds[mask_valid]
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
print("Operational XGBoost Forecast Metrics:")
print(f"  RMSE: {rmse:.3f}")
print(f"  MAE:  {mae:.3f}")
print(f"  R2:   {r2:.3f}")
print(f"Time taken to fit model: {end_time - start_time:.2f} seconds")

# --- 8. Plot ---
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.plot(range(48), y_true[:48], label='Actual', color='black')
plt.plot(range(48), y_pred[:48], label='Predicted', color='blue', alpha=0.7)
plt.title(f"Operational Forecast - XGBoost (First 2 Days from {forecast_start.date()})")
plt.xlabel('Hour')
plt.ylabel('Electricity (kW)')
plt.legend()
plt.grid(True, alpha=0.3)

ax = plt.subplot(1, 2, 2)
plt.scatter(y_true, y_pred, s=10, alpha=0.6, color='blue')
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title(f'Predicted vs Actual (XGBoost Operational)')
plt.grid(True, alpha=0.3)
ax.set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.show()

#%%
# --- 9. Daily metrics table (for the 7 days) ---
forecast_dates = pd.Series(times)[mask_valid].dt.date.values
results = []
for date in np.unique(forecast_dates):
    day_mask = forecast_dates == date
    y_true_day = y_true[day_mask]
    y_pred_day = y_pred[day_mask]
    if len(y_true_day) > 0:
        mae = mean_absolute_error(y_true_day, y_pred_day)
        rmse = np.sqrt(mean_squared_error(y_true_day, y_pred_day))
        r2 = r2_score(y_true_day, y_pred_day)
        results.append({"Date": str(date), "MAE": mae, "RMSE": rmse, "R2": r2})

print("\nDaily Metrics Table (XGBoost Operational):")
print("-" * 50)
print(f"{'Date':<12} {'MAE':<8} {'RMSE':<8} {'R2':<8}")
for row in results:
    print(f"{row['Date']:<12} {row['MAE']:<8.3f} {row['RMSE']:<8.3f} {row['R2']:<8.3f}")

# %%