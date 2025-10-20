#%%
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# For weather features
import pvlib

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
# --- 3. Feature engineering (time, target lags, rolling) ---
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
# --- 4. Weather features using pvlib (poa_total, clearsky_poa, clearsky_index) ---
# Site configuration (example: update for your true site if needed)
latitude = 46.2312
longitude = 7.3589
altitude = 500
timezone = "UTC"
tilt = 35
azimuth = 180

site = pvlib.location.Location(latitude, longitude, tz=timezone, altitude=altitude)

# Solar position
solpos = pvlib.solarposition.get_solarposition(df["time"], latitude, longitude, altitude=altitude)

# Direct/diffuse (you might need to adapt names to your dataset: here assuming you have ghi, dni, dhi columns)
dni = df["irradiance_direct"] if "irradiance_direct" in df else df["dni"] if "dni" in df else None
dhi = df["irradiance_diffuse"] if "irradiance_diffuse" in df else df["dhi"] if "dhi" in df else None
ghi = df["swgdn"] if "swgdn" in df else df["ghi"] if "ghi" in df else None

# Measured POA
poa = pvlib.irradiance.get_total_irradiance(
    surface_tilt=tilt,
    surface_azimuth=azimuth,
    solar_zenith=solpos["zenith"],
    solar_azimuth=solpos["azimuth"],
    dni=dni,
    ghi=ghi,
    dhi=dhi,
)
df["poa_total"] = poa["poa_global"]

# Clear sky - fix pandas datetime compatibility issue
times_for_clearsky = pd.DatetimeIndex(df["time"])
clearsky = site.get_clearsky(times_for_clearsky, model="ineichen")
poa_clearsky = pvlib.irradiance.get_total_irradiance(
    surface_tilt=tilt,
    surface_azimuth=azimuth,
    solar_zenith=solpos["zenith"],
    solar_azimuth=solpos["azimuth"],
    dni=clearsky["dni"],
    ghi=clearsky["ghi"],
    dhi=clearsky["dhi"],
)
df["poa_clearsky"] = poa_clearsky["poa_global"]

# Clear sky index
valid = (df["poa_clearsky"] > 50)
df["clearsky_index"] = np.where(valid, (df["poa_total"] / df["poa_clearsky"]).clip(0, 2), 0)

# Now lag/rolling for weather features
weather_feats = ["poa_total", "poa_clearsky", "clearsky_index"]
for feat in weather_feats:
    for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
        df[f"{feat}_lag{lag}"] = df[feat].shift(lag)
    for lag in [6, 12, 24, 48, 168]:
        df[f"{feat}_rollmean{lag}"] = df[feat].rolling(lag, min_periods=1).mean().shift(1)

#%%
# --- 5. Train-test split ---
train_df = df[df["time"] < forecast_start].copy()
test_df = df[(df["time"] >= forecast_start) & (df["time"] < forecast_start + pd.Timedelta(hours=168))].copy()

# 1. Only drop rows if the essential lags are NaN
essential_feats = [
    'electricity', 'electricity_lag1', 'electricity_lag24', 'hour',
    'month', 'dayofyear', 'hour_sin'
]
train_df = train_df.dropna(subset=essential_feats)

# 2. For weather lag/rolling features, fill NaN with 0 (OK for night/rare gaps)
for feat in weather_feats:
    for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
        col = f"{feat}_lag{lag}"
        if col in train_df.columns:
            train_df[col] = train_df[col].fillna(0)
    for lag in [6, 12, 24, 48, 168]:
        col = f"{feat}_rollmean{lag}"
        if col in train_df.columns:
            train_df[col] = train_df[col].fillna(0)

feature_cols = [
    # Your original 16
    'electricity_lag1', 'electricity_lag24', 'hour', 'electricity_lag48',
    'electricity_lag3', 'electricity_lag168', 'electricity_lag6', 'electricity_lag2',
    'dayofyear', 'electricity_rollmean6', 'electricity_rollmean24', 'electricity_rollmean12',
    'electricity_rollmean48', 'month', 'hour_sin', 'electricity_rollmean168'
]
# Add weather (all lags/rolls just engineered)
for feat in weather_feats:
    for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
        feature_cols.append(f"{feat}_lag{lag}")
    for lag in [6, 12, 24, 48, 168]:
        feature_cols.append(f"{feat}_rollmean{lag}")

target_col = "electricity"
X_train = train_df[feature_cols].values
y_train = train_df[target_col].values

#%%
# --- 6. Fit + Hyperparameter tuning (Bayesian optimization) ---
from skopt import BayesSearchCV
from sklearn.model_selection import TimeSeriesSplit

search_spaces = {
    'n_estimators': (100, 800),
    'max_depth': (3, 16),
    'learning_rate': (0.01, 0.2, 'log-uniform'),
    'subsample': (0.5, 1.0, 'uniform'),
    'colsample_bytree': (0.6, 1.0, 'uniform'),
    'min_child_weight': (10, 40)
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
print(f"Total training time: {end_time - start_time:.2f} seconds")
final_model = XGBRegressor(
    **best_params,
    random_state=42,
    objective="reg:squarederror",
    n_jobs=1,
)
final_model.fit(X_train, y_train)

#%%
# --- 7. Feature Importance (plot, color-coded: time=green, lag/rolling=blue, weather=orange) ---
importances = final_model.feature_importances_
imp_df = pd.DataFrame({'feature': feature_cols, 'importance': importances})
imp_df = imp_df.sort_values(by="importance", ascending=False)
top_n = 12
top_features = imp_df.head(top_n)
colors = []
for feat in top_features['feature']:
    if feat in ['hour', 'month', 'dayofyear', 'hour_sin']:
        colors.append('green')   # Time features
    elif feat.startswith('electricity'):
        colors.append('blue')    # Target lags/rollings
    else:
        colors.append('orange')  # Weather features

plt.figure(figsize=(12, 6))
plt.barh(range(len(top_features)), top_features['importance'], color=colors)
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance')
plt.title(f'Top {top_n} Feature Importances (XGBoost)')
plt.gca().invert_yaxis()
import matplotlib.patches as mpatches
patches = [
    mpatches.Patch(color='green', label='Time Features'),
    mpatches.Patch(color='blue', label='Target Lags/Rollings'),
    mpatches.Patch(color='orange', label='Weather Features'),
]
plt.legend(handles=patches, loc='lower right')
plt.tight_layout()
plt.show()

#%%
# --- 8. Operational Forecasting (using all features) ---
context = train_df.copy().reset_index(drop=True)
preds = []
true_vals = []
times = []

for h in range(168):
    curr_time = forecast_start + pd.Timedelta(hours=h)
    feat_dict = {}
    feat_dict['hour'] = curr_time.hour
    feat_dict['month'] = curr_time.month
    feat_dict['dayofyear'] = curr_time.dayofyear
    feat_dict['hour_sin'] = np.sin(2 * np.pi * curr_time.hour / 24)
    for feat in feature_cols:
        # skip if already filled
        if feat in feat_dict:
            continue
        if feat.startswith('electricity_lag'):
            lag = int(feat.split('lag')[1])
            feat_dict[feat] = context[target_col].iloc[-lag] if len(context) >= lag else 0.0
        elif feat.startswith('electricity_rollmean'):
            lag = int(feat.split('rollmean')[1])
            feat_dict[feat] = context[target_col].iloc[-lag:].mean() if len(context) >= lag else context[target_col].mean()
        elif feat.startswith('poa_total_lag'):
            lag = int(feat.split('lag')[1])
            feat_dict[feat] = context["poa_total"].iloc[-lag] if len(context) >= lag else 0.0
        elif feat.startswith('poa_total_rollmean'):
            lag = int(feat.split('rollmean')[1])
            feat_dict[feat] = context["poa_total"].iloc[-lag:].mean() if len(context) >= lag else context["poa_total"].mean()
        elif feat.startswith('poa_clearsky_lag'):
            lag = int(feat.split('lag')[1])
            feat_dict[feat] = context["poa_clearsky"].iloc[-lag] if len(context) >= lag else 0.0
        elif feat.startswith('poa_clearsky_rollmean'):
            lag = int(feat.split('rollmean')[1])
            feat_dict[feat] = context["poa_clearsky"].iloc[-lag:].mean() if len(context) >= lag else context["poa_clearsky"].mean()
        elif feat.startswith('clearsky_index_lag'):
            lag = int(feat.split('lag')[1])
            feat_dict[feat] = context["clearsky_index"].iloc[-lag] if len(context) >= lag else 0.0
        elif feat.startswith('clearsky_index_rollmean'):
            lag = int(feat.split('rollmean')[1])
            feat_dict[feat] = context["clearsky_index"].iloc[-lag:].mean() if len(context) >= lag else context["clearsky_index"].mean()
        else:
            # Handle any other weather features dynamically
            for weather_feat in weather_feats:
                if feat.startswith(f"{weather_feat}_lag"):
                    lag = int(feat.split('lag')[1])
                    feat_dict[feat] = context[weather_feat].iloc[-lag] if len(context) >= lag and weather_feat in context.columns else 0.0
                    break
                elif feat.startswith(f"{weather_feat}_rollmean"):
                    lag = int(feat.split('rollmean')[1])
                    feat_dict[feat] = context[weather_feat].iloc[-lag:].mean() if len(context) >= lag and weather_feat in context.columns else 0.0
                    break
            else:
                # If feature not found, set to 0
                feat_dict[feat] = 0.0
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
    new_row = feat_dict.copy()
    new_row["time"] = curr_time
    new_row[target_col] = y_pred
    for w in ["poa_total", "poa_clearsky", "clearsky_index"]:
        if w not in new_row:
            # Use last available value (won't be used unless rolling/lag asks for new history)
            new_row[w] = context[w].iloc[-1] if w in context and len(context) else 0.0
    context = pd.concat([context, pd.DataFrame([new_row])], ignore_index=True)

preds = np.array(preds)
true_vals = np.array(true_vals)
times = np.array(times)

#%%
# --- 9. Metrics ---
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

# --- 10. Plot ---
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
# --- 11. Daily metrics table (for the 7 days) ---
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