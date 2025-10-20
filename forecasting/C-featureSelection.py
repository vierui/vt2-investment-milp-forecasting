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
df["dayofweek"] = df["time"].dt.dayofweek
df["dayofyear"] = df["time"].dt.dayofyear
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

lags = [1, 2, 3, 6, 12, 24, 48, 168]
for lag in lags:
    df[f"electricity_lag{lag}"] = df["electricity"].shift(lag)
    df[f"electricity_rollmean{lag}"] = df["electricity"].rolling(lag, min_periods=1).mean().shift(1)

#%%
# --- 4. Train-test split ---
train_df = df[df["time"] < forecast_start].copy()
test_df = df[(df["time"] >= forecast_start) & (df["time"] < forecast_start + pd.Timedelta(hours=168))].copy()
lag_feats = [f"electricity_lag{lag}" for lag in lags]
roll_feats = [f"electricity_rollmean{lag}" for lag in lags]
required_feats = lag_feats + roll_feats + ["electricity"]
train_df = train_df.dropna(subset=required_feats)

feature_cols = (
    ["hour", "month", "dayofweek", "dayofyear", "hour_sin", "hour_cos"] +
    lag_feats +
    roll_feats
)
target_col = "electricity"
X_train = train_df[feature_cols].values
y_train = train_df[target_col].values

#%%
# --- 5. Bayesian Optimization: Top-k feature selection only (fixed XGB params) ---
from skopt import BayesSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import BaseEstimator, RegressorMixin

class TopKFeatureXGB(BaseEstimator, RegressorMixin):
    def __init__(self, k_features=10):
        self.k_features = k_features
        self.base_params = dict(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=1.0,
            min_child_weight=1,
            random_state=42,
            objective="reg:squarederror",
            n_jobs=1,
        )
    def fit(self, X, y):
        self.feature_cols_ = feature_cols.copy()
        # Fit a base XGB model to get importances
        model0 = XGBRegressor(**self.base_params)
        model0.fit(X, y)
        importances = model0.feature_importances_
        ranking = np.argsort(importances)[::-1]
        self.selected_idx_ = ranking[:self.k_features]
        self.selected_features_ = [self.feature_cols_[i] for i in self.selected_idx_]
        # Fit XGB again, only on the selected features
        self.model_ = XGBRegressor(**self.base_params)
        self.model_.fit(X[:, self.selected_idx_], y)
        return self
    def predict(self, X):
        return self.model_.predict(X[:, self.selected_idx_])

# Search only for k_features (number of features to use)
search_spaces = {
    'k_features': (6, min(30, len(feature_cols)))
}
tscv = TimeSeriesSplit(n_splits=3)
opt = BayesSearchCV(
    TopKFeatureXGB(),
    search_spaces,
    n_iter=15,
    scoring="neg_mean_absolute_error",
    cv=tscv,
    n_jobs=1,
    verbose=2,
    random_state=42
)

print("Bayesian optimization for number of features in progress...")
opt.fit(X_train, y_train)
best_k = opt.best_params_['k_features']
print(f"\nOptimal number of features: {best_k}")

# Re-train final model with only the selected features found
final_model = opt.best_estimator_
selected_idx = final_model.selected_idx_
selected_features = final_model.selected_features_
print("Selected features:", selected_features)

#%%
# --- 6. Plot feature importance of final model, color coded (top 8) ---
# Fit a new model to full training set, for plotting
model_for_plot = XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=1.0,
    min_child_weight=1,
    random_state=42,
    objective="reg:squarederror",
    n_jobs=1,
)
start_time = time.time()
model_for_plot.fit(X_train, y_train)
end_time = time.time()
importances = model_for_plot.feature_importances_
imp_df = pd.DataFrame({'feature': feature_cols, 'importance': importances})
imp_df = imp_df.sort_values(by="importance", ascending=False)
top_features = imp_df.head(8)

colors = []
for feat in top_features['feature']:
    if any(t in feat for t in ['hour', 'month', 'day']):
        colors.append('green')   # Time features
    elif any(t in feat for t in ['lag', 'rollmean']):
        colors.append('blue')    # Target lags/rolls
    else:
        colors.append('yellow')  # (not expected)

plt.figure(figsize=(12, 6))
plt.barh(range(len(top_features)), top_features['importance'], color=colors)
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance')
plt.title('Top 8 Feature Importances - Feature Selection Optimized Model (Time+Lag Features)')
plt.gca().invert_yaxis()
import matplotlib.patches as mpatches
patches = [
    mpatches.Patch(color='green', label='Time Features'),
    mpatches.Patch(color='blue', label='Target Lags/Rollings'),
]
plt.legend(handles=patches, loc='lower right')
plt.tight_layout()
plt.show()

#%%
# --- 7. Operational Forecasting (using only selected features) ---
context = train_df.copy().reset_index(drop=True)
preds = []
true_vals = []
times = []

for h in range(168):
    curr_time = forecast_start + pd.Timedelta(hours=h)
    # Build dict of all feature values
    feat_dict = {}
    feat_dict["hour"] = curr_time.hour
    feat_dict["month"] = curr_time.month
    feat_dict["dayofweek"] = curr_time.dayofweek
    feat_dict["dayofyear"] = curr_time.dayofyear
    feat_dict["hour_sin"] = np.sin(2 * np.pi * curr_time.hour / 24)
    feat_dict["hour_cos"] = np.cos(2 * np.pi * curr_time.hour / 24)
    for lag in lags:
        feat_dict[f"electricity_lag{lag}"] = context[target_col].iloc[-lag] if len(context) >= lag else 0.0
    for lag in lags:
        if len(context) >= lag:
            roll_val = context[target_col].iloc[-lag:].mean()
        else:
            roll_val = context[target_col].mean()
        feat_dict[f"electricity_rollmean{lag}"] = roll_val
    # Build feature vector for selected features only
    feat = np.array([[feat_dict[c] for c in selected_features]])
    y_pred = final_model.model_.predict(feat)[0]
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
# --- 8. Metrics ---
mask_valid = ~np.isnan(true_vals)
y_true = true_vals[mask_valid]
y_pred = preds[mask_valid]
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
print("Operational XGBoost Forecast Metrics (Feature selection):")
print(f"  RMSE: {rmse:.3f}")
print(f"  MAE:  {mae:.3f}")
print(f"  R2:   {r2:.3f}")
# print(f"Time taken to fit model: {end_time - start_time:.2f} seconds")

# --- 9. Plot ---
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
plt.plot([0, 0.6], [0, 0.6], 'r--', lw=2)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title(f'Predicted vs Actual (XGBoost Operational)')
plt.grid(True, alpha=0.3)

# Set axis range from 0 to 0.6 for both axes
ax.set_xlim(-0.05, 0.65)
ax.set_ylim(-0.05, 0.65)
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.show()

#%%
# --- 10. Daily metrics table (for the 7 days) ---
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

print("\nDaily Metrics Table (XGBoost Operational, Feature Selection):")
print("-" * 50)
print(f"{'Date':<12} {'MAE':<8} {'RMSE':<8} {'R2':<8}")
for row in results:
    print(f"{row['Date']:<12} {row['MAE']:<8.3f} {row['RMSE']:<8.3f} {row['R2']:<8.3f}")

# %%