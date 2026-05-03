from pathlib import Path
import sys

import yaml
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.features import add_target_history_features, add_status_history_features
from src.validation import prepare_backtest_frame
from src.models import make_ridge_model


with open(ROOT_DIR / "configs" / "final_ridge.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)


# =========================
# CONFIG
# =========================
TRAIN_PATH = ROOT_DIR / config["data"]["train_path"]
TEST_PATH = ROOT_DIR / config["data"]["test_path"]
SUBMISSION_DIR = ROOT_DIR / config["data"]["submission_dir"]
SUBMISSION_NAME = config["data"]["submission_name"]

ALPHA = config["model"]["alpha"]
TRAIN_WINDOW_DAYS = config["model"]["train_window_days"]
STEP_MINUTES = config["model"]["step_minutes"]
FORECAST_POINTS = config["model"]["forecast_horizon"]

SELECTED_FEATURES = config["features"]["selected_features"]
GLOBAL_SCALE = config["calibration"]["global_scale"]

TARGET_COL = "target_1h"
FUTURE_TARGET_COLS = [f"target_step_{step}" for step in range(1, FORECAST_POINTS + 1)]
STATUS_COLS = ["status_1", "status_2", "status_3", "status_4", "status_5", "status_6"]


# =========================
# LOAD DATA
# =========================
train_df = pd.read_parquet(TRAIN_PATH, engine="fastparquet")
test_df = pd.read_parquet(TEST_PATH, engine="fastparquet")

train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])

train_df = train_df.sort_values(["route_id", "timestamp"]).reset_index(drop=True)
test_df = test_df.sort_values(["route_id", "timestamp"]).reset_index(drop=True)


# =========================
# FEATURE ENGINEERING
# =========================
train_df = add_target_history_features(
    df=train_df,
    target_col=TARGET_COL,
    group_col="route_id",
    ts_col="timestamp",
    lags=(1, 2, 4, 8, 48, 336),
    roll_windows=(4, 8, 48),
)

train_df = add_status_history_features(
    df=train_df,
    status_cols=STATUS_COLS,
    group_col="route_id",
    ts_col="timestamp",
    lags=(1, 2, 4, 8, 48),
    roll_windows=(4, 8, 48),
)

route_group = train_df.groupby("route_id", sort=False)

for step in range(1, FORECAST_POINTS + 1):
    train_df[f"target_step_{step}"] = route_group[TARGET_COL].shift(-step)

supervised_df = train_df.dropna(subset=FUTURE_TARGET_COLS).copy()


# =========================
# TRAIN FRAME
# =========================
train_model_df = prepare_backtest_frame(
    supervised_df=supervised_df,
    feature_cols=SELECTED_FEATURES,
    future_target_cols=FUTURE_TARGET_COLS,
)

train_ts_max = train_model_df["source_timestamp"].max()
train_window_start = train_ts_max - pd.Timedelta(days=TRAIN_WINDOW_DAYS)

fit_df = train_model_df[
    (train_model_df["source_timestamp"] >= train_window_start) &
    (train_model_df["source_timestamp"] <= train_ts_max)
].copy()

X_fit = fit_df[SELECTED_FEATURES].copy()
y_fit = fit_df[FUTURE_TARGET_COLS].copy()


# =========================
# INFERENCE FRAME
# =========================
inference_ts = train_df["timestamp"].max()

test_inference_df = train_df[
    train_df["timestamp"] == inference_ts
].copy()

X_test = test_inference_df[SELECTED_FEATURES].copy()


# =========================
# TRAIN + PREDICT
# =========================
final_model = make_ridge_model(SELECTED_FEATURES, alpha=ALPHA)
final_model.fit(X_fit, y_fit)

test_pred_raw = np.asarray(final_model.predict(X_test), dtype=float)
test_pred_final = np.clip(test_pred_raw, 0.0, None)
test_pred_final = test_pred_final * GLOBAL_SCALE
test_pred_final = np.clip(test_pred_final, 0.0, None)


# =========================
# BUILD SUBMISSION
# =========================
test_pred_df = pd.DataFrame(
    test_pred_final,
    columns=FUTURE_TARGET_COLS,
    index=test_inference_df.index,
)
test_pred_df["route_id"] = test_inference_df["route_id"].values

forecast_df = test_pred_df.melt(
    id_vars="route_id",
    value_vars=FUTURE_TARGET_COLS,
    var_name="step",
    value_name="forecast",
)

forecast_df["step_num"] = forecast_df["step"].str.extract(r"(\d+)").astype(int)
forecast_df["timestamp"] = (
    inference_ts + pd.to_timedelta(forecast_df["step_num"] * STEP_MINUTES, unit="m")
)

forecast_df = forecast_df[["route_id", "timestamp", "forecast"]].sort_values(
    ["route_id", "timestamp"]
).reset_index(drop=True)

submission_df = test_df.merge(
    forecast_df,
    how="left",
    on=["route_id", "timestamp"],
)[["id", "forecast"]].rename(columns={"forecast": "y_pred"})

if submission_df["y_pred"].isna().any():
    missing_count = int(submission_df["y_pred"].isna().sum())
    raise ValueError(f"Submission contains {missing_count} missing predictions.")


# =========================
# SAVE
# =========================
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

submission_path = SUBMISSION_DIR / SUBMISSION_NAME
submission_df.to_csv(submission_path, index=False)

print("Submission saved to:", submission_path)
print(submission_df.head())
