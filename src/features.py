import pandas as pd
import numpy as np


def add_target_history_features(
    df,
    target_col,
    group_col="route_id",
    ts_col="timestamp",
    lags=(1, 2, 4, 8, 48, 336),
    roll_windows=(4, 8, 48),
):
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.sort_values([group_col, ts_col]).copy()

    g = df.groupby(group_col, sort=False)

    # текущее известное значение target как фича
    df["target_now"] = df[target_col]

    # лаги
    for lag in lags:
        df[f"target_lag_{lag}"] = g[target_col].shift(lag)

    # rolling mean/std только по прошлому
    shifted = g[target_col].shift(1)

    for w in roll_windows:
        df[f"target_roll_mean_{w}"] = (
            shifted.groupby(df[group_col]).rolling(w).mean().reset_index(level=0, drop=True)
        )
        df[f"target_roll_std_{w}"] = (
            shifted.groupby(df[group_col]).rolling(w).std().reset_index(level=0, drop=True)
        )

    return df
    


def add_status_history_features(
    df,
    status_cols,
    group_col="route_id",
    ts_col="timestamp",
    lags=(1, 2, 4, 8, 48),
    roll_windows=(4, 8, 48),
):
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.sort_values([group_col, ts_col]).copy()

    g = df.groupby(group_col, sort=False)

    for col in status_cols:
        # лаги
        for lag in lags:
            df[f"{col}_lag_{lag}"] = g[col].shift(lag)

        # rolling только по прошлому
        shifted = g[col].shift(1)

        for w in roll_windows:
            df[f"{col}_roll_mean_{w}"] = (
                shifted.groupby(df[group_col]).rolling(w).mean().reset_index(level=0, drop=True)
            )
            df[f"{col}_roll_std_{w}"] = (
                shifted.groupby(df[group_col]).rolling(w).std().reset_index(level=0, drop=True)
            )

    return df


def add_status_aggregate_features(df):
    df = df.copy()

    df["status_out_sum"] = df["status_1"] + df["status_2"] + df["status_3"]
    df["status_in_sum"] = df["status_4"] + df["status_5"] + df["status_6"]
    df["status_total_sum"] = df["status_out_sum"] + df["status_in_sum"]
    df["status_balance"] = df["status_out_sum"] - df["status_in_sum"]

    df["status_out_share"] = df["status_out_sum"] / (df["status_total_sum"] + 1.0)
    df["status_in_share"] = df["status_in_sum"] / (df["status_total_sum"] + 1.0)
    df["status_out_in_ratio"] = df["status_out_sum"] / (df["status_in_sum"] + 1.0)

    return df


def add_calendar_features(
    df,
    ts_col="timestamp",
    forecast_points=8,
    step_minutes=30,
):
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col])

    ts = df[ts_col]

    df["source_hour"] = ts.dt.hour
    df["source_minute"] = ts.dt.minute
    df["source_halfhour_slot"] = ts.dt.hour * 2 + (ts.dt.minute // 30)
    df["source_dayofweek"] = ts.dt.dayofweek
    df["source_is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

    source_hour_float = ts.dt.hour + ts.dt.minute / 60.0
    df["source_hour_sin"] = np.sin(2 * np.pi * source_hour_float / 24.0)
    df["source_hour_cos"] = np.cos(2 * np.pi * source_hour_float / 24.0)

    df["source_dow_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7.0)
    df["source_dow_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7.0)

    step = pd.Timedelta(minutes=step_minutes)

    for h in range(1, forecast_points + 1):
        future_ts = ts + h * step

        df[f"future_hour_step_{h}"] = future_ts.dt.hour
        df[f"future_minute_step_{h}"] = future_ts.dt.minute
        df[f"future_halfhour_slot_step_{h}"] = future_ts.dt.hour * 2 + (future_ts.dt.minute // 30)
        df[f"future_dayofweek_step_{h}"] = future_ts.dt.dayofweek
        df[f"future_is_weekend_step_{h}"] = (future_ts.dt.dayofweek >= 5).astype(int)

        future_hour_float = future_ts.dt.hour + future_ts.dt.minute / 60.0
        df[f"future_hour_sin_step_{h}"] = np.sin(2 * np.pi * future_hour_float / 24.0)
        df[f"future_hour_cos_step_{h}"] = np.cos(2 * np.pi * future_hour_float / 24.0)

        df[f"future_dow_sin_step_{h}"] = np.sin(2 * np.pi * future_ts.dt.dayofweek / 7.0)
        df[f"future_dow_cos_step_{h}"] = np.cos(2 * np.pi * future_ts.dt.dayofweek / 7.0)

    return df