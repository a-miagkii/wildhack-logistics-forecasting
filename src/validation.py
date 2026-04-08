import numpy as np
import pandas as pd
from sklearn.base import clone

from src.metrics import metric_components, metric_by_horizon


def prepare_backtest_frame(supervised_df, feature_cols, future_target_cols):
    df = supervised_df[feature_cols + ["timestamp"] + future_target_cols].copy()
    df = df.rename(columns={"timestamp": "source_timestamp"})
    df["source_timestamp"] = pd.to_datetime(df["source_timestamp"])
    df = df.sort_values(["source_timestamp"]).reset_index(drop=True)
    return df


def make_cutoffs(
    train_model_df,
    future_target_cols,
    reference_ts,
    train_days=14,
    step_minutes=30,
    n_folds=5,
    same_weekday=False,
):
    gap = pd.Timedelta(minutes=len(future_target_cols) * step_minutes)
    min_ts = train_model_df["source_timestamp"].min()

    reference_ts = pd.Timestamp(reference_ts)
    all_ts = sorted(train_model_df["source_timestamp"].unique())

    cutoffs = []
    for ts in all_ts:
        ts = pd.Timestamp(ts)

        train_end = ts - gap
        train_start = train_end - pd.Timedelta(days=train_days)

        if train_start < min_ts:
            continue

        if ts.time() != reference_ts.time():
            continue

        if same_weekday and ts.dayofweek != reference_ts.dayofweek:
            continue

        cutoffs.append(ts)

    return cutoffs[-n_folds:]


def make_fold(
    train_model_df,
    feature_cols,
    future_target_cols,
    cutoff,
    train_days=14,
    step_minutes=30,
):
    gap = pd.Timedelta(minutes=len(future_target_cols) * step_minutes)
    cutoff = pd.Timestamp(cutoff)

    train_end = cutoff - gap
    train_start = train_end - pd.Timedelta(days=train_days)

    fit_df = train_model_df[
        (train_model_df["source_timestamp"] >= train_start) &
        (train_model_df["source_timestamp"] <= train_end)
    ].copy()

    valid_df = train_model_df[
        train_model_df["source_timestamp"] == cutoff
    ].copy()

    if fit_df.empty:
        raise ValueError(f"Empty fit_df for cutoff={cutoff}")
    if valid_df.empty:
        raise ValueError(f"Empty valid_df for cutoff={cutoff}")

    X_fit = fit_df[feature_cols].copy()
    y_fit = fit_df[future_target_cols].copy()

    X_valid = valid_df[feature_cols].copy()
    y_valid = valid_df[future_target_cols].copy()

    return X_fit, y_fit, X_valid, y_valid, fit_df, valid_df


def run_backtest(
    model,
    train_model_df,
    feature_cols,
    future_target_cols,
    cutoffs,
    train_days=14,
    step_minutes=30,
    clip_lower=0.0,
):
    rows = []

    for i, cutoff in enumerate(cutoffs, start=1):
        X_fit, y_fit, X_valid, y_valid, fit_df, valid_df = make_fold(
            train_model_df=train_model_df,
            feature_cols=feature_cols,
            future_target_cols=future_target_cols,
            cutoff=cutoff,
            train_days=train_days,
            step_minutes=step_minutes,
        )

        fold_model = clone(model)
        fold_model.fit(X_fit, y_fit)

        pred = fold_model.predict(X_valid)
        pred = np.asarray(pred, dtype=float)
        pred = np.clip(pred, clip_lower, None)

        y_true_2d = y_valid.to_numpy(dtype=float)
        y_pred_2d = pred

        overall = metric_components(
            y_true_2d.ravel(),
            y_pred_2d.ravel(),
        )

        by_h = metric_by_horizon(
            y_true_2d=y_true_2d,
            y_pred_2d=y_pred_2d,
            horizon_names=future_target_cols,
        )

        row = {
            "fold": i,
            "cutoff": cutoff,
            "train_rows": len(fit_df),
            "valid_rows": len(valid_df),
            "wape": overall["wape"],
            "rbias": overall["rbias"],
            "score": overall["metric"],
        }

        for _, r in by_h.iterrows():
            h = r["horizon"]
            row[f"{h}_wape"] = r["wape"]
            row[f"{h}_rbias"] = r["rbias"]
            row[f"{h}_score"] = r["metric"]

        rows.append(row)

    return pd.DataFrame(rows)