import numpy as np
import pandas as pd

def metric_components(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denom = y_true.sum()
    if denom == 0:
        raise ValueError("Sum of y_true is zero; metric is undefined.")

    wape = np.abs(y_pred - y_true).sum() / denom
    rbias = abs(y_pred.sum() / denom - 1)
    total = wape + rbias

    return {
        "wape": float(wape),
        "rbias": float(rbias),
        "metric": float(total),
    }



def metric_by_horizon(y_true_2d, y_pred_2d, horizon_names=None):
    y_true_2d = np.asarray(y_true_2d, dtype=float)
    y_pred_2d = np.asarray(y_pred_2d, dtype=float)

    n_horizons = y_true_2d.shape[1]
    if horizon_names is None:
        horizon_names = [f"step_{i+1}" for i in range(n_horizons)]

    rows = []
    for i, name in enumerate(horizon_names):
        scores = metric_components(y_true_2d[:, i], y_pred_2d[:, i])
        rows.append({
            "horizon": name,
            "wape": scores["wape"],
            "rbias": scores["rbias"],
            "metric": scores["metric"],
        })

    return pd.DataFrame(rows)