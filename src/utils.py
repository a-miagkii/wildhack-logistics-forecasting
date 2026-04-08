from datetime import datetime
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT_DIR / "experiments" / "experiment_log.csv"

def log_experiment(
    exp_id,
    model,
    features,
    train_window,
    validation_scheme,
    postprocess,
    valid_metric,
    valid_wape,
    valid_rbias,
    public_score="",
    submission_file="",
    notes=""
):
    row = pd.DataFrame([{
        "exp_id": exp_id,
        "date_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model,
        "features": features,
        "train_window": train_window,
        "validation_scheme": validation_scheme,
        "postprocess": postprocess,
        "valid_metric": valid_metric,
        "valid_wape": valid_wape,
        "valid_rbias": valid_rbias,
        "public_score": public_score,
        "submission_file": submission_file,
        "notes": notes,
    }])

    if LOG_PATH.exists():
        log_df = pd.read_csv(LOG_PATH)
        log_df = pd.concat([log_df, row], ignore_index=True)
    else:
        log_df = row

    log_df.to_csv(LOG_PATH, index=False)
    return log_df.tail(5)