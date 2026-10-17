"""
Does wastewater improve forecasts of hospital burden?

Walk-forward test: for each week t (after a warm-up), fit on data up to t and predict the
outcome h weeks ahead. Compare two models that differ only in their inputs:

    baseline : recent outcome values (what a hospital would extrapolate from on its own)
    + wastewater : the same, plus recent wastewater values

If adding wastewater lowers the error, the signal carries early information.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def _lags(s: pd.Series, name: str, n: int) -> pd.DataFrame:
    return pd.concat({f"{name}_lag{k}": s.shift(k) for k in range(n)}, axis=1)


def walk_forward(outcome: pd.Series, wastewater: pd.Series, horizon: int = 1, n_lags: int = 3,
                 warmup: int = 26, alpha: float = 1.0) -> pd.DataFrame:
    y_log = np.log(outcome.clip(lower=1e-3))
    w_log = np.log(wastewater.clip(lower=1e-9))
    target = y_log.shift(-horizon).rename("target")
    X_base = _lags(y_log, "y", n_lags)
    X_ww = pd.concat([X_base, _lags(w_log, "ww", n_lags)], axis=1)
    data = pd.concat([target, X_ww], axis=1).dropna(subset=list(X_ww.columns))
    rows = []
    for i in range(warmup, len(data)):
        train = data.iloc[: i - horizon + 1].dropna(subset=["target"])  # only targets already observed at t
        test = data.iloc[[i]]
        if len(train) < warmup // 2 or test["target"].isna().all():
            continue
        row = {"week": data.index[i], "actual": float(np.exp(test["target"].iloc[0])),
               "naive": float(np.exp(test["y_lag0"].iloc[0]))}
        for name, cols in [("baseline", list(X_base.columns)), ("with_wastewater", list(X_ww.columns))]:
            m = Ridge(alpha=alpha).fit(train[cols], train["target"])
            row[name] = float(np.exp(m.predict(test[cols])[0]))
        rows.append(row)
    return pd.DataFrame(rows).set_index("week")


def score(preds: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute percentage error and direction accuracy for each model."""
    out = []
    actual = preds["actual"]
    last = preds["naive"]
    for m in ["naive", "baseline", "with_wastewater"]:
        p = preds[m]
        mape = (abs(p - actual) / actual.clip(lower=1e-3)).mean() * 100
        direction = (np.sign(p - last) == np.sign(actual - last)).mean() * 100 if m != "naive" else np.nan
        out.append({"model": m, "MAPE_%": round(mape, 1), "direction_accuracy_%": round(direction, 1) if m != "naive" else None,
                    "n_weeks": len(preds)})
    return pd.DataFrame(out)
