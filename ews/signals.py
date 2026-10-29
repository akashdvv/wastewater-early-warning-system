"""
Early-warning signal processing: smoothing, growth, alert rule, and lead-time measurement.

Everything here uses only data available up to each week (trailing windows), so the alerts
are what a health department would actually have seen in real time - no look-ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def smooth(s: pd.Series, window: int = 3) -> pd.Series:
    """Trailing geometric mean: wastewater is log-normal and spiky, so average in log space."""
    return np.exp(np.log(s.clip(lower=1e-9)).rolling(window, min_periods=window).mean())


def growth(s: pd.Series, window: int = 3) -> pd.Series:
    """Week-over-week log growth of the smoothed series."""
    return np.log(smooth(s, window)).diff()


def alert_series(s: pd.Series, rise: float = 2.0, window: int = 3, baseline: int = 8) -> pd.Series:
    """True in weeks where the smoothed signal is >= `rise` x its lowest smoothed value in the
    previous `baseline` weeks - i.e. a wave is taking off from its recent floor."""
    sm = smooth(s, window)
    floor = sm.shift(1).rolling(baseline, min_periods=4).min()
    return (sm >= rise * floor).fillna(False)


def alert_onsets(alerts: pd.Series, min_gap: int = 6) -> list[pd.Timestamp]:
    """First week of each alert episode; episodes closer than `min_gap` weeks are merged."""
    onsets, last = [], None
    for date, on in alerts.items():
        if on and (last is None or (date - last).days > 7 * min_gap):
            onsets.append(date)
        if on:
            last = date
    return onsets


def match_onsets(signal_onsets, outcome_onsets, max_weeks: int = 4) -> pd.DataFrame:
    """Pair each outcome wave with the nearest signal alert within +/- max_weeks.
    lead_weeks > 0 means wastewater alerted first."""
    rows = []
    for o in outcome_onsets:
        cands = [s for s in signal_onsets if abs((o - s).days) <= 7 * max_weeks]
        if cands:
            s = min(cands, key=lambda x: abs((o - x).days))
            rows.append({"outcome_onset": o.date(), "wastewater_onset": s.date(), "lead_weeks": (o - s).days / 7})
        else:
            rows.append({"outcome_onset": o.date(), "wastewater_onset": None, "lead_weeks": np.nan})
    return pd.DataFrame(rows)


def lead_lag_profile(signal: pd.Series, outcome: pd.Series, max_lag: int = 4, window: int = 3) -> pd.DataFrame:
    """Correlation between wastewater growth at week t-L and outcome growth at week t.
    Using growth rates (not levels) avoids the inflated correlations that trending series give.
    The lag with the highest correlation estimates how far ahead wastewater moves."""
    gs, go = growth(signal, window), growth(outcome.clip(lower=1e-3), window)
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        pair = pd.concat([gs.shift(lag), go], axis=1).dropna()
        rows.append({"lag_weeks": lag, "corr": pair.iloc[:, 0].corr(pair.iloc[:, 1]), "n": len(pair)})
    return pd.DataFrame(rows)


def status(s: pd.Series, window: int = 3) -> dict:
    """Current traffic-light status for the latest week."""
    sm = smooth(s, window).dropna()
    g = np.log(sm).diff()
    last, prev = sm.iloc[-1], sm.iloc[-2]
    pct_rank = (sm.iloc[-52:] < last).mean()
    weekly_change = last / prev - 1
    if weekly_change > 0 and (alert_series(s, window=window).iloc[-1] or (weekly_change > 0.25 and pct_rank > 0.75)):
        level = "HIGH - rising fast"
    elif weekly_change > 0.10:
        level = "ELEVATED - rising"
    elif weekly_change < -0.10:
        level = "falling"
    else:
        level = "stable"
    return {"week": sm.index[-1].date(), "level": level, "weekly_change_pct": round(100 * weekly_change, 1),
            "percentile_vs_past_year": round(100 * pct_rank), "two_week_trend_pct": round(100 * (np.exp(g.iloc[-2:].sum()) - 1), 1)}
