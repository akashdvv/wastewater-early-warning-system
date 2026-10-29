import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ews import data, forecast, signals  # noqa: E402


def weekly(values, start="2025-01-04"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="W-SAT"), dtype=float)


def test_alert_fires_when_wave_takes_off_and_only_once():
    s = weekly([10] * 12 + [12, 18, 30, 55, 90, 140, 150, 120, 80, 50])
    onsets = signals.alert_onsets(signals.alert_series(s))
    assert len(onsets) == 1 and pd.Timestamp("2025-03-29") <= onsets[0] <= pd.Timestamp("2025-04-26")


def test_lead_lag_detects_a_known_two_week_lead():
    rng = np.random.default_rng(1)
    base = np.exp(np.cumsum(rng.normal(0, 0.3, 80)))
    signal = weekly(base)
    outcome = weekly(np.r_[np.full(2, base[0]), base[:-2]])      # outcome = signal delayed 2 weeks
    prof = signals.lead_lag_profile(signal, outcome)
    assert int(prof.loc[prof["corr"].idxmax(), "lag_weeks"]) == 2


def test_onset_matching_sign_convention():
    w = [pd.Timestamp("2025-03-01")]
    o = [pd.Timestamp("2025-03-15")]
    assert signals.match_onsets(w, o).lead_weeks.iloc[0] == 2.0   # wastewater first -> positive


def test_walk_forward_uses_no_future_data():
    s = weekly(np.linspace(1, 50, 40))
    preds = forecast.walk_forward(s, s * 3, horizon=1, warmup=10)
    assert preds.index.min() >= s.index[10] and len(preds) > 0


def test_real_data_loads_and_aligns():
    df = data.multi_pathogen()
    assert len(df) >= 50
    for cols in data.PATHOGENS.values():
        assert cols["wastewater"] in df and cols["ed"] in df
        assert (df[cols["wastewater"]] > 0).all()            # below-detection values were substituted
