"""
NYC wastewater early-warning report.

    python run.py              # analyse the data snapshot in data/raw
    python run.py --refresh    # first download this week's NYC Health release from GitHub

Outputs: results/status.csv, results/lead_lag.csv, results/onsets.csv, results/forecast.csv,
and charts in images/.
"""
from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ews import data, forecast, signals

RESULTS, IMAGES = data.ROOT / "results", data.ROOT / "images"
BLUE, ORANGE, INK, INK_2, GRID, RED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#e34948"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
    "text.color": INK, "xtick.color": INK_2, "ytick.color": INK_2, "axes.spines.top": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.grid": True,
    "grid.color": GRID,
})


def chart_signals(df: pd.DataFrame, onsets: dict) -> None:
    """Both series indexed to their own peak (0-100%) on ONE axis, so their timing can be compared
    without a misleading second y-scale."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for ax, (name, cols) in zip(axes, data.PATHOGENS.items()):
        ww = signals.smooth(df[cols["wastewater"]])
        ed = signals.smooth(df[cols["ed"]].clip(lower=1e-3))
        ax.plot(ww.index, 100 * ww / ww.max(), color=BLUE, linewidth=2, label="Wastewater viral load")
        ax.plot(ed.index, 100 * ed / ed.max(), color=ORANGE, linewidth=2, label="% of ED visits")
        for d in onsets[name]["wastewater"]:
            ax.axvline(d, color=BLUE, linestyle="--", linewidth=1)
        for d in onsets[name]["ed"]:
            ax.axvline(d, color=ORANGE, linestyle=":", linewidth=1.5)
        ax.set_ylim(0, 108)
        ax.set_ylabel("% of season peak")
        ax.set_title(f"{name} (3-week averages; vertical lines = alert onsets)")
        ax.legend(frameon=False, fontsize=9, loc="upper left")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.tight_layout()
    fig.savefig(IMAGES / "wastewater_vs_ed.png", dpi=150)
    plt.close(fig)


def chart_lead_lag(profiles: dict) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.6))
    colors = {"COVID-19": BLUE, "Influenza A": ORANGE, "RSV": "#1baf7a"}
    for name, prof in profiles.items():
        ax.plot(prof.lag_weeks, prof["corr"], marker="o", markersize=7, linewidth=2, color=colors[name], label=name)
    ax.axvline(0, color=INK_2, linewidth=0.9)
    ax.axhline(0, color=INK_2, linewidth=0.9)
    ax.text(0.2, ax.get_ylim()[1] * 0.92, "wastewater leads →", fontsize=9, color=INK_2)
    ax.text(-0.2, ax.get_ylim()[1] * 0.92, "← ED visits lead", fontsize=9, color=INK_2, ha="right")
    ax.set_xlabel("Lag (weeks): wastewater growth at t − lag vs. ED-visit growth at t")
    ax.set_ylabel("Correlation of weekly growth")
    ax.set_title("Which signal moves first?")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(IMAGES / "lead_lag.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    if args.refresh:
        data.refresh()
    RESULTS.mkdir(exist_ok=True)
    IMAGES.mkdir(exist_ok=True)

    df = data.multi_pathogen()
    print(f"Weekly data: {df.index.min().date()} to {df.index.max().date()} ({len(df)} weeks)\n")

    status_rows, lead_rows, onset_rows, fc_rows, profiles, onsets = [], [], [], [], {}, {}
    for name, cols in data.PATHOGENS.items():
        ww, ed = df[cols["wastewater"]], df[cols["ed"]]
        st = signals.status(ww)
        status_rows.append({"pathogen": name, **st, "latest_pct_ed_visits": ed.iloc[-1]})

        prof = signals.lead_lag_profile(ww, ed)
        profiles[name] = prof
        best = prof.loc[prof["corr"].idxmax()]
        lead_rows.append({"pathogen": name, "best_lag_weeks": int(best.lag_weeks), "corr_at_best": round(best["corr"], 2),
                          "corr_at_lag0": round(prof.loc[prof.lag_weeks == 0, "corr"].iloc[0], 2)})

        w_on = signals.alert_onsets(signals.alert_series(ww))
        e_on = signals.alert_onsets(signals.alert_series(ed, rise=1.5))
        onsets[name] = {"wastewater": w_on, "ed": e_on}
        m = signals.match_onsets(w_on, e_on)
        m.insert(0, "pathogen", name)
        onset_rows.append(m)

        preds = forecast.walk_forward(ed, ww, horizon=1, warmup=20)
        sc = forecast.score(preds)
        sc.insert(0, "pathogen", name)
        fc_rows.append(sc)

    status = pd.DataFrame(status_rows)
    lead = pd.DataFrame(lead_rows)
    ons = pd.concat(onset_rows, ignore_index=True)
    fc = pd.concat(fc_rows, ignore_index=True)
    for frame, fname in [(status, "status"), (lead, "lead_lag"), (ons, "onsets"), (fc, "forecast")]:
        frame.to_csv(RESULTS / f"{fname}.csv", index=False)

    chart_signals(df, onsets)
    chart_lead_lag(profiles)
    print("Current status\n", status.to_string(index=False))
    print("\nLead-lag (positive lag = wastewater leads)\n", lead.to_string(index=False))
    print("\nWave onsets (lead_weeks > 0 = wastewater alerted first)\n", ons.to_string(index=False))
    print("\nNext-week ED forecast, walk-forward\n", fc.to_string(index=False))

    plants = data.plant_wastewater()
    if plants is None:
        print("\n(Optional) Add data/raw/nyc_wastewater_by_plant.csv from NYC Open Data (f7dc-2q9f) "
              "to run the 2020-2025 COVID hospitalization analysis.")
    else:
        city = data.citywide_from_plants(plants)
        hosp = data.daily_covid_hospitalizations().resample("W-SAT").sum()
        joint = pd.concat([city, hosp.rename("hosp")], axis=1).dropna()
        prof = signals.lead_lag_profile(joint["wastewater"], joint["hosp"], max_lag=6)
        prof.to_csv(RESULTS / "covid_long_history_lead_lag.csv", index=False)
        fcl = forecast.score(forecast.walk_forward(joint["hosp"], joint["wastewater"], horizon=2, warmup=52))
        fcl.to_csv(RESULTS / "covid_long_history_forecast.csv", index=False)
        print("\nCOVID 2020-2025 lead-lag vs hospitalizations\n", prof.to_string(index=False))
        print(fcl.to_string(index=False))


if __name__ == "__main__":
    main()
