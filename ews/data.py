"""
Load NYC Health Department surveillance data into tidy weekly tables.

Sources (all public, NYC Department of Health and Mental Hygiene):
- nychealth/respiratory-illness-data : weekly citywide wastewater viral load (SARS-CoV-2, Flu A,
  Flu B, RSV) and weekly % of ED visits / admissions by pathogen. Updated every Thursday.
- nychealth/coronavirus-data         : daily COVID-19 hospitalizations, Feb 2020 - Oct 2025.
- NYC Open Data f7dc-2q9f (optional)  : SARS-CoV-2 concentration in wastewater for each of NYC's
  14 treatment plants, Aug 2020 onward. Download it to data/raw/nyc_wastewater_by_plant.csv.

`refresh()` re-downloads the GitHub files so the system picks up each new weekly release.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

RESP = "https://raw.githubusercontent.com/nychealth/respiratory-illness-data/main/data/"
COVID = "https://raw.githubusercontent.com/nychealth/coronavirus-data/master/trends/"
SOURCES = {
    "wastewaterData.csv": RESP + "wastewaterData.csv",
    "emergencyDeptData.csv": RESP + "emergencyDeptData.csv",
    "historical_ED_data.csv": RESP + "archive/historical_ED_data.csv",
    "hosp-by-day.csv": COVID + "hosp-by-day.csv",
}
PLANT_FILE = RAW / "nyc_wastewater_by_plant.csv"

# wastewater signal -> the clinical outcome it should warn about
PATHOGENS = {
    "COVID-19": {"wastewater": "SARS-CoV-2 viral load", "ed": "COVID-19 visits", "admits": "COVID-19 hospitalizations"},
    "Influenza A": {"wastewater": "Flu A viral load", "ed": "Influenza visits", "admits": "Influenza hospitalizations"},
    "RSV": {"wastewater": "RSV viral load", "ed": "RSV visits", "admits": "RSV hospitalizations"},
}


def refresh() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        urllib.request.urlretrieve(url, RAW / name)
        print(f"downloaded {name}")


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.strip(), errors="coerce")


def weekly_wastewater() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Citywide weekly viral load per pathogen.
    Values reported as "< LOD" (below the limit of detection) become half the lowest detected
    value for that pathogen - a standard substitution - and are flagged in a second frame."""
    ww = pd.read_csv(RAW / "wastewaterData.csv")
    ww = ww[ww["submetric"].str.contains("Citywide", case=False, na=False)]
    ww["date"] = pd.to_datetime(ww["date"])
    raw = ww.pivot_table(index="date", columns="metric", values="value", aggfunc="first")
    below_lod = raw.apply(lambda s: s.astype(str).str.contains("LOD", na=False))
    vals = raw.apply(_num)
    vals = vals.fillna({c: vals[c].min() / 2 for c in vals})
    return vals.sort_index(), below_lod.sort_index()


def weekly_ed() -> pd.DataFrame:
    """Weekly % of all ED visits (and % admitted) attributed to each pathogen, citywide."""
    ed = pd.concat([pd.read_csv(RAW / "historical_ED_data.csv"), pd.read_csv(RAW / "emergencyDeptData.csv")])
    ed = ed.drop_duplicates(["date", "metric", "submetric"], keep="last")
    ed = ed[ed["submetric"] == "Overall"]
    ed["date"] = pd.to_datetime(ed["date"])
    return ed.pivot(index="date", columns="metric", values="value").apply(_num).sort_index()


def multi_pathogen() -> pd.DataFrame:
    ww, _ = weekly_wastewater()
    return ww.join(weekly_ed(), how="inner")


def daily_covid_hospitalizations() -> pd.Series:
    h = pd.read_csv(RAW / "hosp-by-day.csv")
    h["date"] = pd.to_datetime(h["date_of_interest"])
    return h.set_index("date")["HOSPITALIZED_COUNT"].astype(float).sort_index()


def plant_wastewater(path: Path = PLANT_FILE) -> pd.DataFrame | None:
    """Per-plant SARS-CoV-2 wastewater data from NYC Open Data, standardised to
    columns: date, plant, conc (copies/L), load_per_capita, population."""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    def pick(*keys):
        for k in keys:
            for lc, orig in cols.items():
                if all(part in lc for part in k.split("|")):
                    return orig
        return None

    date_c = pick("sample date", "collection date", "date")
    plant_c = pick("wrrf name", "wrrf|name", "plant", "wrrf")
    conc_c = pick("concentration|n1", "concentration")
    load_c = pick("per capita", "per-capita", "load")
    pop_c = pick("population")
    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_c], errors="coerce"),
        "plant": df[plant_c].astype(str).str.strip(),
        "conc": _num(df[conc_c]) if conc_c else np.nan,
        "load_per_capita": _num(df[load_c]) if load_c else np.nan,
        "population": _num(df[pop_c]) if pop_c else np.nan,
    })
    return out.dropna(subset=["date"]).sort_values("date")


def citywide_from_plants(plants: pd.DataFrame, freq: str = "W-SAT") -> pd.Series:
    """Population-weighted citywide signal: weekly geometric mean of per-capita load per plant,
    then weighted by population served (falls back to equal weights)."""
    metric = "load_per_capita" if plants["load_per_capita"].notna().mean() > 0.5 else "conc"
    p = plants.dropna(subset=[metric])
    p = p[p[metric] > 0]
    p["log"] = np.log(p[metric])
    wk = p.groupby([pd.Grouper(key="date", freq=freq), "plant"]).agg(log=("log", "mean"), pop=("population", "max"))
    wk["pop"] = wk["pop"].fillna(1.0)
    city = wk.groupby(level=0).apply(lambda g: np.average(g["log"], weights=g["pop"]))
    return np.exp(city).rename("wastewater")
