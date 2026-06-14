"""Download, cache, and clean the historical international-results dataset."""
from __future__ import annotations

import pandas as pd
import requests

from . import config

# Reconcile team-name differences between the historical CSV (martj42) and the
# 2026 fixture lists (openfootball / football-data.org). Keys are aliases we may
# encounter; values are the canonical name used in the historical dataset.
TEAM_ALIASES = {
    "USA": "United States",
    "US": "United States",
    "United States of America": "United States",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "IR Iran": "Iran",
    "Iran (Islamic Republic of)": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Czechia": "Czech Republic",
    "Cabo Verde": "Cape Verde",
    "Curacao": "Curaçao",
    "Turkiye": "Turkey",
    "Türkiye": "Turkey",
    "China PR": "China",
    "DR Congo": "DR Congo",
    "Congo DR": "DR Congo",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
}


def canonical_team(name: str) -> str:
    """Map any known alias to the canonical historical-dataset team name."""
    if name is None:
        return name
    name = name.strip()
    return TEAM_ALIASES.get(name, name)


def download_results(force: bool = False) -> pd.DataFrame:
    """Return the historical results, downloading and caching the CSV if needed."""
    if force or not config.RESULTS_CSV.exists():
        resp = requests.get(config.RESULTS_URL, timeout=60)
        resp.raise_for_status()
        config.RESULTS_CSV.write_bytes(resp.content)
    return load_results()


def load_results() -> pd.DataFrame:
    """Load the cached results CSV into a cleaned, date-sorted DataFrame."""
    if not config.RESULTS_CSV.exists():
        return download_results()

    df = pd.read_csv(config.RESULTS_CSV)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_score", "away_score"])
    df["home_team"] = df["home_team"].map(canonical_team)
    df["away_team"] = df["away_team"].map(canonical_team)
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    # The CSV stores neutral as the strings "TRUE"/"FALSE".
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")
    df = df.sort_values("date").reset_index(drop=True)
    return df


if __name__ == "__main__":
    data = download_results(force=True)
    print(f"Loaded {len(data):,} matches from {data['date'].min().date()} "
          f"to {data['date'].max().date()}")
    print(data.tail())
