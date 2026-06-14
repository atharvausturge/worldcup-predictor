"""Fetch the 2026 World Cup fixture list.

Prefers the football-data.org API when a key is present, otherwise falls back to
the openfootball public JSON (no key required). Both are normalised to a common
shape: a list of dicts with keys home_team, away_team, group, date, status,
and (if already played) actual_home / actual_away.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import requests
from dotenv import load_dotenv

from . import config
from .data_loader import canonical_team

load_dotenv()


def _from_football_data(api_key: str) -> List[Dict]:
    resp = requests.get(
        config.FOOTBALL_DATA_URL,
        headers={"X-Auth-Token": api_key},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    fixtures = []
    for m in data.get("matches", []):
        score = m.get("score", {}).get("fullTime", {})
        played = m.get("status") == "FINISHED"
        fixtures.append({
            "home_team": canonical_team((m.get("homeTeam") or {}).get("name", "TBD")),
            "away_team": canonical_team((m.get("awayTeam") or {}).get("name", "TBD")),
            "group": m.get("group") or m.get("stage", "Group Stage"),
            "date": (m.get("utcDate") or "")[:10],
            "status": m.get("status", "SCHEDULED"),
            "actual_home": score.get("home") if played else None,
            "actual_away": score.get("away") if played else None,
        })
    return fixtures


def _from_openfootball() -> List[Dict]:
    resp = requests.get(config.OPENFOOTBALL_FIXTURES_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    fixtures = []
    for m in data.get("matches", []):
        ft = (m.get("score") or {}).get("ft")
        played = bool(ft)
        fixtures.append({
            "home_team": canonical_team(m.get("team1", "TBD")),
            "away_team": canonical_team(m.get("team2", "TBD")),
            "group": m.get("group") or m.get("round", "Group Stage"),
            "date": m.get("date", ""),
            "status": "FINISHED" if played else "SCHEDULED",
            "actual_home": ft[0] if played else None,
            "actual_away": ft[1] if played else None,
        })
    return fixtures


def get_fixtures(use_cache: bool = True) -> List[Dict]:
    """Return the fixture list, caching the result to data/fixtures.json."""
    if use_cache and config.FIXTURES_JSON.exists():
        return json.loads(config.FIXTURES_JSON.read_text())

    api_key = os.getenv("FOOTBALL_DATA_API_KEY", "").strip()
    fixtures: List[Dict] = []
    if api_key:
        try:
            fixtures = _from_football_data(api_key)
        except Exception as exc:  # noqa: BLE001 - fall back gracefully
            print(f"football-data.org fetch failed ({exc}); using openfootball fallback.")
    if not fixtures:
        fixtures = _from_openfootball()

    config.FIXTURES_JSON.write_text(json.dumps(fixtures, indent=2))
    return fixtures


def group_fixtures(fixtures: List[Dict]) -> Dict[str, List[Dict]]:
    """Group fixtures by their group/stage label, preserving first-seen order."""
    grouped: Dict[str, List[Dict]] = {}
    for f in fixtures:
        grouped.setdefault(f["group"], []).append(f)
    return grouped


if __name__ == "__main__":
    fx = get_fixtures(use_cache=False)
    print(f"Loaded {len(fx)} fixtures")
    for label, games in list(group_fixtures(fx).items())[:2]:
        print(f"\n{label}")
        for g in games[:4]:
            print(f"  {g['date']}  {g['home_team']} vs {g['away_team']} ({g['status']})")
