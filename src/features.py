"""Build a leak-free pre-match feature table for goal modelling.

Every feature for a given match is derived only from matches played *before*
that match's date, so there is no information leakage from the result we are
trying to predict.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, List

import pandas as pd

from . import config
from .elo import TOURNAMENT_WEIGHT, DEFAULT_K

FEATURE_COLUMNS: List[str] = []  # populated by build_features()


def _tournament_weight(tournament: str) -> float:
    return TOURNAMENT_WEIGHT.get(tournament, DEFAULT_K)


def _form_block(history: Deque, window: int):
    """Return (avg points, avg goals for, avg goals against) over a team's
    most recent ``window`` matches. Falls back to neutral defaults when a team
    has little history."""
    recent = list(history)[-window:]
    if not recent:
        return 1.0, 1.0, 1.0  # neutral priors: a draw, one goal each way
    pts = sum(r[0] for r in recent) / len(recent)
    gf = sum(r[1] for r in recent) / len(recent)
    ga = sum(r[2] for r in recent) / len(recent)
    return pts, gf, ga


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Given an Elo-augmented, date-sorted results frame, return a feature table
    including the goal targets ``home_score`` and ``away_score``."""
    global FEATURE_COLUMNS

    # Per-team rolling history of (points, goals_for, goals_against).
    history: Dict[str, Deque] = defaultdict(lambda: deque(maxlen=max(config.FORM_WINDOWS)))
    rows = []

    for r in df.itertuples(index=False):
        feat = {
            "date": r.date,
            "home_team": r.home_team,
            "away_team": r.away_team,
            "home_elo": r.home_elo,
            "away_elo": r.away_elo,
            "elo_diff": r.home_elo - r.away_elo,
            "neutral": int(r.neutral),
            "tournament_weight": _tournament_weight(r.tournament),
        }
        for w in config.FORM_WINDOWS:
            hp, hgf, hga = _form_block(history[r.home_team], w)
            ap, agf, aga = _form_block(history[r.away_team], w)
            feat[f"home_pts_{w}"] = hp
            feat[f"home_gf_{w}"] = hgf
            feat[f"home_ga_{w}"] = hga
            feat[f"away_pts_{w}"] = ap
            feat[f"away_gf_{w}"] = agf
            feat[f"away_ga_{w}"] = aga
        feat["home_score"] = r.home_score
        feat["away_score"] = r.away_score
        rows.append(feat)

        # Update rolling history AFTER recording features (keeps it leak-free).
        if r.home_score > r.away_score:
            hp_pts, ap_pts = 3, 0
        elif r.home_score < r.away_score:
            hp_pts, ap_pts = 0, 3
        else:
            hp_pts, ap_pts = 1, 1
        history[r.home_team].append((hp_pts, r.home_score, r.away_score))
        history[r.away_team].append((ap_pts, r.away_score, r.home_score))

    table = pd.DataFrame(rows)
    FEATURE_COLUMNS = [
        c for c in table.columns
        if c not in ("date", "home_team", "away_team", "home_score", "away_score")
    ]
    return table


def feature_columns(table: pd.DataFrame) -> List[str]:
    return [
        c for c in table.columns
        if c not in ("date", "home_team", "away_team", "home_score", "away_score")
    ]


def final_form(df: pd.DataFrame) -> Dict[str, list]:
    """Return each team's most-recent (pts, gf, ga) results after replaying all
    matches in ``df`` — used to build feature rows for upcoming fixtures."""
    history: Dict[str, Deque] = defaultdict(lambda: deque(maxlen=max(config.FORM_WINDOWS)))
    for r in df.itertuples(index=False):
        if r.home_score > r.away_score:
            hp_pts, ap_pts = 3, 0
        elif r.home_score < r.away_score:
            hp_pts, ap_pts = 0, 3
        else:
            hp_pts, ap_pts = 1, 1
        history[r.home_team].append((hp_pts, r.home_score, r.away_score))
        history[r.away_team].append((ap_pts, r.away_score, r.home_score))
    return {team: list(dq) for team, dq in history.items()}


def feature_row_for(
    home_team: str,
    away_team: str,
    ratings: Dict[str, float],
    form: Dict[str, list],
    neutral: bool = True,
    tournament: str = "FIFA World Cup",
) -> Dict[str, float]:
    """Construct a single feature row for an upcoming fixture from stored
    Elo ratings and recent form."""
    from .elo import BASE_RATING

    home_elo = ratings.get(home_team, BASE_RATING)
    away_elo = ratings.get(away_team, BASE_RATING)
    row = {
        "home_elo": home_elo,
        "away_elo": away_elo,
        "elo_diff": home_elo - away_elo,
        "neutral": int(neutral),
        "tournament_weight": _tournament_weight(tournament),
    }
    for w in config.FORM_WINDOWS:
        hp, hgf, hga = _form_block(deque(form.get(home_team, [])), w)
        ap, agf, aga = _form_block(deque(form.get(away_team, [])), w)
        row[f"home_pts_{w}"] = hp
        row[f"home_gf_{w}"] = hgf
        row[f"home_ga_{w}"] = hga
        row[f"away_pts_{w}"] = ap
        row[f"away_gf_{w}"] = agf
        row[f"away_ga_{w}"] = aga
    return row
