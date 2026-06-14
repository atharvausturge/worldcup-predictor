"""Iterative Elo ratings over the full history of international matches.

We replay every match in date order. Before each match we record the teams'
current ratings (these become leak-free pre-match features), then update them
based on the result, the margin of victory, and the importance of the fixture.
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

BASE_RATING = 1500.0
HOME_ADVANTAGE = 65.0  # Elo points added to the home side when not on neutral ground.

# K-factor by tournament importance (bigger games move ratings more).
TOURNAMENT_WEIGHT = {
    "FIFA World Cup": 60.0,
    "FIFA World Cup qualification": 40.0,
    "UEFA Euro": 50.0,
    "Copa América": 50.0,
    "Copa America": 50.0,
    "African Cup of Nations": 45.0,
    "AFC Asian Cup": 45.0,
    "UEFA Nations League": 40.0,
    "Confederations Cup": 45.0,
    "Friendly": 20.0,
}
DEFAULT_K = 30.0


def _k_factor(tournament: str) -> float:
    return TOURNAMENT_WEIGHT.get(tournament, DEFAULT_K)


def _expected(rating_a: float, rating_b: float) -> float:
    """Expected score for A vs B under the logistic Elo curve."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def _mov_multiplier(goal_diff: int, rating_diff: float) -> float:
    """Margin-of-victory multiplier (World Football Elo style).

    Larger wins move ratings more, with a dampening term so a strong favourite
    thrashing a minnow does not over-inflate.
    """
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8.0 * (2.2 / (rating_diff * 0.001 + 2.2))


def compute_elo(df: pd.DataFrame) -> pd.DataFrame:
    """Add pre-match Elo columns to ``df`` and return it.

    Adds: ``home_elo``, ``away_elo`` (ratings *before* each match is played).
    ``df`` must be sorted by date ascending.
    """
    ratings: Dict[str, float] = {}
    home_elos: List[float] = []
    away_elos: List[float] = []

    for row in df.itertuples(index=False):
        ra = ratings.get(row.home_team, BASE_RATING)
        rb = ratings.get(row.away_team, BASE_RATING)
        home_elos.append(ra)
        away_elos.append(rb)

        # Apply home advantage only for the expectation calculation.
        adj = 0.0 if row.neutral else HOME_ADVANTAGE
        exp_home = _expected(ra + adj, rb)

        if row.home_score > row.away_score:
            actual_home = 1.0
        elif row.home_score < row.away_score:
            actual_home = 0.0
        else:
            actual_home = 0.5

        k = _k_factor(row.tournament)
        mult = _mov_multiplier(row.home_score - row.away_score, (ra + adj) - rb)
        delta = k * mult * (actual_home - exp_home)

        ratings[row.home_team] = ra + delta
        ratings[row.away_team] = rb - delta

    out = df.copy()
    out["home_elo"] = home_elos
    out["away_elo"] = away_elos
    return out


def current_ratings(df: pd.DataFrame) -> Dict[str, float]:
    """Return each team's Elo rating *after* replaying all matches in ``df``."""
    ratings: Dict[str, float] = {}
    for row in df.itertuples(index=False):
        ra = ratings.get(row.home_team, BASE_RATING)
        rb = ratings.get(row.away_team, BASE_RATING)
        adj = 0.0 if row.neutral else HOME_ADVANTAGE
        exp_home = _expected(ra + adj, rb)
        if row.home_score > row.away_score:
            actual_home = 1.0
        elif row.home_score < row.away_score:
            actual_home = 0.0
        else:
            actual_home = 0.5
        k = _k_factor(row.tournament)
        mult = _mov_multiplier(row.home_score - row.away_score, (ra + adj) - rb)
        delta = k * mult * (actual_home - exp_home)
        ratings[row.home_team] = ra + delta
        ratings[row.away_team] = rb - delta
    return ratings


if __name__ == "__main__":
    from .data_loader import load_results

    data = compute_elo(load_results())
    ratings = current_ratings(data)
    top = sorted(ratings.items(), key=lambda kv: kv[1], reverse=True)[:20]
    print("Top 20 teams by current Elo:")
    for i, (team, rating) in enumerate(top, 1):
        print(f"{i:>2}. {team:<20} {rating:7.1f}")
