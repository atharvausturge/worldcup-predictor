"""Turn a fixture into a predicted scoreline, confidence, and W/D/L odds.

From the two expected-goal values we build a score-probability matrix assuming
each side's goals are Poisson-distributed, then read everything off that matrix.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from scipy.stats import poisson

from . import config
from .data_loader import canonical_team
from .features import feature_row_for
from .model import GoalModels, load_model, predict_goals


@dataclass
class Prediction:
    home_team: str
    away_team: str
    exp_home_goals: float
    exp_away_goals: float
    scoreline: Tuple[int, int]      # most-likely exact score
    confidence: float               # probability of that exact score
    prob_home_win: float
    prob_draw: float
    prob_away_win: float
    top_scorelines: List[Tuple[Tuple[int, int], float]] = field(default_factory=list)


def score_matrix(lam_home: float, lam_away: float, max_goals: int = config.MAX_GOALS):
    """P(i,j) for home=i, away=j, i,j in 0..max_goals (renormalised)."""
    home_p = poisson.pmf(np.arange(max_goals + 1), lam_home)
    away_p = poisson.pmf(np.arange(max_goals + 1), lam_away)
    matrix = np.outer(home_p, away_p)
    return matrix / matrix.sum()


def _derive(home_team: str, away_team: str, lam_home: float, lam_away: float) -> Prediction:
    matrix = score_matrix(lam_home, lam_away)
    i, j = np.unravel_index(np.argmax(matrix), matrix.shape)
    confidence = float(matrix[i, j])

    prob_home = float(np.tril(matrix, -1).sum())  # home goals > away goals
    prob_draw = float(np.trace(matrix))
    prob_away = float(np.triu(matrix, 1).sum())

    flat = [((a, b), float(matrix[a, b]))
            for a in range(matrix.shape[0]) for b in range(matrix.shape[1])]
    flat.sort(key=lambda kv: kv[1], reverse=True)

    return Prediction(
        home_team=home_team,
        away_team=away_team,
        exp_home_goals=lam_home,
        exp_away_goals=lam_away,
        scoreline=(int(i), int(j)),
        confidence=confidence,
        prob_home_win=prob_home,
        prob_draw=prob_draw,
        prob_away_win=prob_away,
        top_scorelines=flat[:3],
    )


def predict_match(
    home_team: str,
    away_team: str,
    neutral: bool = True,
    tournament: str = "FIFA World Cup",
    models: GoalModels | None = None,
) -> Prediction:
    """Predict a single fixture. ``models`` is loaded from disk if not provided."""
    if models is None:
        models = load_model()
    home_team = canonical_team(home_team)
    away_team = canonical_team(away_team)
    row = feature_row_for(
        home_team, away_team, models.ratings, models.form,
        neutral=neutral, tournament=tournament,
    )
    lam_home, lam_away = predict_goals(models, row)
    return _derive(home_team, away_team, lam_home, lam_away)


if __name__ == "__main__":
    for h, a in [("Brazil", "Argentina"), ("United States", "Canada"),
                 ("France", "Germany"), ("Spain", "England")]:
        p = predict_match(h, a)
        print(f"{p.home_team} {p.scoreline[0]}-{p.scoreline[1]} {p.away_team} "
              f"| conf {p.confidence:.0%} "
              f"| W/D/L {p.prob_home_win:.0%}/{p.prob_draw:.0%}/{p.prob_away_win:.0%} "
              f"| xG {p.exp_home_goals:.2f}-{p.exp_away_goals:.2f}")
