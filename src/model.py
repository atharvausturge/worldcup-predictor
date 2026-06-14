"""Train, persist, and apply the two Poisson goal models.

We train one regressor to predict expected home goals and another for expected
away goals, both with a Poisson objective. LightGBM is preferred; if it is not
installed we fall back to scikit-learn's PoissonRegressor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from . import config

try:
    from lightgbm import LGBMRegressor
    _HAVE_LGBM = True
except Exception:  # pragma: no cover - optional dependency
    _HAVE_LGBM = False
    from sklearn.linear_model import PoissonRegressor


@dataclass
class GoalModels:
    home_model: object
    away_model: object
    feature_columns: List[str]
    # Prediction context, captured from the full history at train time:
    ratings: Dict[str, float]            # current Elo per team
    form: Dict[str, list]                # recent (pts, gf, ga) per team


def _new_regressor():
    if _HAVE_LGBM:
        return LGBMRegressor(
            objective="poisson",
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=40,
            subsample=0.8,
            colsample_bytree=0.8,
            verbose=-1,
        )
    return PoissonRegressor(alpha=1e-4, max_iter=500)


def train(table: pd.DataFrame, feature_columns: List[str]) -> tuple:
    """Fit and return (home_model, away_model) on the given feature table."""
    X = table[feature_columns].to_numpy()
    home_model = _new_regressor().fit(X, table["home_score"].to_numpy())
    away_model = _new_regressor().fit(X, table["away_score"].to_numpy())
    return home_model, away_model


def predict_goals(models: GoalModels, feature_row: Dict[str, float]) -> tuple:
    """Return (lambda_home, lambda_away) expected goals for one feature row."""
    x = np.array([[feature_row[c] for c in models.feature_columns]])
    lam_home = float(np.clip(models.home_model.predict(x)[0], 0.05, 8.0))
    lam_away = float(np.clip(models.away_model.predict(x)[0], 0.05, 8.0))
    return lam_home, lam_away


def save_model(models: GoalModels) -> None:
    joblib.dump(models, config.MODEL_FILE)


def load_model() -> GoalModels:
    if not config.MODEL_FILE.exists():
        raise FileNotFoundError(
            f"No trained model at {config.MODEL_FILE}. Run `python train.py` first."
        )
    return joblib.load(config.MODEL_FILE)
