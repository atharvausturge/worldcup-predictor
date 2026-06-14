"""Train and evaluate the World Cup goal models.

Pipeline: download history -> compute Elo -> build leak-free features ->
time-based train/test split -> train Poisson goal models -> evaluate against a
pure-Elo baseline -> save artifacts for the app.

Run:  python train.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

# LightGBM/sklearn emit a cosmetic feature-name warning when predicting on arrays.
warnings.filterwarnings("ignore", message="X does not have valid feature names")

from src import config
from src.data_loader import download_results
from src.elo import compute_elo, current_ratings, _expected, HOME_ADVANTAGE
from src.features import build_features, feature_columns, final_form
from src.model import GoalModels, train, predict_goals, save_model
from src.predict import score_matrix


def _outcome_probs_from_matrix(lam_home, lam_away):
    m = score_matrix(lam_home, lam_away)
    return (
        float(np.tril(m, -1).sum()),  # home win
        float(np.trace(m)),           # draw
        float(np.triu(m, 1).sum()),   # away win
    )


def _actual_outcome(hs, as_):
    return 0 if hs > as_ else (1 if hs == as_ else 2)


def ranked_probability_score(probs, outcome):
    """RPS for a 3-outcome (home/draw/away) ordered prediction. Lower is better."""
    cum_p, cum_o, total = 0.0, 0.0, 0.0
    onehot = [0, 0, 0]
    onehot[outcome] = 1
    for k in range(3):
        cum_p += probs[k]
        cum_o += onehot[k]
        total += (cum_p - cum_o) ** 2
    return total / 2.0


def evaluate(name, prob_list, outcomes):
    """prob_list: list of (home, draw, away) triples; outcomes: list of 0/1/2."""
    rps, logloss, correct = [], [], 0
    eps = 1e-12
    for probs, outcome in zip(prob_list, outcomes):
        rps.append(ranked_probability_score(probs, outcome))
        logloss.append(-np.log(max(probs[outcome], eps)))
        if int(np.argmax(probs)) == outcome:
            correct += 1
    print(f"\n{name}")
    print(f"  RPS (lower better):     {np.mean(rps):.4f}")
    print(f"  Log-loss (lower better):{np.mean(logloss):.4f}")
    print(f"  W/D/L accuracy:         {correct / len(outcomes):.1%}")


def main():
    print("Downloading / loading historical results ...")
    df = download_results()
    print(f"  {len(df):,} matches, {df['date'].min().date()} -> {df['date'].max().date()}")

    print("Computing Elo ratings ...")
    df = compute_elo(df)

    print("Building leak-free features ...")
    table = build_features(df)
    cols = feature_columns(table)

    split = pd.Timestamp(config.TEST_SPLIT_DATE)
    train_tbl = table[table["date"] < split]
    test_tbl = table[table["date"] >= split]
    print(f"  train: {len(train_tbl):,}  test (>= {config.TEST_SPLIT_DATE}): {len(test_tbl):,}")

    print("Training Poisson goal models ...")
    home_model, away_model = train(train_tbl, cols)

    outcomes = [_actual_outcome(hs, a) for hs, a
                in zip(test_tbl["home_score"], test_tbl["away_score"])]

    # --- ML model: batch-predict expected goals, then build outcome probs ---
    X_test = test_tbl[cols].to_numpy()
    lam_h = np.clip(home_model.predict(X_test), 0.05, 8)
    lam_a = np.clip(away_model.predict(X_test), 0.05, 8)
    ml_probs = [_outcome_probs_from_matrix(h, a) for h, a in zip(lam_h, lam_a)]

    # --- Pure-Elo baseline: expected score -> (win, draw, loss) split ---
    DRAW_RATE = 0.26  # historical draw frequency
    elo_probs = []
    for r in test_tbl.itertuples(index=False):
        adj = 0.0 if r.neutral else HOME_ADVANTAGE
        p_home = _expected(r.home_elo + adj, r.away_elo)
        elo_probs.append(
            (p_home * (1 - DRAW_RATE), DRAW_RATE, (1 - p_home) * (1 - DRAW_RATE))
        )

    print("\n=== Evaluation on held-out matches ===")
    evaluate("Pure-Elo baseline", elo_probs, outcomes)
    evaluate("Hybrid Elo + ML (this model)", ml_probs, outcomes)

    print("\nRefitting on ALL data for deployment ...")
    home_model, away_model = train(table, cols)
    models = GoalModels(
        home_model=home_model,
        away_model=away_model,
        feature_columns=cols,
        ratings=current_ratings(df),
        form=final_form(df),
    )
    save_model(models)
    print(f"Saved model -> {config.MODEL_FILE}")


if __name__ == "__main__":
    main()
