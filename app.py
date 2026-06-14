"""Streamlit UI: every 2026 World Cup match with a Predict button.

Run:  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from src.fixtures import get_fixtures, group_fixtures
from src.model import load_model
from src.predict import predict_match

st.set_page_config(page_title="World Cup 2026 Predictor", page_icon="⚽", layout="centered")


@st.cache_resource
def _load_model():
    return load_model()


@st.cache_data(ttl=3600)
def _load_fixtures():
    return get_fixtures(use_cache=True)


def render_prediction(p):
    score = f"**{p.home_team} {p.scoreline[0]} – {p.scoreline[1]} {p.away_team}**"
    st.markdown(f"{score}  ·  confidence {p.confidence:.0%}")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{p.home_team} win", f"{p.prob_home_win:.0%}")
    c2.metric("Draw", f"{p.prob_draw:.0%}")
    c3.metric(f"{p.away_team} win", f"{p.prob_away_win:.0%}")
    st.progress(min(p.prob_home_win, 1.0), text="Home win probability")
    tops = "   ".join(f"{a}-{b} ({pr:.0%})" for (a, b), pr in p.top_scorelines)
    st.caption(f"Expected goals {p.exp_home_goals:.2f}–{p.exp_away_goals:.2f}  ·  "
               f"most likely scores: {tops}")


def main():
    st.title("⚽ World Cup 2026 — Score Predictor")
    st.caption(
        "Hybrid Elo + machine-learning model trained on ~49,000 international "
        "matches (1872–2026). Click **Predict** on any fixture."
    )

    try:
        models = _load_model()
    except FileNotFoundError:
        st.error("No trained model found. Run `python train.py` first, then reload.")
        st.stop()

    fixtures = _load_fixtures()
    grouped = group_fixtures(fixtures)

    st.info(
        "Confidence is the probability of the single most-likely exact score — "
        "naturally low, since exact scores are hard. The **W/D/L** percentages are "
        "the more reliable signal.",
        icon="ℹ️",
    )

    for label, games in grouped.items():
        st.subheader(label)
        for idx, g in enumerate(games):
            home, away = g["home_team"], g["away_team"]
            # Knockout slots (e.g. "W73", "1A") aren't real nations yet.
            if home not in models.ratings or away not in models.ratings:
                st.write(f"🔒 {home} vs {away} — teams not yet decided")
                continue

            cols = st.columns([3, 2, 2])
            cols[0].markdown(f"**{home}**  vs  **{away}**")
            if g.get("status") == "FINISHED" and g.get("actual_home") is not None:
                cols[1].markdown(f"Actual: {g['actual_home']}–{g['actual_away']}")

            key = f"{label}-{idx}-{home}-{away}"
            if cols[2].button("Predict", key=key):
                with st.spinner("Predicting..."):
                    p = predict_match(home, away, neutral=True,
                                      tournament="FIFA World Cup", models=models)
                render_prediction(p)
            st.divider()


if __name__ == "__main__":
    main()
