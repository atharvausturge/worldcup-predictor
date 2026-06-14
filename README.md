# World Cup 2026 — Score Predictor

A Python app that learns from ~49,000 international football matches (1872–2026)
and predicts the score of 2026 FIFA World Cup games with a confidence number and
Win/Draw/Loss probabilities, served through a Streamlit UI.

## How it works

1. **Data** — historical results come from the public
   [martj42 international-results dataset](https://github.com/martj42/international_results)
   (auto-downloaded). The 2026 fixture list comes from
   [football-data.org](https://www.football-data.org) if you supply a free API key,
   otherwise from the no-key [openfootball](https://github.com/openfootball/worldcup.json)
   JSON.
2. **Model (hybrid Elo + ML)** — we replay history to compute each team's **Elo**
   rating, add recent-form features, and train two **Poisson** gradient-boosted
   regressors: one for expected home goals, one for expected away goals.
3. **Prediction** — those two expected-goal values define a Poisson
   score-probability matrix. From it we read the most-likely scoreline, its
   probability (**confidence**), the **W/D/L** odds, and the top-3 scorelines.

> **Note on accuracy:** exact scorelines are genuinely hard to predict (~10–15%
> even for strong models), so "confidence" is the probability of the single most
> likely score. The **W/D/L** percentages are the more reliable signal.

## Setup

```bash
cd ~/projects/worldcup-predictor
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

(Optional) for live fixtures, get a **free** key at
<https://www.football-data.org/client/register>, then:

```bash
cp .env.example .env   # paste your key into FOOTBALL_DATA_API_KEY
```

Without a key the app uses the openfootball fixture list automatically.

## Run

```bash
python train.py          # downloads data, trains, evaluates vs Elo baseline, saves model
streamlit run app.py     # opens the UI: each match has a Predict button
```

Quick spot-check from the command line:

```bash
python -m src.predict    # prints a few sample match predictions
```

## Project layout

| File | Purpose |
|------|---------|
| `src/data_loader.py` | Download/cache history; normalise team names |
| `src/elo.py` | Iterative Elo ratings over all history |
| `src/features.py` | Leak-free pre-match feature table |
| `src/model.py` | Train / save / load the two Poisson goal models |
| `src/predict.py` | Expected goals → scoreline, confidence, W/D/L |
| `src/fixtures.py` | Fetch 2026 fixtures (API → openfootball fallback) |
| `train.py` | End-to-end training + evaluation CLI |
| `app.py` | Streamlit UI |
