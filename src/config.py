"""Central paths and constants for the World Cup 2026 predictor."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Cached files
RESULTS_CSV = DATA_DIR / "results.csv"
FIXTURES_JSON = DATA_DIR / "fixtures.json"
MODEL_FILE = MODELS_DIR / "goal_models.joblib"

# Data sources
RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)
OPENFOOTBALL_FIXTURES_URL = (
    "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json"
)
FOOTBALL_DATA_URL = "https://api.football-data.org/v4/competitions/WC/matches"

# Score matrix is computed for goal counts 0..MAX_GOALS inclusive.
MAX_GOALS = 10

# Time-based validation split: matches on/after this date form the test set.
TEST_SPLIT_DATE = "2022-01-01"

# Rolling-form windows (number of previous matches).
FORM_WINDOWS = (5, 10)
