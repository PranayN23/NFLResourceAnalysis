"""Team name and abbreviation mappings for scheme data."""
import os

# Output directory for scraped and clustered data (relative to backend/ML)
SCHEME_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(SCHEME_DATA_DIR, exist_ok=True)

# Sharp Football uses full team names; map to standard abbreviations for joining
SHARP_TEAM_TO_ABBR = {
    "Cardinals": "ARI",
    "Falcons": "ATL",
    "Ravens": "BAL",
    "Bills": "BUF",
    "Panthers": "CAR",
    "Bears": "CHI",
    "Bengals": "CIN",
    "Browns": "CLE",
    "Cowboys": "DAL",
    "Broncos": "DEN",
    "Lions": "DET",
    "Packers": "GB",
    "Texans": "HOU",
    "Colts": "IND",
    "Jaguars": "JAX",
    "Chiefs": "KC",
    "Raiders": "LV",
    "Chargers": "LAC",
    "Rams": "LAR",
    "Dolphins": "MIA",
    "Vikings": "MIN",
    "Patriots": "NE",
    "Saints": "NO",
    "Giants": "NYG",
    "Jets": "NYJ",
    "Eagles": "PHI",
    "Steelers": "PIT",
    "49ers": "SF",
    "Seahawks": "SEA",
    "Buccaneers": "TB",
    "Titans": "TEN",
    "Commanders": "WAS",
}

# Columns we use as scheme features for clustering
SCHEME_FEATURE_COLUMNS = [
    "motion_rate",
    "play_action_rate",
    "shotgun_rate",
    "under_center_rate",
    "no_huddle_rate",
    "air_yards_per_att",
]

# Formation-specific rates (from PBP)
FORMATION_FEATURE_COLUMNS = [
    "under_center_play_action_rate",
    "under_center_pass_rate",
    "under_center_run_rate",
    "shotgun_play_action_rate",
    "shotgun_pass_rate",
    "shotgun_run_rate",
]

# Down-specific rates (from PBP)
DOWN_FEATURE_COLUMNS = [
    "down_1_pass_rate",
    "down_1_run_rate",
    "down_2_pass_rate",
    "down_2_run_rate",
    "down_3_pass_rate",
    "down_3_run_rate",
]

# Optional: PROE and personnel if we add nfelo or PBP-derived metrics
SCHEME_FEATURE_COLUMNS_EXTENDED = (
    SCHEME_FEATURE_COLUMNS
    + FORMATION_FEATURE_COLUMNS
    + DOWN_FEATURE_COLUMNS
    + [
        "proe",
        "personnel_11_rate",
        "personnel_12_rate",
        "personnel_21_rate",
    ]
)
