# Offensive Scheme Data & Clustering

This module scrapes **offensive scheme/tendency** data (motion rate, play action rate, shotgun rate, under center rate, no huddle rate, air yards per attempt) and runs **K-means clustering** to group team-seasons (or teams) into scheme “signatures” for playcaller/scheme analysis.

## Data sources

1. **Sharp Football Analysis** – Current-season tendencies (Motion Rate, Play Action Rate, AirYards/Att, Shotgun Rate, No Huddle Rate). One URL; typically current year only.
2. **Play-by-play (nfl_data_py)** – Multi-year scheme metrics derived from nflfastR-style PBP (shotgun, no_huddle, play_action, air_yards). Use this for 10-year deep dives.

## Quick start

From `backend/ML`:

```bash
# 1) Scrape Sharp Football (current season)
python -m scheme.scrape_scheme_data

# 2) Cluster the scraped data (e.g. 5 scheme clusters)
python -m scheme.scheme_clustering scheme/data/sharp_scheme_2026.csv -k 5
```

Outputs:

- `scheme/data/sharp_scheme_YYYY.csv` – raw scraped metrics
- `scheme/data/sharp_scheme_YYYY_clustered.csv` – same data + `scheme_cluster` (0..k-1)
- `scheme/data/sharp_scheme_YYYY_clustered_centers.csv` – cluster centroids (for interpretation)

## Multi-year data (PBP)

For 10-year scheme analysis, build metrics from play-by-play:

```bash
# Default: 2015 through current year
python -m scheme.scheme_from_pbp

# Specific years
python -m scheme.scheme_from_pbp 2018 2019 2020 2021 2022 2023 2024
```

Requires: `nfl_data_py` (or `nflreadpy`). Writes `scheme/data/scheme_from_pbp.csv`. Then:

```bash
python -m scheme.scheme_clustering scheme/data/scheme_from_pbp.csv -k 6 -o scheme/data/scheme_from_pbp_clustered.csv
```

## Features used for clustering

- `motion_rate` – % of plays with pre-snap motion  
- `play_action_rate` – % of pass attempts that are play action  
- `shotgun_rate` – % of plays from shotgun  
- `under_center_rate` – 100 − shotgun_rate  
- `no_huddle_rate` – % of plays in no-huddle  
- `air_yards_per_att` – average air yards per pass attempt  

You can add PROE or personnel rates later by extending `config.SCHEME_FEATURE_COLUMNS` and ensuring your CSV has those columns.

## Using clusters downstream

- **Coach ↔ scheme:** Join `scheme_cluster` to a coaching DB (team + season) to label each coach with a cluster (e.g. “Cluster 3: High motion / high PROE”).
- **Player predictions:** Use `scheme_cluster` (or cluster centroid features) as input to models that predict player stats, so scheme is part of the feature set.

## Optional: nfelo

nfelo’s Team Tendencies (PROE, personnel, formations) are a good complement. Their site is JS-heavy; adding a Selenium scraper or using their data export (if available) would allow PROE and personnel features in the same pipeline.
