# Offensive Scheme Data & Clustering

This module scrapes **offensive scheme/tendency** data (motion rate, play action rate, shotgun rate, under center rate, no huddle rate, air yards per attempt) and runs **K-means clustering** to group team-seasons (or teams) into scheme “signatures” for playcaller/scheme analysis.

## Data sources

1. **Sharp Football Analysis** – Current-season tendencies (Motion Rate, Play Action Rate, AirYards/Att, Shotgun Rate, No Huddle Rate). One URL; typically current year only.
2. **Play-by-play (nflfastR / nfl_data_py / nflreadpy)** – Multi-year scheme metrics derived from nflfastR-style PBP. Includes:
   - **Basic rates**: motion, play action, shotgun, under center, no huddle, air yards
   - **Formation-specific rates**: Under center play action/dropback/run, Shotgun play action/dropback/run
   - **Down-specific rates**: 1st/2nd/3rd down pass and run rates
   - **Personnel rates** (coming): 11/12/21 personnel groupings
   - **Target hotspots** (coming): Short/medium/deep × left/middle/right target rates

   Use this for 10-year deep dives. For **true play-action rates**, download nflfastR PBP CSVs with a `play_action` column and place them in `backend/ML/scheme/raw_pbp/` as `pbp_2015.csv`, `pbp_2016.csv`, ..., `pbp_2025.csv`. The `scheme_from_pbp` script will automatically prefer these local files.

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

For 10-year scheme analysis, build metrics from play-by-play. This includes all formation-specific and down-specific rates:

```bash
# Default: 2015 through current year
python -m scheme.scheme_from_pbp

# Specific years
python -m scheme.scheme_from_pbp 2018 2019 2020 2021 2022 2023 2024
```

**Data source**:
- Preferred: **nflfastR PBP CSVs** with a `play_action` column, saved as `scheme/raw_pbp/pbp_<year>.csv`.
- Fallback: `nfl_data_py` or `nflreadpy` if local CSVs are missing (these lack reliable `play_action`/motion).

The PBP data includes:
- `shotgun` (0/1) – formation indicator
- `play_action` (0/1) – play action flag
- `pass_attempt` (0/1) – pass indicator
- `rush_attempt` (0/1) – run indicator
- `down` (1/2/3/4) – down number
- `air_yards` – air yards for passes
- `pre_snap_motion` or `motion` – motion indicator (if available)

Writes `scheme/data/scheme_from_pbp.csv` with all formation and down-specific rates. Then:

```bash
python -m scheme.scheme_clustering scheme/data/scheme_from_pbp.csv -k 6 -o scheme/data/scheme_from_pbp_clustered.csv
```

## Features used for clustering

**Basic features** (from Sharp Football or PBP):
- `motion_rate` – % of plays with pre-snap motion  
- `play_action_rate` – % of pass attempts that are play action  
- `shotgun_rate` – % of plays from shotgun  
- `under_center_rate` – 100 − shotgun_rate  
- `no_huddle_rate` – % of plays in no-huddle  
- `air_yards_per_att` – average air yards per pass attempt  

**Formation-specific features** (from PBP only):
- `under_center_play_action_rate` – % of under-center plays that are play action passes
- `under_center_pass_rate` – % of under-center plays that are passes (all passes: dropback + play action)
- `under_center_run_rate` – % of under-center plays that are runs
- `shotgun_play_action_rate` – % of shotgun plays that are play action passes
- `shotgun_pass_rate` – % of shotgun plays that are passes (all passes: dropback + play action)
- `shotgun_run_rate` – % of shotgun plays that are runs

**QB rushing features** (from PBP only):
- `qb_rush_rate` – % of all plays that are designed QB runs (excluding scrambles)
- `read_option_rate` – % of all plays that are read option (subset of QB rushing)

**Down-specific features** (from PBP only):
- `down_1_pass_rate`, `down_1_run_rate` – Pass/run rates on 1st down
- `down_2_pass_rate`, `down_2_run_rate` – Pass/run rates on 2nd down
- `down_3_pass_rate`, `down_3_run_rate` – Pass/run rates on 3rd down

You can add PROE, personnel rates, or target hotspots later by extending `config.SCHEME_FEATURE_COLUMNS_EXTENDED` and ensuring your CSV has those columns.

## Using clusters downstream

- **Coach ↔ scheme:** Join `scheme_cluster` to a coaching DB (team + season) to label each coach with a cluster (e.g. “Cluster 3: High motion / high PROE”).
- **Player predictions:** Use `scheme_cluster` (or cluster centroid features) as input to models that predict player stats, so scheme is part of the feature set.

## Optional: nfelo

nfelo’s Team Tendencies (PROE, personnel, formations) are a good complement. Their site is JS-heavy; adding a Selenium scraper or using their data export (if available) would allow PROE and personnel features in the same pipeline.
