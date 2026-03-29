"""
RB Correlation Analysis — Siddarth Nachannagari

Analyzes how Running Back performance grades correlate with team success metrics
(Win %, Net EPA). Output is used as context for the GM LLM agent to reason about
RB-grade-to-team-outcome relationships.
"""

import os
import pandas as pd
import numpy as np
from scipy import stats

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
DATA_FILE  = os.path.join(BASE_DIR, "../HB.csv")
OUT_CSV    = os.path.join(BASE_DIR, "RB_team_correlation.csv")
TIER_CSV   = os.path.join(BASE_DIR, "RB_tier_win_rates.csv")
MIN_TOUCHES = 50
YEARS       = (2010, 2024)

TIER_THRESHOLDS = {"Elite": 75.0, "Starter": 65.0, "Rotation": 55.0}


def assign_tier(grade):
    if grade >= TIER_THRESHOLDS["Elite"]:
        return "Elite"
    elif grade >= TIER_THRESHOLDS["Starter"]:
        return "Starter"
    elif grade >= TIER_THRESHOLDS["Rotation"]:
        return "Rotation"
    else:
        return "Reserve/Poor"


def load_data():
    df = pd.read_csv(DATA_FILE)
    df["grades_offense"] = pd.to_numeric(df["grades_offense"], errors="coerce")
    df["total_touches"]  = pd.to_numeric(df["total_touches"],  errors="coerce")
    df["Win %"]          = pd.to_numeric(
        df["Win %"].astype(str).str.replace("%", "", regex=False).str.strip(),
        errors="coerce"
    )
    df["Net EPA"]        = pd.to_numeric(df["Net EPA"], errors="coerce")
    df["age"]            = pd.to_numeric(df["age"],     errors="coerce")
    df["yards"]          = pd.to_numeric(df["yards"],   errors="coerce")
    df["touchdowns"]     = pd.to_numeric(df["touchdowns"], errors="coerce")
    df["elusive_rating"] = pd.to_numeric(df["elusive_rating"], errors="coerce")
    df["breakaway_percent"] = pd.to_numeric(df["breakaway_percent"], errors="coerce")
    df["yards_after_contact"] = pd.to_numeric(df["yards_after_contact"], errors="coerce")
    df["receptions"]     = pd.to_numeric(df["receptions"], errors="coerce")

    df = df[
        (df["total_touches"] >= MIN_TOUCHES) &
        (df["Year"] >= YEARS[0]) &
        (df["Year"] <= YEARS[1])
    ].copy()
    df["Tier"] = df["grades_offense"].apply(assign_tier)
    return df


def build_team_year_agg(df):
    """
    Aggregate per team-year:
      - best_rb_grade:    top RB grade (starter quality signal)
      - top2_avg_grade:   average of top-2 RB grades (depth signal)
      - weighted_avg_grade: touches-weighted average grade
      - top_rb_tier:      tier of the best RB
    Then merge Win % and Net EPA from the same rows.
    """
    team_success = (
        df.groupby(["Team", "Year"])[["Win %", "Net EPA"]]
        .first()
        .reset_index()
    )

    def team_agg(g):
        g_sorted = g.sort_values("grades_offense", ascending=False)
        best_grade = g_sorted.iloc[0]["grades_offense"]
        top2_avg   = g_sorted.head(2)["grades_offense"].mean()
        total_touches = g["total_touches"].sum()
        w_avg = (g["grades_offense"] * g["total_touches"]).sum() / total_touches if total_touches > 0 else np.nan
        return pd.Series({
            "best_rb_grade":    best_grade,
            "top2_avg_grade":   top2_avg,
            "weighted_avg_grade": w_avg,
            "top_rb_tier":      assign_tier(best_grade),
            "n_rb_seasons":     len(g),
        })

    agg = df.groupby(["Team", "Year"]).apply(team_agg, include_groups=False).reset_index()
    team_df = agg.merge(team_success, on=["Team", "Year"])
    return team_df


def correlate_features_with_success(df, features, success_metrics):
    """
    Pearson correlation of each feature against each success metric.
    Returns a tidy DataFrame sorted by abs correlation with Win %.
    """
    rows = []
    for feat in features:
        for metric in success_metrics:
            sub = df[[feat, metric]].copy()
            sub[feat]   = pd.to_numeric(sub[feat],   errors="coerce")
            sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
            sub = sub.dropna()
            if len(sub) < 10:
                continue
            r, p = stats.pearsonr(sub[feat], sub[metric])
            rows.append({
                "feature":        feat,
                "success_metric": metric,
                "correlation":    round(r, 4),
                "abs_correlation": round(abs(r), 4),
                "p_value":        round(p, 6),
                "n_samples":      len(sub),
            })
    result = pd.DataFrame(rows)
    return result.sort_values(["success_metric", "abs_correlation"], ascending=[True, False])


def tier_win_rates(team_df):
    """
    For each tier, compute mean Win % and Net EPA, and playoff rate (Win % > 56.25%).
    """
    rows = []
    for tier in ["Elite", "Starter", "Rotation", "Reserve/Poor"]:
        sub = team_df[team_df["top_rb_tier"] == tier]
        if len(sub) == 0:
            continue
        playoff_rate = (sub["Win %"] > 56.25).mean()  # ~9+ wins / 16-game season
        rows.append({
            "tier":             tier,
            "n_team_seasons":   len(sub),
            "mean_win_pct":     round(sub["Win %"].mean(), 2),
            "std_win_pct":      round(sub["Win %"].std(), 2),
            "mean_net_epa":     round(sub["Net EPA"].mean(), 4),
            "playoff_rate":     round(playoff_rate, 4),
        })
    return pd.DataFrame(rows)


def individual_rb_correlations(df):
    """
    Correlate individual RB stat features directly with team Win % and Net EPA.
    Useful for understanding which RB attributes matter most to team outcomes.
    """
    rb_features = [
        "grades_offense", "grades_run", "grades_pass_route", "elusive_rating",
        "yards", "yards_after_contact", "breakaway_percent", "touchdowns",
        "receptions", "total_touches",
    ]
    return correlate_features_with_success(df, rb_features, ["Win %", "Net EPA"])


def team_level_correlations(team_df):
    """
    Correlate team-aggregated RB metrics with team success.
    """
    team_features = ["best_rb_grade", "top2_avg_grade", "weighted_avg_grade"]
    return correlate_features_with_success(team_df, team_features, ["Win %", "Net EPA"])


def print_summary(ind_corr, team_corr, tier_df):
    print("\n" + "=" * 70)
    print("RB CORRELATION ANALYSIS SUMMARY")
    print("=" * 70)

    print("\n--- Top individual RB stats correlated with Win % ---")
    sub = ind_corr[ind_corr["success_metric"] == "Win %"].head(8)
    print(sub[["feature", "correlation", "p_value", "n_samples"]].to_string(index=False))

    print("\n--- Top individual RB stats correlated with Net EPA ---")
    sub = ind_corr[ind_corr["success_metric"] == "Net EPA"].head(8)
    print(sub[["feature", "correlation", "p_value", "n_samples"]].to_string(index=False))

    print("\n--- Team-level RB grade aggregates vs. team success ---")
    print(team_corr.to_string(index=False))

    print("\n--- Win rates by RB tier (team's best RB) ---")
    print(tier_df.to_string(index=False))

    # LLM-friendly summary sentences
    best_row = team_corr[team_corr["success_metric"] == "Win %"].iloc[0]
    tier_elite = tier_df[tier_df["tier"] == "Elite"]
    tier_reserve = tier_df[tier_df["tier"] == "Reserve/Poor"]

    print("\n" + "=" * 70)
    print("KEY INSIGHTS (for LLM GM agent context)")
    print("=" * 70)
    print(f"1. A team's best RB grade correlates {best_row['correlation']:+.3f} with Win % "
          f"(p={best_row['p_value']:.4f}).")

    if not tier_elite.empty:
        e = tier_elite.iloc[0]
        print(f"2. Teams with an Elite RB (grade ≥75) win {e['mean_win_pct']:.1f}% of games on average "
              f"and reach the playoffs {e['playoff_rate']*100:.1f}% of the time.")

    tier_bottom = tier_df[tier_df["tier"].isin(["Reserve/Poor", "Rotation"])].sort_values("mean_win_pct")
    if not tier_bottom.empty:
        r = tier_bottom.iloc[0]
        print(f"3. Teams whose best RB is only '{r['tier']}' win {r['mean_win_pct']:.1f}% of games "
              f"and reach the playoffs only {r['playoff_rate']*100:.1f}% of the time.")

    ind_win = ind_corr[ind_corr["success_metric"] == "Win %"].iloc[0]
    print(f"4. The strongest individual RB correlate of winning is '{ind_win['feature']}' "
          f"(r={ind_win['correlation']:+.3f}).")
    print("=" * 70)


if __name__ == "__main__":
    print("Loading RB data...")
    df = load_data()
    print(f"  {len(df)} qualifying RB seasons ({YEARS[0]}–{YEARS[1]})")

    print("Building team-year aggregates...")
    team_df = build_team_year_agg(df)
    print(f"  {len(team_df)} team-year observations")

    print("Computing correlations...")
    ind_corr  = individual_rb_correlations(df)
    team_corr = team_level_correlations(team_df)
    tier_df   = tier_win_rates(team_df)

    print_summary(ind_corr, team_corr, tier_df)

    # Save outputs
    all_corr = pd.concat([ind_corr, team_corr], ignore_index=True).sort_values(
        ["success_metric", "abs_correlation"], ascending=[True, False]
    )
    all_corr.to_csv(OUT_CSV, index=False)
    tier_df.to_csv(TIER_CSV, index=False)

    print(f"\nSaved: {OUT_CSV}")
    print(f"Saved: {TIER_CSV}")
