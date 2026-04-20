import os
import pandas as pd
import numpy as np
from scipy import stats

_CACHE = {}

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NFL_RANKINGS_PATH = os.path.join(DATA_DIR, "data", "nflpowerrankings.csv")
TEAM_TENDENCIES_PATH = os.path.join(DATA_DIR, "data", "TeamTendencies.csv")

POSITION_DEFS = {
    "qb": {
        "file": os.path.join(DATA_DIR, "QB.csv"),
        "grade_field": "grades_offense",
        "snap_field": "passing_snaps",
        "threshold": 50,
    },
    "hb": {
        "file": os.path.join(DATA_DIR, "HB.csv"),
        "grade_field": "grades_offense",
        "snap_field": "run_plays",  # proxy for snaps
        "threshold": 100,
    },
    "wr": {
        "file": os.path.join(DATA_DIR, "WR.csv"),
        "grade_field": "grades_offense",
        "snap_field": "total_snaps",
        "threshold": 100,
    },
    "te": {
        # use dedicated TE PFF source with true TE snaps and grades
        "file": os.path.join(DATA_DIR, "TightEnds", "TEPFF.csv"),
        "grade_field": "grades_offense",
        "snap_field": "total_snaps",
        "threshold": 100,
    },
    "ol": {
        "file": [
            os.path.join(DATA_DIR, "G.csv"),
            os.path.join(DATA_DIR, "C.csv"),
            os.path.join(DATA_DIR, "T.csv"),
        ],
        "grade_field": "grades_offense",
        "snap_field": "snap_counts_offense",
        "threshold": 200,
    },
    "di": {
        "file": os.path.join(DATA_DIR, "DI.csv"),
        "grade_field": "grades_defense",
        "snap_field": "snap_counts_defense",
        "threshold": 100,
    },
    "ed": {
        "file": os.path.join(DATA_DIR, "ED.csv"),
        "grade_field": "grades_defense",
        "snap_field": "snap_counts_defense",
        "threshold": 100,
    },
    "lb": {
        "file": os.path.join(DATA_DIR, "LB.csv"),
        "grade_field": "grades_defense",
        "snap_field": "snap_counts_defense",
        "threshold": 100,
    },
    "cb": {
        "file": os.path.join(DATA_DIR, "CB.csv"),
        "grade_field": "grades_defense",
        "snap_field": "snap_counts_defense",
        "threshold": 100,
    },
    "s": {
        "file": os.path.join(DATA_DIR, "S.csv"),
        "grade_field": "grades_defense",
        "snap_field": "snap_counts_defense",
        "threshold": 100,
    },
}

TEAM_ABBREV_FIXES = {
    ("OAK", 2020): "LV",
    ("STL", 2016): "LAR",
    ("SD", 2017): "LAC",
}

TEAM_NAME_TO_ABBR = {
    "49ERS": "SF",
    "CARDINALS": "ARI",
    "BILLS": "BUF",
    "BROWNS": "CLE",
    "BENGALS": "CIN",
    "BRONCOS": "DEN",
    "BEARS": "CHI",
    "RAVENS": "BAL",
    "CHIEFS": "KC",
    "COLTS": "IND",
    "COWBOYS": "DAL",
    "DOLPHINS": "MIA",
    "EAGLES": "PHI",
    "FALCONS": "ATL",
    "JAGUARS": "JAX",
    "JAX": "JAC",
    "WSH": "WAS",
    "LIONS": "DET",
    "PACKERS": "GB",
    "PATRIOTS": "NE",
    "RAIDERS": "OAK",
    "SAINTS": "NO",
    "SEAHAWKS": "SEA",
    "STEELERS": "PIT",
    "TEXANS": "HOU",
    "VIKINGS": "MIN",
    "TITANS": "TEN",
    "REDSKINS": "WAS",
    "COMMANDERS": "WAS",
    "PANTHERS": "CAR",
    "BUCCANEERS": "TB",
    "GIANTS": "NYG",
    "JETS": "NYJ",
    "CHARGERS": "LAC",  # later becomes LAC from 2017
    "RAMS": "LAR",  # later becomes LAR from 2016
}


def normalize_team(team: str, season: int):
    if pd.isna(team):
        return team
    team = str(team).strip().upper()

    # map explicit full names when present (e.g., Jaguars -> JAX, etc.)
    if team in TEAM_NAME_TO_ABBR:
        team = TEAM_NAME_TO_ABBR[team]

    # canonical relocations and renames across datasets
    if team in {"OAK", "LV"}:
        return "LV"
    if team in {"STL", "LAR", "RAMS"}:
        return "LAR"
    if team in {"SD", "LAC", "CHARGERS"}:
        return "LAC"
    if team in {"JAX", "JAGUARS"}:
        return "JAC"
    if team in {"WSH", "REDSKINS", "COMMANDERS"}:
        return "WAS"

    # fallback preserving existing codes
    return team


# for backward compatibility
_normalize_team = normalize_team


def _safe_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _load_position_aggregates():
    position_dfs = {}

    for pos, defn in POSITION_DEFS.items():
        path = defn["file"]

        if isinstance(path, list):
            dfs = []
            for p in path:
                if os.path.exists(p):
                    dfs.append(pd.read_csv(p))
            if not dfs:
                continue
            df = pd.concat(dfs, ignore_index=True)
        else:
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
        df = df.rename(columns={
            "Team": "team",
            "Year": "season",
            "year": "season",
        })

        # normalize team names and season fields
        if "team" in df.columns and "season" in df.columns:
            df["team"] = df.apply(lambda r: normalize_team(str(r.get("team", "")).strip(), int(r.get("season", np.nan))) if not pd.isna(r.get("season")) else r.get("team"), axis=1)
            df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")

        grade_col = defn["grade_field"]
        snap_col = defn["snap_field"]
        if grade_col not in df.columns or snap_col not in df.columns:
            continue

        df[grade_col] = pd.to_numeric(df[grade_col], errors="coerce")
        df[snap_col] = pd.to_numeric(df[snap_col], errors="coerce")

        # filter by threshold
        threshold = defn["threshold"]
        df = df[df[snap_col] >= threshold]

        # aggregate snap-weighted mean
        agg = df.dropna(subset=[grade_col, snap_col]).copy()
        agg["weighted_grade_prod"] = agg[grade_col] * agg[snap_col]
        grouped = agg.groupby(["team", "season"]).agg(
            total_snap=(snap_col, "sum"),
            weighted_grade_sum=("weighted_grade_prod", "sum"),
        ).reset_index()

        grouped[f"{pos}_grade_mean"] = grouped.apply(
            lambda r: float(r["weighted_grade_sum"] / r["total_snap"]) if r["total_snap"] > 0 else np.nan,
            axis=1,
        )

        position_dfs[pos] = grouped[["team", "season", f"{pos}_grade_mean"]]

    # keep CB and S separate for position-level correlation analysis
    return position_dfs


def _load_data():
    if "merged" in _CACHE:
        return _CACHE["merged"]

    # load nfl power rankings
    team_df = pd.read_csv(NFL_RANKINGS_PATH)
    team_df = team_df.rename(columns={
        "Team": "team",
        "Season": "season",
        "nfelo": "nfelo",
        "Wins": "wins",
    })

    if "team" in team_df.columns:
        team_df["team"] = team_df.apply(lambda r: normalize_team(str(r.get("team", "")).strip(), int(r.get("season", np.nan))) if not pd.isna(r.get("season")) else r.get("team"), axis=1)
        team_df["season"] = pd.to_numeric(team_df["season"], errors="coerce").astype("Int64")

    # compute win percent
    if "wins" in team_df.columns:
        team_df["wins"] = pd.to_numeric(team_df["wins"], errors="coerce")
        team_df["games"] = team_df.apply(lambda r: 16 if int(r["season"] or 0) < 2021 else 17, axis=1)
        team_df["win_pct"] = team_df.apply(lambda r: float(r["wins"] / r["games"]) if pd.notna(r["wins"]) and r["games"] > 0 else np.nan, axis=1)

    # load team tendencies
    tendencies = pd.read_csv(TEAM_TENDENCIES_PATH)
    tendencies = tendencies.rename(columns={
        "Team": "team",
        "Season": "season",
        "PROE": "proe",
        "aDOT": "adot",
        "Pass Rate": "pass_rate",
        "Rush Rate": "rush_rate",
        "TOP": "time_of_possession",
    })

    if "team" in tendencies.columns:
        tendencies["team"] = tendencies.apply(lambda r: normalize_team(str(r.get("team", "")).strip(), int(r.get("season", np.nan))) if not pd.isna(r.get("season")) else r.get("team"), axis=1)
        tendencies["season"] = pd.to_numeric(tendencies["season"], errors="coerce").astype("Int64")

    num_cols = [
        "pass_rate", "rush_rate", "proe", "adot", "nfelo", "Play", "Pass", "Rush", "For", "Against", "Dif"
    ]
    _safe_numeric(team_df, num_cols)
    _safe_numeric(tendencies, ["pass_rate", "rush_rate", "proe", "adot"])

    tt_cols = ["team", "season"] + [c for c in tendencies.columns if c not in ["team", "season"]]
    merged = pd.merge(
        team_df,
        tendencies[tt_cols],
        on=["team", "season"],
        how="left",
    )

    # attach positional grades
    position_dfs = _load_position_aggregates()

    # Coverage check for required positions
    if "wr" in position_dfs:
        wr_df = position_dfs["wr"]
        print("WR team-seasons:", wr_df.shape)
        print("unique teams:", wr_df.team.nunique())
        print("unique seasons:", wr_df.season.nunique())

    for pos, pos_df in position_dfs.items():
        merged = pd.merge(merged, pos_df, on=["team", "season"], how="left")

    # compute deltas by team
    for pos in ["qb", "hb", "wr", "te", "ol", "di", "ed", "lb", "cb", "s"]:
        col = f"{pos}_grade_mean"
        delta_col = f"{pos}_grade_delta"
        if col in merged.columns:
            merged = merged.sort_values(["team", "season"])
            merged[delta_col] = merged.groupby("team")[col].diff().fillna(0.0)

    _CACHE["merged"] = merged
    return merged


def compute_correlation(
    feature: str,
    target: str = "win_pct",
    filter_column: str | None = None,
    filter_condition: str | None = None,
    method: str = "pearson",
) -> dict:
    # support both call styles:
    # compute_correlation(feature, target, method)
    # compute_correlation(feature, target, filter_column, filter_condition, method)
    if filter_column is not None and filter_column in {"pearson", "spearman", "quartile_breakdown"}:
        method = filter_column
        filter_column = None
        filter_condition = None

    df = _load_data()
    if feature not in df.columns:
        return {
            "error": f"Feature '{feature}' not found. Available features: {sorted(df.columns.tolist())}"
        }
    if target not in df.columns:
        return {
            "error": f"Target '{target}' not found. Available targets: {sorted(df.columns.tolist())}"
        }

    cols = ["team", "season", "wins", "win_pct"]
    if feature not in cols:
        cols.append(feature)
    if target not in cols:
        cols.append(target)
    if filter_column and filter_column not in cols:
        cols.append(filter_column)
    query_df = df[cols].copy()

    filter_applied = None
    if filter_column and filter_condition:
        if filter_column not in df.columns:
            return {"error": f"Filter column '{filter_column}' not found. Available columns: {sorted(df.columns.tolist())}"}

        op = None
        cond = filter_condition.strip()
        for candidate in ["<=", ">=", "<", ">", "=="]:
            if cond.startswith(candidate):
                op = candidate
                num_s = cond[len(candidate):].strip()
                break
        if op is None:
            return {"error": "filter_condition should use one of <, >, ==, <=, >="}

        try:
            num = float(num_s)
        except ValueError:
            return {"error": f"filter_condition value '{num_s}' is not numeric"}

        if op == "<":
            query_df = query_df[query_df[filter_column] < num]
        elif op == ">":
            query_df = query_df[query_df[filter_column] > num]
        elif op == "<=":
            query_df = query_df[query_df[filter_column] <= num]
        elif op == ">=":
            query_df = query_df[query_df[filter_column] >= num]
        elif op == "==":
            query_df = query_df[query_df[filter_column] == num]

        filter_applied = f"{filter_column} {op} {num}"

    query_df = query_df.dropna(subset=[feature, target])
    sample_size = int(len(query_df))

    warning = None
    if sample_size < 20:
        warning = f"Only {sample_size} team-seasons matched filter. Correlation may not be reliable."

    corr_value = None
    p_value = None

    if sample_size < 2:
        corr_value = None
        p_value = None
    else:
        if method == "pearson":
            corr_value, p_value = stats.pearsonr(query_df[feature], query_df[target])
        elif method == "spearman":
            corr_value, p_value = stats.spearmanr(query_df[feature], query_df[target])
        elif method == "quartile_breakdown":
            corr_value, p_value = stats.pearsonr(query_df[feature], query_df[target])
        else:
            return {"error": "method must be pearson, spearman, or quartile_breakdown"}

    quartile_breakdown = []
    if sample_size > 0:
        try:
            query_df = query_df.assign(_q=pd.qcut(query_df[feature], 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]))
            for quartile_label, grp in query_df.groupby("_q"):
                if grp.empty:
                    continue
                low = grp[feature].min()
                high = grp[feature].max()
                quartile_breakdown.append({
                    "quartile": quartile_label,
                    "grade_range": f"{float(low):.1f} - {float(high):.1f}",
                    "mean_win_pct": float(grp["win_pct"].mean()) if "win_pct" in grp.columns else None,
                    "mean_wins": float(grp["wins"].mean()) if "wins" in grp.columns else None,
                    "sample_size": int(len(grp)),
                })
        except Exception:
            quartile_breakdown = []

    interpretation = "Correlation is weak or not computed." 
    if corr_value is not None:
        direction = "positive" if corr_value > 0 else "negative" if corr_value < 0 else "no"
        interpretation = f"{method.capitalize()} correlation between {feature} and {target} is {corr_value:.3f} (p={p_value:.3f}), indicating {direction} association." 

    out = {
        "feature": feature,
        "target": target,
        "method": method,
        "correlation": float(corr_value) if corr_value is not None else None,
        "p_value": float(p_value) if p_value is not None else None,
        "sample_size": sample_size,
        "filter_applied": filter_applied,
        "quartile_breakdown": quartile_breakdown,
        "interpretation": interpretation,
    }

    if warning:
        out["warning"] = warning
    return out


def get_win_pct_by_grade_delta(position: str, delta_threshold: float) -> dict:
    pos = position.lower()
    merged = _load_data()
    delta_col = f"{pos}_grade_delta"
    if delta_col not in merged.columns:
        return {"error": f"Position '{position}' not found (expected delta column {delta_col})."}

    history = merged[["team", "season", "win_pct", "wins", delta_col]].copy().sort_values(["team", "season"])
    history["win_pct_next"] = history.groupby("team")["win_pct"].shift(-1)
    history["wins_next"] = history.groupby("team")["wins"].shift(-1)

    # filter for only seasons where next year exists
    history = history.dropna(subset=["win_pct_next"])

    improved = history[history[delta_col] >= delta_threshold]
    not_improved = history[history[delta_col] < delta_threshold]

    def summarize(df):
        if df.empty:
            return {"count": 0, "avg_win_pct_change": None, "avg_wins_change": None}
        return {
            "count": int(len(df)),
            "avg_win_pct_change": float((df["win_pct_next"] - df["win_pct"]).mean()),
            "avg_wins_change": float((df["wins_next"] - df["wins"]).mean()),
        }

    improved_summary = summarize(improved)
    not_improved_summary = summarize(not_improved)

    examples = improved.sort_values(delta_col, ascending=False).head(5)
    example_rows = []
    for _, r in examples.iterrows():
        example_rows.append({
            "team": r["team"],
            "season": int(r["season"]),
            "delta": float(r[delta_col]),
            "win_pct_change": float(r["win_pct_next"] - r["win_pct"]),
        })

    interpretation = "Teams that improved positional grade by threshold have stronger future win_pct growth." if improved_summary["count"] > 0 else "No teams matched the improvement threshold."

    return {
        "position": pos,
        "delta_threshold": float(delta_threshold),
        "teams_that_improved": {**improved_summary, "examples": example_rows},
        "teams_that_did_not": not_improved_summary,
        "interpretation": interpretation,
    }


def get_team_context(team: str, season: int) -> dict:
    merged = _load_data()
    team_norm = normalize_team(team, season)
    row = merged[(merged["team"] == team_norm) & (merged["season"] == int(season))]
    if row.empty:
        return {"error": f"Team {team} season {season} not found."}

    r = row.iloc[0]

    pos_keys = ["qb", "hb", "wr", "te", "ol", "di", "ed", "lb", "cb", "s"]
    positional_grades = {p: float(r.get(f"{p}_grade_mean")) if pd.notna(r.get(f"{p}_grade_mean")) else None for p in pos_keys}
    positional_grade_deltas = {p: float(r.get(f"{p}_grade_delta")) if pd.notna(r.get(f"{p}_grade_delta")) else None for p in pos_keys}

    sorted_by_grade = sorted(
        ((p, v) for p, v in positional_grades.items() if v is not None),
        key=lambda x: x[1]
    )

    weakest = [p for p, _ in sorted_by_grade[:2]]
    strongest = [p for p, _ in sorted_by_grade[-2:]][::-1]

    return {
        "team": team_norm,
        "season": int(season),
        "nfelo": float(r.get("nfelo")) if pd.notna(r.get("nfelo")) else None,
        "win_pct": float(r.get("win_pct")) if pd.notna(r.get("win_pct")) else None,
        "wins": float(r.get("wins")) if pd.notna(r.get("wins")) else None,
        "off_epa_play": float(r.get("Play")) if "Play" in r and pd.notna(r.get("Play")) else None,
        "off_epa_pass": float(r.get("Pass")) if "Pass" in r and pd.notna(r.get("Pass")) else None,
        "off_epa_rush": float(r.get("Rush")) if "Rush" in r and pd.notna(r.get("Rush")) else None,
        "def_epa_play": None,
        "def_epa_pass": None,
        "def_epa_rush": None,
        "point_diff": float(r.get("Dif")) if "Dif" in r and pd.notna(r.get("Dif")) else None,
        "proe": float(r.get("proe")) if pd.notna(r.get("proe")) else None,
        "adot": float(r.get("adot")) if pd.notna(r.get("adot")) else None,
        "personnel_stability": None,
        "positional_grades": positional_grades,
        "positional_grade_deltas": positional_grade_deltas,
        "weakest_position_groups": weakest,
        "strongest_position_groups": strongest,
    }


def get_positional_grade_distribution(position: str, season: int | None = None) -> dict:
    pos = position.lower()
    merged = _load_data()

    source = merged
    col = f"{pos}_grade_mean"
    if col not in source.columns:
        return {"error": f"Position '{position}' not found. Available positions: qb,hb,wr,te,ol,di,ed,lb,cb,s,secondary."}

    if season is not None:
        source = source[source["season"] == int(season)]

    source = source.dropna(subset=[col])
    if source.empty:
        return {"error": "No data for requested position/season."}

    values = source[col].astype(float)

    percentiles = {
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
    }

    return {
        "position": pos,
        "season": int(season) if season is not None else None,
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "percentiles": percentiles,
        "elite_threshold": float(percentiles["p75"]),
        "below_average_threshold": float(percentiles["p25"]),
    }


def export_sample_correlations(output_path: str = "correlation_audit.csv") -> None:
    jobs = [
        ("qb_grade_mean", "win_pct", "qb_grade_win_pct"),
        ("wr_grade_mean", "win_pct", "wr_grade_win_pct"),
        ("ol_grade_mean", "win_pct", "ol_grade_win_pct"),
        ("te_grade_mean", "win_pct", "te_grade_win_pct"),
        ("hb_grade_mean", "win_pct", "hb_grade_win_pct"),
        ("ed_grade_mean", "win_pct", "ed_grade_win_pct"),
        ("di_grade_mean", "win_pct", "di_grade_win_pct"),
        ("lb_grade_mean", "win_pct", "lb_grade_win_pct"),
        ("cb_grade_mean", "win_pct", "cb_grade_win_pct"),
        ("s_grade_mean", "win_pct", "s_grade_win_pct"),

        # key offensive and defensive EPA correlations
        ("qb_grade_mean", "Pass", "qb_off_pass_epa"),
        ("hb_grade_mean", "Rush", "hb_off_run_epa"),
        ("cb_grade_mean", "Pass.1", "cb_def_pass_epa"),
        ("s_grade_mean", "Pass.1", "s_def_pass_epa"),
        ("di_grade_mean", "Rush.1", "di_def_run_epa"),

        # more team-level metrics
        ("Play", "win_pct", "off_epa_play"),
        ("Pass", "win_pct", "off_epa_pass"),
        ("Rush", "win_pct", "off_epa_rush"),
        ("For", "win_pct", "point_for"),
        ("Against", "win_pct", "point_against"),
        ("Dif", "win_pct", "point_diff"),
        ("nfelo", "win_pct", "nfelo"),
        ("proe", "win_pct", "proe"),
    ]

    rows = []
    for feature, target, label in jobs:
        result = compute_correlation(feature, target=target)
        if "error" in result:
            continue

        quartiles = {q: None for q in ["q1_win_pct", "q2_win_pct", "q3_win_pct", "q4_win_pct"]}
        for q in result.get("quartile_breakdown", []):
            quartile_label = q.get("quartile", "")
            win_pct = q.get("mean_win_pct")
            if "Q1" in quartile_label:
                quartiles["q1_win_pct"] = win_pct
            elif "Q2" in quartile_label:
                quartiles["q2_win_pct"] = win_pct
            elif "Q3" in quartile_label:
                quartiles["q3_win_pct"] = win_pct
            elif "Q4" in quartile_label:
                quartiles["q4_win_pct"] = win_pct

        rows.append({
            "feature": label,
            "base_feature": feature,
            "target": target,
            "correlation": result.get("correlation"),
            "p_value": result.get("p_value"),
            "sample_size": result.get("sample_size"),
            "q1_win_pct": quartiles["q1_win_pct"],
            "q2_win_pct": quartiles["q2_win_pct"],
            "q3_win_pct": quartiles["q3_win_pct"],
            "q4_win_pct": quartiles["q4_win_pct"],
            "interpretation": result.get("interpretation"),
        })

    out_df = pd.DataFrame(rows)
    out_df["abs_correlation"] = out_df["correlation"].abs()
    out_df = out_df.sort_values(by="abs_correlation", ascending=False).drop(columns=["abs_correlation"])
    out_df.to_csv(output_path, index=False)

    # optional split files for offense/defense positional and key metric correlations
    offense_feats = [
        "qb_grade_win_pct", "wr_grade_win_pct", "ol_grade_win_pct", "te_grade_win_pct", "hb_grade_win_pct",
        "qb_pass_epa", "hb_run_epa", "off_epa_play", "off_epa_pass", "off_epa_rush", "point_diff", "nfelo", "proe",
    ]
    defense_feats = [
        "di_grade_win_pct", "lb_grade_win_pct", "ed_grade_win_pct", "cb_grade_win_pct", "s_grade_win_pct",
        "cb_pass_epa", "s_pass_epa", "di_run_epa", "point_for", "point_against", "point_diff", "nfelo", "proe",
    ]

    if not out_df.empty:
        out_df[out_df["feature"].isin(offense_feats)].to_csv("correlation_audit_offense.csv", index=False)
        out_df[out_df["feature"].isin(defense_feats)].to_csv("correlation_audit_defense.csv", index=False)


if __name__ == "__main__":
    print("Test 1: basic correlation")
    print(compute_correlation("wr_grade_mean", method="pearson"))

    print("\nTest 2: filtered correlation")
    print(compute_correlation("ol_grade_mean", filter_column="proe", filter_condition="< -0.02", method="spearman"))

    print("\nTest 3: delta impact analysis")
    print(get_win_pct_by_grade_delta("wr", delta_threshold=5.0))

    print("\nTest 4: team context")
    print(get_team_context("BAL", 2023))
    print("Exporting correlation audit CSVs...")
    export_sample_correlations()
    print("Correlation audit exported to correlation_audit.csv (plus offense/defense split files).")