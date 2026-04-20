#!/usr/bin/env python3
"""
Fit / sanity-check fair-AAV knots from historical cap hits + PFF-style grades.

- Joins ``ML/cap_data.csv`` (Player, Position, Cap_Space as fraction of league cap,
  year) to position stat CSVs on (player, year).
- Drops obvious rookie-compression rows: age < 25 and cap hit below a floor, or
  cap hit in the bottom rookie band.
- Writes quantiles of estimated AAV ($M) by grade decile and optional Ridge
  suggestions — run manually when refreshing knots:

  python3 backend/agent/fit_market_curves_from_cap_data.py

Requires: pandas, numpy, scikit-learn (optional for Ridge block).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.agent.team_context import league_cap_millions  # noqa: E402

CAP_CSV = _ROOT / "backend" / "ML" / "cap_data.csv"
WR_CSV = _ROOT / "backend" / "ML" / "WR.csv"

# Map raw positions to FA API keys
POS_MAP = {
    "QB": "QB",
    "WR": "WR",
    "RB": "HB",
    "HB": "HB",
    "TE": "TE",
    "T": "T",
    "G": "G",
    "C": "C",
    "LT": "T",
    "RT": "T",
    "OLB": "ED",
    "DE": "ED",
    "ED": "ED",
    "DI": "DI",
    "DT": "DI",
    "NT": "DI",
    "LB": "LB",
    "ILB": "LB",
    "CB": "CB",
    "S": "S",
    "FS": "S",
    "SS": "S",
}


def _norm_name(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def load_wr_sample() -> pd.DataFrame:
    if not CAP_CSV.exists() or not WR_CSV.exists():
        raise FileNotFoundError("Need cap_data.csv and WR.csv under backend/ML/")
    cap = pd.read_csv(CAP_CSV)
    wr = pd.read_csv(WR_CSV)
    cap = cap.rename(columns={"year": "Year"})
    cap["player_key"] = cap["Player"].map(_norm_name)
    wr["player_key"] = wr["player"].map(_norm_name)
    m = cap.merge(
        wr,
        left_on=["player_key", "Year"],
        right_on=["player_key", "Year"],
        how="inner",
        suffixes=("_cap", "_stat"),
    )
    pos_col = "Position" if "Position" in m.columns else "Position_cap"
    cap_col = "Cap_Space_cap" if "Cap_Space_cap" in m.columns else "Cap_Space"
    m["pos_key"] = m[pos_col].map(lambda x: POS_MAP.get(str(x).split("/")[0].strip(), None))
    m = m[m["pos_key"] == "WR"]
    # Cap_Space is % of league cap (e.g. 13.88 → 13.88% of that year's cap).
    m["cap_m"] = (m[cap_col].astype(float) / 100.0) * m["Year"].map(lambda y: league_cap_millions(int(y)))
    m["grade"] = pd.to_numeric(m.get("grades_offense"), errors="coerce")
    m["age"] = pd.to_numeric(m.get("age"), errors="coerce")
    m = m.dropna(subset=["grade", "cap_m"])
    m = m[(m["grade"] >= 45) & (m["grade"] <= 100)]
    # Veteran / non-rookie-scaled: age or $ floor
    m = m[(m["age"].isna() | (m["age"] >= 25)) | (m["cap_m"] >= 4.0)]
    m = m[m["cap_m"] >= 2.0]
    return m


def grade_decile_table(df: pd.DataFrame, year_lo: int = 2018, year_hi: int = 2024) -> pd.DataFrame:
    s = df[(df["Year"] >= year_lo) & (df["Year"] <= year_hi)].copy()
    s["bin"] = pd.qcut(s["grade"], q=10, duplicates="drop")
    g = s.groupby("bin", observed=True).agg(
        grade_mid=("grade", "median"),
        aav_p50=("cap_m", "median"),
        aav_p75=("cap_m", lambda x: float(np.percentile(x, 75))),
        n=("cap_m", "count"),
    )
    return g.reset_index(drop=True)


def main() -> None:
    df = load_wr_sample()
    print(f"[fit] WR merged rows (veteran-ish filter): {len(df)}")
    tbl = grade_decile_table(df)
    print(tbl.to_string(index=False))
    try:
        from sklearn.linear_model import Ridge

        s = df[(df["Year"] >= 2018) & (df["Year"] <= 2024)]
        X = np.column_stack([np.ones(len(s)), s["grade"].values, s["grade"].values**2])
        y = s["cap_m"].values
        ridge = Ridge(alpha=10.0, fit_intercept=False)
        ridge.fit(X, y)
        print("\n[Ridge cap_m ~ 1 + g + g^2] coef:", ridge.coef_)
    except ImportError:
        print("\n(sklearn not installed — skip Ridge demo)")


if __name__ == "__main__":
    main()
