#!/usr/bin/env python3
"""
Build tier + position player rankings for a Free Agency analysis year.

Uses the same CSV paths, grade/snap columns, and tier cutoffs as
`team_summary.POS_CFG` and `grade_projection.grade_to_tier_universal`.

Season row: latest `Year` in each CSV that is <= clamp(analysis_year).
(For analysis_year=2025 and current ML files, this is 2024.)
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

import pandas as pd

from backend.agent.api_year_utils import clamp_analysis_year, effective_year_for_df
from backend.agent.grade_projection import grade_to_tier_universal
from backend.agent.team_summary import POS_CFG

TIER_ORDER = ("Elite", "Good", "Starter", "Rotation/backup")

_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_REPO = os.path.abspath(os.path.join(_BASE, ".."))
REPORTS_DIR = os.path.join(_REPO, "reports")


def _snap_col_for_df(df: pd.DataFrame, preferred: str) -> str:
    if preferred in df.columns:
        return preferred
    for c in (
        "snap_counts_offense",
        "snap_counts_defense",
        "passing_snaps",
        "total_snaps",
        "routes",
        "targets",
    ):
        if c in df.columns:
            return c
    return preferred


def _load_position_df(pos_key: str) -> tuple[pd.DataFrame, str, str, int]:
    cfg = POS_CFG[pos_key]
    path = cfg["path"]
    df = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    if df.empty or "player" not in df.columns:
        return df, cfg["grade"], cfg["snaps"], 0
    y_eff = effective_year_for_df(df, None, "Year")
    if y_eff is None:
        return df, cfg["grade"], cfg["snaps"], 0
    sub = df[pd.to_numeric(df["Year"], errors="coerce") == y_eff].copy()
    snap_col = _snap_col_for_df(sub, cfg["snaps"])
    return sub, cfg["grade"], snap_col, int(y_eff)


def _aggregate_players(
    sub: pd.DataFrame,
    grade_col: str,
    snap_col: str,
    min_snaps: float,
) -> pd.DataFrame:
    if sub.empty or grade_col not in sub.columns:
        return pd.DataFrame()
    g = pd.to_numeric(sub[grade_col], errors="coerce")
    sn = pd.to_numeric(sub.get(snap_col, 0), errors="coerce").fillna(0.0).clip(lower=0.0)
    work = sub.assign(_grade=g, _snap=sn).dropna(subset=["_grade", "player"])
    rows: list[dict] = []
    for player, grp in work.groupby(work["player"].astype(str).str.strip()):
        if not player:
            continue
        w = grp["_snap"].clip(lower=1.0)
        grade = float((grp["_grade"] * w).sum() / w.sum())
        total_snap = float(grp["_snap"].sum())
        if total_snap < min_snaps:
            continue
        if "Team" in grp.columns:
            mx = grp.loc[grp["_snap"].idxmax(), "Team"]
            primary_team = str(mx) if pd.notna(mx) else ""
            teams = ", ".join(sorted({str(t) for t in grp["Team"].dropna().unique()}))
        else:
            primary_team, teams = "", ""
        rows.append(
            {
                "player": player,
                "grade": round(grade, 2),
                "snaps": round(total_snap, 1),
                "primary_team": primary_team,
                "teams": teams,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("grade", ascending=False).reset_index(drop=True)
    out["rank_overall"] = range(1, len(out) + 1)
    out["tier"] = out["grade"].apply(grade_to_tier_universal)
    out["rank_in_tier"] = out.groupby("tier").cumcount() + 1
    return out


def build_report(analysis_year: int, min_snaps: float) -> tuple[str, pd.DataFrame, dict]:
    ay = clamp_analysis_year(analysis_year)
    lines: list[str] = []
    meta: dict = {"analysis_year": ay, "positions": {}}
    all_rows: list[pd.DataFrame] = []

    lines.append("# Free Agency — position rankings by tier")
    lines.append("")
    lines.append(
        "- **Regenerate:** from repo root, `PYTHONPATH=. python backend/agent/export_fa_tier_rankings.py "
        "[--analysis-year 2025] [--min-snaps 100]`"
    )
    lines.append(f"- **Generated (UTC):** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}")
    lines.append(f"- **Requested analysis_year:** {analysis_year} (clamped to {ay})")
    lines.append("- **Tier cutoffs (same as FA UI):** Elite ≥80 · Good ≥74 · Starter ≥62 · else Rotation/backup")
    lines.append(
        "- **Season (`Year` in ML CSVs):** per position, latest season with data such that `Year ≤ analysis_year` "
        "(see each section; if 2025 rows are absent, this is **2024**)."
    )
    lines.append(
        f"- **Eligibility:** snap-weighted grade from listed snap column; players with **total snaps < {min_snaps:g}** "
        "in that season are omitted."
    )
    lines.append("")

    for pos in sorted(POS_CFG.keys()):
        sub, grade_col, snap_col, y_used = _load_position_df(pos)
        label = POS_CFG[pos]["label"]
        meta["positions"][pos] = {
            "season_year": y_used,
            "grade_col": grade_col,
            "snap_col": snap_col,
            "n_players_ranked": 0,
        }
        lines.append(f"## {pos} — {label}")
        lines.append("")
        lines.append(f"- **Season used:** `{y_used}`")
        lines.append(f"- **Grade column:** `{grade_col}` · **Snap column:** `{snap_col}`")
        lines.append("")

        agg = _aggregate_players(sub, grade_col, snap_col, min_snaps)
        meta["positions"][pos]["n_players_ranked"] = int(len(agg))
        if agg.empty:
            lines.append("_No qualifying players (missing data or below snap threshold)._")
            lines.append("")
            continue

        part = agg.assign(position_key=pos, position_label=label, season_year=y_used)
        all_rows.append(part)

        for tier in TIER_ORDER:
            bucket = agg[agg["tier"] == tier]
            lines.append(f"### {tier} ({len(bucket)} players)")
            lines.append("")
            if bucket.empty:
                lines.append("_None._")
                lines.append("")
                continue
            lines.append("| rank_pos | rank_in_tier | player | grade | snaps | primary_team |")
            lines.append("|---:|---:|---|---:|---:|---|")
            for _, r in bucket.iterrows():
                lines.append(
                    f"| {int(r['rank_overall'])} | {int(r['rank_in_tier'])} | {r['player']} | "
                    f"{r['grade']:.2f} | {r['snaps']:.0f} | {r['primary_team']} |"
                )
            lines.append("")

    combined = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    return "\n".join(lines), combined, meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Export FA tier rankings by position.")
    ap.add_argument("--analysis-year", type=int, default=2025)
    ap.add_argument("--min-snaps", type=float, default=100.0)
    ap.add_argument("--out-md", type=str, default="")
    ap.add_argument("--out-csv", type=str, default="")
    args = ap.parse_args()

    os.makedirs(REPORTS_DIR, exist_ok=True)
    stem = f"fa_tier_rankings_analysis_year_{clamp_analysis_year(args.analysis_year)}"
    out_md = args.out_md or os.path.join(REPORTS_DIR, f"{stem}.md")
    out_csv = args.out_csv or os.path.join(REPORTS_DIR, f"{stem}.csv")

    md, df, _meta = build_report(args.analysis_year, args.min_snaps)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    if not df.empty:
        df.to_csv(out_csv, index=False)
    print(f"Wrote {out_md}")
    if not df.empty:
        print(f"Wrote {out_csv}")
    else:
        print("No CSV (no qualifying rows).")


if __name__ == "__main__":
    main()
