#!/usr/bin/env python3
"""
Build tier + position player rankings for a Free Agency analysis year.

Uses `team_summary.POS_CFG` paths, `grade_projection.grade_to_tier_universal`
tiers, and the **same composite grade** as each position's FA
``predict_performance`` (PFF model + stats blend, health / inactivity; ED/DI
include the ML model grade).

Season row: latest `Year` in each CSV that is <= clamp(analysis_year).
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

import pandas as pd

from backend.agent.api_year_utils import LATEST_ANALYSIS_YEAR, clamp_analysis_year, effective_year_for_df
from backend.agent.fa_tier_export_composite import composite_for_player_row
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


def _load_position_frames(
    pos_key: str, analysis_year: int
) -> tuple[pd.DataFrame, pd.DataFrame, str, str, int]:
    cfg = POS_CFG[pos_key]
    path = cfg["path"]
    df = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    if df.empty or "player" not in df.columns:
        return df, pd.DataFrame(), cfg["grade"], cfg["snaps"], 0
    y_eff = effective_year_for_df(df, analysis_year, "Year")
    if y_eff is None:
        return df, pd.DataFrame(), cfg["grade"], cfg["snaps"], 0
    sub = df[pd.to_numeric(df["Year"], errors="coerce") == y_eff].copy()
    snap_src = sub if not sub.empty else df
    snap_col = _snap_col_for_df(snap_src, cfg["snaps"])
    return df, sub, cfg["grade"], snap_col, int(y_eff)


def ml_data_year_range() -> tuple[int, int]:
    """Min / max `Year` present across `POS_CFG` CSVs (ignores missing files)."""
    lo: int | None = None
    hi: int | None = None
    for cfg in POS_CFG.values():
        path = cfg["path"]
        if not os.path.exists(path):
            continue
        try:
            ydf = pd.read_csv(path, usecols=["Year"])
        except (ValueError, KeyError, pd.errors.EmptyDataError):
            t = pd.read_csv(path)
            if "Year" not in t.columns:
                continue
            ydf = t[["Year"]]
        ys = pd.to_numeric(ydf["Year"], errors="coerce").dropna()
        if ys.empty:
            continue
        y_min, y_max = int(ys.min()), int(ys.max())
        lo = y_min if lo is None else min(lo, y_min)
        hi = y_max if hi is None else max(hi, y_max)
    if lo is None or hi is None:
        return 2020, LATEST_ANALYSIS_YEAR
    return lo, hi


def export_analysis_year_range() -> tuple[int, int]:
    """Inclusive analysis years: from earliest ML season through `LATEST_ANALYSIS_YEAR`."""
    lo, _hi_data = ml_data_year_range()
    return lo, int(LATEST_ANALYSIS_YEAR)


def _aggregate_players_fa_composite(
    pos_key: str,
    full_df: pd.DataFrame,
    sub: pd.DataFrame,
    snap_col: str,
    analysis_year: int,
    min_snaps: float,
) -> pd.DataFrame:
    if sub.empty or "player" not in sub.columns:
        return pd.DataFrame()
    sn = pd.to_numeric(sub.get(snap_col, 0), errors="coerce").fillna(0.0).clip(lower=0.0)
    work = sub.assign(_snap=sn)
    rows: list[dict] = []
    ay = clamp_analysis_year(analysis_year)
    for player, grp in work.groupby(work["player"].astype(str).str.strip()):
        if not player:
            continue
        total_snap = float(grp["_snap"].sum())
        if total_snap < min_snaps:
            continue
        if "Team" in grp.columns:
            mx = grp.loc[grp["_snap"].idxmax(), "Team"]
            primary_team = str(mx) if pd.notna(mx) else ""
            teams = ", ".join(sorted({str(t) for t in grp["Team"].dropna().unique()}))
        else:
            primary_team, teams = "", ""

        comp = composite_for_player_row(pos_key, player, full_df, ay)
        if not comp:
            continue
        g = float(comp["composite_grade"])
        mg = comp.get("model_grade")
        sg = comp.get("stats_grade")
        if isinstance(mg, (int, float)):
            mg = round(float(mg), 2)
        if isinstance(sg, (int, float)):
            sg = round(float(sg), 2)
        rows.append(
            {
                "player": player,
                "grade": round(g, 2),
                "model_grade": mg,
                "stats_grade": sg,
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


def _fmt_opt_num(v: object) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    try:
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return "—"


def build_report(analysis_year: int, min_snaps: float) -> tuple[str, pd.DataFrame, dict]:
    ay = clamp_analysis_year(analysis_year)
    lines: list[str] = []
    meta: dict = {"analysis_year": ay, "positions": {}}
    all_rows: list[pd.DataFrame] = []

    lines.append("# Free Agency — position rankings by tier")
    lines.append("")
    lines.append(
        "- **Regenerate:** from repo root, `PYTHONPATH=. python backend/agent/export_fa_tier_rankings.py "
        "[--analysis-year 2025] [--min-snaps 100]`, "
        "`--year-min 2020 --year-max 2025`, or `--all-analysis-years`."
    )
    lines.append(f"- **Generated (UTC):** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}")
    lines.append(f"- **Requested analysis_year:** {analysis_year} (clamped to {ay})")
    lines.append("- **Tier cutoffs (same as FA UI):** Elite ≥80 · Good ≥74 · Starter ≥62 · else Rotation/backup")
    lines.append(
        "- **Grade (composite):** same pipeline as FA `/evaluate` with **no team selected** — each position's "
        "`predict_performance` (PFF-style model grade + stats grade, weights vary by position; QB adds sample "
        "reliability and volume logic), then health and inactivity adjustments. **ED/DI** use the transformer "
        "ML `predicted_grade` as the model component."
    )
    lines.append(
        "- **Season (`Year` in ML CSVs):** per position, latest season with data such that `Year ≤ analysis_year` "
        "(see each section header)."
    )
    lines.append(
        f"- **Eligibility:** players with **total snaps < {min_snaps:g}** in that season row are omitted; "
        "composite still uses full history through that season (via `history_as_of_year`)."
    )
    lines.append("")

    for pos in sorted(POS_CFG.keys()):
        full_df, sub, grade_col, snap_col, y_used = _load_position_frames(pos, ay)
        label = POS_CFG[pos]["label"]
        meta["positions"][pos] = {
            "season_year": y_used,
            "pff_grade_column": grade_col,
            "snap_col": snap_col,
            "n_players_ranked": 0,
        }
        lines.append(f"## {pos} — {label}")
        lines.append("")
        lines.append(f"- **Season used:** `{y_used}`")
        lines.append(
            f"- **PFF column (model anchor):** `{grade_col}` · **Snap column (volume filter):** `{snap_col}`"
        )
        lines.append("")

        agg = _aggregate_players_fa_composite(pos, full_df, sub, snap_col, ay, min_snaps)
        meta["positions"][pos]["n_players_ranked"] = int(len(agg))
        if agg.empty:
            lines.append("_No qualifying players (missing data, composite failed, or below snap threshold)._")
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
            lines.append(
                "| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |"
            )
            lines.append("|---:|---:|---|---:|---:|---:|---:|---|")
            for _, r in bucket.iterrows():
                lines.append(
                    f"| {int(r['rank_overall'])} | {int(r['rank_in_tier'])} | {r['player']} | "
                    f"{r['grade']:.2f} | {_fmt_opt_num(r.get('model_grade'))} | "
                    f"{_fmt_opt_num(r.get('stats_grade'))} | {r['snaps']:.0f} | {r['primary_team']} |"
                )
            lines.append("")

    combined = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    return "\n".join(lines), combined, meta


def _years_to_export(args: argparse.Namespace) -> list[int]:
    """Inclusive range [year_min, year_max] if both set; else single analysis_year."""
    if getattr(args, "all_analysis_years", False):
        lo, hi = export_analysis_year_range()
        lo = clamp_analysis_year(lo)
        hi = clamp_analysis_year(hi)
        if lo > hi:
            lo, hi = hi, lo
        return list(range(hi, lo - 1, -1))
    if args.year_min is not None and args.year_max is not None:
        lo = clamp_analysis_year(args.year_min)
        hi = clamp_analysis_year(args.year_max)
        if lo > hi:
            lo, hi = hi, lo
        return list(range(hi, lo - 1, -1))
    return [clamp_analysis_year(args.analysis_year)]


def main() -> None:
    ap = argparse.ArgumentParser(description="Export FA tier rankings by position.")
    ap.add_argument("--analysis-year", type=int, default=2025)
    ap.add_argument(
        "--year-min",
        type=int,
        default=None,
        metavar="Y",
        help="With --year-max, export one report per year (inclusive). Descending order (newest first).",
    )
    ap.add_argument(
        "--year-max",
        type=int,
        default=None,
        metavar="Y",
        help="With --year-min, export one report per year (inclusive).",
    )
    ap.add_argument(
        "--all-analysis-years",
        action="store_true",
        help=f"Export every analysis year from earliest ML season through {LATEST_ANALYSIS_YEAR} (overrides --analysis-year; do not combine with --year-min/--year-max).",
    )
    ap.add_argument("--min-snaps", type=float, default=100.0)
    ap.add_argument("--out-md", type=str, default="")
    ap.add_argument("--out-csv", type=str, default="")
    args = ap.parse_args()

    if args.all_analysis_years:
        if args.year_min is not None or args.year_max is not None:
            ap.error("Do not combine --all-analysis-years with --year-min/--year-max.")
    elif (args.year_min is None) ^ (args.year_max is None):
        ap.error("Use both --year-min and --year-max together, or neither.")

    years = _years_to_export(args)
    multi = len(years) > 1
    if multi and (args.out_md or args.out_csv):
        ap.error("Custom --out-md/--out-csv not supported with --year-min/--year-max; omit them.")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    for y in years:
        stem = f"fa_tier_rankings_analysis_year_{clamp_analysis_year(y)}"
        out_md = args.out_md or os.path.join(REPORTS_DIR, f"{stem}.md")
        out_csv = args.out_csv or os.path.join(REPORTS_DIR, f"{stem}.csv")

        md, df, _meta = build_report(y, args.min_snaps)
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
