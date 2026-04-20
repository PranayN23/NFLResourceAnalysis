"""
Derive position-specific _AGE_DELTAS from historical season-to-season grade changes,
same idea as EDGE/DI: for consecutive seasons (Year diff 1, age diff 1), median of
(grade_curr - grade_prev) grouped by age at the *start* of the transition (matches
agent convention: _annual_grade_delta(age - 1) when entering age).

Run from repo root:
  python3 backend/agent/compute_position_age_curves.py

Uses stdlib only (csv + statistics).
"""
from __future__ import annotations

import csv
import os
import statistics
import sys
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_rows(path: str) -> List[dict]:
    full = os.path.join(_BASE, path)
    if not os.path.isfile(full):
        raise FileNotFoundError(full)
    with open(full, newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def _to_float(x) -> Optional[float]:
    try:
        v = float(x)
        if v != v:  # NaN
            return None
        return v
    except (TypeError, ValueError):
        return None


def _to_int(x) -> Optional[int]:
    try:
        return int(round(float(x)))
    except (TypeError, ValueError):
        return None


def median_deltas_by_age(
    rows: List[dict],
    grade_col: str,
    min_n: int = 12,
    age_min: int = 20,
    age_max: int = 38,
) -> Dict[int, float]:
    """Return age_start -> median delta (rounded 1 decimal)."""
    by_player: Dict[str, List[Tuple[int, int, float]]] = defaultdict(list)
    for r in rows:
        pid = (r.get("player") or "").strip()
        if not pid:
            continue
        yr = _to_int(r.get("Year"))
        ag = _to_int(r.get("age"))
        g = _to_float(r.get(grade_col))
        if yr is None or ag is None or g is None:
            continue
        by_player[pid].append((yr, ag, g))

    deltas_by_age: Dict[int, List[float]] = defaultdict(list)
    for pid, seq in by_player.items():
        seq.sort(key=lambda t: t[0])
        for i in range(1, len(seq)):
            y0, a0, g0 = seq[i - 1]
            y1, a1, g1 = seq[i]
            if y1 - y0 != 1:
                continue
            if a1 != a0 + 1:
                continue
            if not (age_min <= a0 <= age_max):
                continue
            deltas_by_age[a0].append(g1 - g0)

    out: Dict[int, float] = {}
    for a in range(age_min, age_max + 1):
        xs = deltas_by_age.get(a, [])
        if len(xs) >= min_n:
            out[a] = round(float(statistics.median(xs)), 1)
    return out


def fill_age_table(
    raw: Dict[int, float],
    age_lo: int,
    age_hi: int,
    default: float,
    cap: Tuple[float, float] = (-12.0, 12.0),
) -> Dict[int, float]:
    """Fill gaps with linear interpolation; below min(raw) uses youngest raw; above max(raw) uses default."""
    ages = list(range(age_lo, age_hi + 1))
    known_a = sorted(raw.keys())
    filled: Dict[int, float] = {}
    min_raw = min(known_a) if known_a else age_lo
    max_raw = max(known_a) if known_a else age_hi
    for a in ages:
        if a in raw:
            v = raw[a]
        elif a < min_raw:
            v = raw[min_raw]
        elif a > max_raw:
            v = default
        else:
            below = [x for x in known_a if x < a]
            above = [x for x in known_a if x > a]
            if below and above:
                b, c = max(below), min(above)
                t = (a - b) / (c - b) if c != b else 0.0
                v = raw[b] + t * (raw[c] - raw[b])
            elif below:
                v = raw[max(below)]
            elif above:
                v = raw[min(above)]
            else:
                v = default
        v = max(cap[0], min(cap[1], round(float(v), 1)))
        filled[a] = v
    return filled


def py_dict_literal(name: str, d: Dict[int, float], indent: str = "    ") -> str:
    lines = [f"{name} = {{"]
    for k in sorted(d.keys()):
        lines.append(f"{indent}{k}: {d[k]:+.1f},")
    lines.append(f"{indent}}}")
    return "\n".join(lines)


def run_all() -> None:
    specs: List[Tuple[str, str, str, str, Callable[[Dict[int, float]], Dict[int, float]]]] = [
        ("QB", "ML/QB.csv", "grades_offense", "QB", lambda r: fill_age_table(r, 20, 38, -2.8)),
        ("HB", "ML/HB.csv", "grades_offense", "HB", lambda r: fill_age_table(r, 20, 33, -4.0)),
        ("WR", "ML/WR.csv", "grades_offense", "WR", lambda r: fill_age_table(r, 20, 35, -3.5)),
        ("TE", "ML/TightEnds/TE.csv", "grades_offense", "TE", lambda r: fill_age_table(r, 20, 34, -4.0)),
        ("LB", "ML/LB.csv", "grades_defense", "LB", lambda r: fill_age_table(r, 20, 34, -4.0)),
        ("CB", "ML/CB.csv", "grades_defense", "CB", lambda r: fill_age_table(r, 20, 34, -5.0)),
        ("S", "ML/S.csv", "grades_defense", "S", lambda r: fill_age_table(r, 20, 35, -4.0)),
    ]

    ol_specs = [
        ("T", "ML/T.csv"),
        ("G", "ML/G.csv"),
        ("C", "ML/C.csv"),
    ]

    print("# --- Offense / defense skill (from median season transitions) ---\n")
    for _label, rel, gcol, short, filler in specs:
        rows = _load_rows(rel)
        raw = median_deltas_by_age(rows, gcol, min_n=10)
        filled = filler(raw)
        print(f"# {short} n_keys_raw={len(raw)} path={rel} col={gcol}")
        print(py_dict_literal(f"_AGE_DELTAS_{short}", filled))
        print()

    print("# --- OL (T / G / C separate CSVs) ---\n")
    for pos, rel in ol_specs:
        rows = _load_rows(rel)
        raw = median_deltas_by_age(rows, "grades_offense", min_n=8, age_max=36)
        filled = fill_age_table(raw, 20, 36, -3.5)
        print(f"# OL {pos} n_keys_raw={len(raw)} path={rel}")
        print(py_dict_literal(f"_AGE_DELTAS_{pos}", filled))
        print()


if __name__ == "__main__":
    try:
        run_all()
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)
