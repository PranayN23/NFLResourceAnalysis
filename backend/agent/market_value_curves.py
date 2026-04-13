"""
Grade → fair veteran-market AAV ($M), piecewise-nonlinear between anchors.

- Between each pair of ``GRADE_ANCHORS``, dollars follow ``v0 + (v1-v0) * t**p``
  where ``t`` is normalized grade within the segment and ``p`` is a per-segment
  exponent (≠1 ⇒ not linear). High-end segments use ``p>1`` so pay rises steeply
  at the top of the band (elite premium / grade compression).
- League-year adjustment is ``(cap_y / cap_ref) ** exponent[pos]`` so positions
  whose markets outpaced the cap (e.g. EDGE vs WR) scale differently across years.
- Knots are calibrated in ``VALUE_ANCHOR_CALIBRATION_YEAR`` dollars; rookie-deal
  compression is handled offline via ``fit_market_curves_from_cap_data.py``.

Keep ``FA_GRADE_ANCHORS``, ``FA_VALUE_ANCHORS``, ``FA_SEGMENT_POWERS``, and
``FA_CAP_EXPONENT`` in ``frontend/src/config/freeAgencyPositionConfig.js`` aligned.
"""

from __future__ import annotations

import bisect
from typing import Sequence

import numpy as np

from backend.agent.team_context import league_cap_millions, VALUE_ANCHOR_CALIBRATION_YEAR

# Keep in sync with frontend `FA_GRADE_ANCHORS`.
GRADE_ANCHORS: list[int] = [45, 55, 60, 65, 70, 75, 80, 85, 88, 92, 96, 100]

MARKET_CALIBRATION_FACTOR = 1.0

# Fair AAV ($M) at VALUE_ANCHOR_CALIBRATION_YEAR — veteran market, not rookie scale.
VALUE_BY_POSITION: dict[str, list[float]] = {
    "QB": [0.91, 3.64, 20.93, 25.48, 30.94, 38.22, 50.05, 52.78, 54.60, 56.42, 58.24, 60.06],
    "HB": [1.20, 2.40, 5.83, 7.77, 9.72, 11.65, 15.53, 18.12, 19.41, 22.02, 24.59, 27.18],
    # WR: fair AAV is (piecewise grade curve) × cap-year × snap reliability, then a multi-year
    # front-weighted average. Keep 65–75 knots strong for true #1 money; trim 75+ slightly so
    # elite legend / top-of-curve stays below QB and slot/WR3 profiles do not inherit WR1 caps.
    "WR": [1.49, 3.73, 7.46, 13.0, 20.0, 31.0, 42.5, 48.2, 51.8, 54.5, 56.5, 58.2],
    "TE": [1.94, 4.15, 9.66, 13.12, 14.50, 15.88, 22.10, 24.17, 24.86, 25.55, 26.25, 26.65],
    "T": [1.21, 3.03, 7.25, 13.30, 18.13, 22.96, 26.58, 30.21, 32.62, 33.84, 35.06, 36.25],
    "G": [0.74, 1.48, 3.20, 5.42, 8.38, 11.33, 14.29, 17.74, 20.20, 23.16, 26.61, 29.57],
    "C": [0.74, 1.48, 3.03, 4.84, 6.04, 7.87, 10.28, 13.30, 15.12, 17.53, 20.54, 23.35],
    "ED": [1.32, 3.30, 6.58, 10.53, 17.12, 23.69, 30.26, 35.54, 40.81, 47.38, 52.64, 56.58],
    "DI": [1.28, 3.19, 7.64, 14.01, 19.10, 24.19, 29.28, 34.37, 36.28, 38.19, 42.01, 44.57],
    "LB": [0.93, 1.86, 3.72, 6.20, 8.68, 11.16, 13.65, 16.76, 21.09, 24.80, 27.29, 29.78],
    "CB": [1.24, 2.48, 6.20, 9.93, 13.65, 18.61, 23.58, 28.54, 31.02, 33.50, 36.00, 38.47],
    "S": [0.99, 1.86, 3.72, 6.20, 8.68, 11.16, 14.89, 18.00, 22.33, 26.05, 28.53, 31.01],
}

# One exponent per segment [g_i, g_{i+1}], len = len(GRADE_ANCHORS) - 1.
# p<1: concave (most $ in lower part); p>1: convex (elite premium at top of segment).
def _p11(*xs: float) -> list[float]:
    return list(xs)


SEGMENT_POWER_BY_POSITION: dict[str, list[float]] = {
    "QB": _p11(0.95, 1.0, 1.0, 1.0, 1.02, 1.05, 1.1, 1.15, 1.28, 1.42, 1.55),
    "HB": _p11(1.0, 1.0, 1.0, 1.0, 1.0, 1.02, 1.05, 1.08, 1.12, 1.18, 1.22),
    "WR": _p11(0.98, 1.0, 1.0, 1.0, 1.0, 1.0, 1.03, 1.08, 1.22, 1.38, 1.50),
    "TE": _p11(1.0, 1.0, 1.0, 1.0, 1.0, 1.03, 1.08, 1.12, 1.22, 1.32, 1.38),
    "T": _p11(1.0, 1.0, 1.0, 1.0, 1.02, 1.04, 1.06, 1.1, 1.18, 1.22, 1.26),
    "G": _p11(1.0, 1.0, 1.0, 1.0, 1.02, 1.04, 1.06, 1.1, 1.15, 1.2, 1.24),
    "C": _p11(1.0, 1.0, 1.0, 1.0, 1.02, 1.04, 1.06, 1.1, 1.15, 1.2, 1.24),
    "ED": _p11(0.98, 1.0, 1.0, 1.0, 1.02, 1.05, 1.1, 1.15, 1.3, 1.48, 1.62),
    "DI": _p11(1.0, 1.0, 1.0, 1.02, 1.04, 1.06, 1.1, 1.14, 1.22, 1.3, 1.36),
    "LB": _p11(1.0, 1.0, 1.0, 1.0, 1.02, 1.04, 1.08, 1.12, 1.2, 1.26, 1.3),
    "CB": _p11(0.98, 1.0, 1.0, 1.0, 1.02, 1.05, 1.1, 1.14, 1.26, 1.4, 1.52),
    "S": _p11(1.0, 1.0, 1.0, 1.0, 1.02, 1.04, 1.08, 1.12, 1.18, 1.24, 1.28),
}

# Applied as (cap_y / cap_ref) ** exponent — >1 ⇒ market grew faster than cap vs ref.
CAP_EXPONENT_BY_POSITION: dict[str, float] = {
    "QB": 1.04,
    "WR": 1.0,
    "HB": 0.96,
    "TE": 1.02,
    "T": 1.05,
    "G": 1.04,
    "C": 1.04,
    "ED": 1.12,
    "DI": 1.07,
    "LB": 1.03,
    "CB": 1.08,
    "S": 1.02,
}

_GA: Sequence[float] = tuple(float(x) for x in GRADE_ANCHORS)


def _piecewise_fair_aav_calibration_year(grade: float, position: str) -> float:
    pos = position if position in VALUE_BY_POSITION else "WR"
    ys = VALUE_BY_POSITION[pos]
    powers = SEGMENT_POWER_BY_POSITION.get(pos, SEGMENT_POWER_BY_POSITION["WR"])
    g = max(45.0, min(100.0, float(grade)))
    if g <= _GA[0]:
        return float(ys[0])
    if g >= _GA[-1]:
        return float(ys[-1])
    i = bisect.bisect_right(_GA, g) - 1
    i = max(0, min(i, len(_GA) - 2))
    g0, g1 = _GA[i], _GA[i + 1]
    v0, v1 = ys[i], ys[i + 1]
    p = float(powers[i]) if i < len(powers) else 1.0
    if p <= 0:
        p = 1.0
    t = (g - g0) / (g1 - g0) if g1 > g0 else 0.0
    t = max(0.0, min(1.0, t))
    w = t**p
    return float(v0 + (v1 - v0) * w)


def fair_market_aav_millions(grade: float, position: str, analysis_year: int) -> float:
    """Piecewise fair AAV in ``analysis_year`` nominal $M (veteran market shape)."""
    v = _piecewise_fair_aav_calibration_year(grade, position)
    y = int(analysis_year)
    cap_y = league_cap_millions(y)
    cap_ref = league_cap_millions(int(VALUE_ANCHOR_CALIBRATION_YEAR))
    if cap_ref <= 0:
        cap_ref = 1.0
    pos = position if position in CAP_EXPONENT_BY_POSITION else "WR"
    exp = float(CAP_EXPONENT_BY_POSITION[pos])
    out = v * ((cap_y / cap_ref) ** exp)
    return round(float(out), 2)


def grade_to_market_value(grade: float, position: str = "WR") -> float:
    """
    Fair AAV in calibration-year dollars only (piecewise), for legacy call sites
    that apply their own cap scaling — prefer ``fair_market_aav_millions``.
    """
    return round(
        _piecewise_fair_aav_calibration_year(grade, position) * MARKET_CALIBRATION_FACTOR,
        2,
    )


def sample_curve_for_export(
    position: str, grades: Sequence[float] | None = None
) -> list[tuple[float, float, float]]:
    """Debug: (grade, aav_2026_knots, aav_2026_linear) for comparing to linear interp."""
    if grades is None:
        grades = list(range(45, 101))
    ys = VALUE_BY_POSITION.get(position, VALUE_BY_POSITION["WR"])
    out = []
    for g in grades:
        lin = float(np.interp(g, list(_GA), ys))
        pw = _piecewise_fair_aav_calibration_year(float(g), position)
        out.append((float(g), round(pw, 2), round(lin, 2)))
    return out
