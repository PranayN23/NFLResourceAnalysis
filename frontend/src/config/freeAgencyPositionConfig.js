/**
 * Free Agency evaluator: API routes, copy, market legend, and stats table columns per position.
 * Backend ports must match backend/agent/*_main_api.py
 *
 * Market tier bands use the same grade→AAV curve as `backend/agent/market_value_curves.py`
 * (piecewise power law between FA_GRADE_ANCHORS with FA_SEGMENT_POWERS[pos],
 * FA_CAP_EXPONENT[pos] on league cap vs calibration year, grade clamped to [45, 100]).
 * Tier thresholds match `_grade_to_tier`: Elite ≥80, Good ≥74, Starter ≥62, else Rotation/backup.
 */

export const NOTE_STD =
  'Fair AAV uses veteran-market knots (not rookie-scale), piecewise grade segments, and position-specific cap-year growth (e.g. EDGE vs WR). Age curve from position data where applicable; contract PV uses 8%/yr discount.';

/** Keep in sync with `backend/agent/team_context.py` (`_LEAGUE_CAP_BY_YEAR`, `CAP_GROWTH_RATE`). */
export const FA_LEAGUE_CAP_GROWTH_RATE = 0.065;
const FA_LEAGUE_CAP_BY_YEAR = {
  2010: 102.0,
  2011: 120.0,
  2012: 120.6,
  2013: 123.0,
  2014: 133.0,
  2015: 143.28,
  2016: 155.27,
  2017: 167.0,
  2018: 177.2,
  2019: 188.2,
  2020: 198.2,
  2021: 182.5,
  2022: 208.2,
  2023: 224.8,
  2024: 255.4,
  2025: 279.2,
  2026: 301.2,
};

/**
 * NFL salary cap in $M for a league year (published where known; else projected from nearest year).
 */
export function leagueCapMillions(year) {
  const y = Math.floor(Number(year));
  if (!Number.isFinite(y)) return FA_LEAGUE_CAP_BY_YEAR[2024];
  if (Object.prototype.hasOwnProperty.call(FA_LEAGUE_CAP_BY_YEAR, y)) {
    return FA_LEAGUE_CAP_BY_YEAR[y];
  }
  const keys = Object.keys(FA_LEAGUE_CAP_BY_YEAR).map(Number);
  const hi = Math.max(...keys);
  const lo = Math.min(...keys);
  if (y > hi) {
    return Math.round(FA_LEAGUE_CAP_BY_YEAR[hi] * (1 + FA_LEAGUE_CAP_GROWTH_RATE) ** (y - hi) * 100) / 100;
  }
  return Math.round((FA_LEAGUE_CAP_BY_YEAR[lo] / (1 + FA_LEAGUE_CAP_GROWTH_RATE) ** (lo - y)) * 100) / 100;
}

/** Keep legend ranges exactly aligned with backend valuation curve. */
export const FA_LEGEND_AAV_DISPLAY_FACTOR = 1.0;

/**
 * Elite tier is grades ≥80 in the agent; the legend's **upper** $ endpoint is display-only.
 * WR composites essentially never sit at grade 100, so showing m(100) inflates the Elite band.
 * Other positions keep 100 — adjust here only when the same issue shows up.
 */
export const FA_LEGEND_ELITE_DISPLAY_MAX_GRADE_BY_POSITION = {
  WR: 92,
};

function legendEliteUpperGrade(positionKey) {
  const g = FA_LEGEND_ELITE_DISPLAY_MAX_GRADE_BY_POSITION[positionKey];
  return typeof g === 'number' && Number.isFinite(g) ? g : 100;
}

/** Same as all backend `*_agent_graph.py` files */
export const FA_GRADE_ANCHORS = [45, 55, 60, 65, 70, 75, 80, 85, 88, 92, 96, 100];
/** Keep in sync with `backend/agent/market_value_curves.py` (anchors embed calibration; no extra haircut). */
export const FA_MARKET_CALIBRATION_FACTOR = 1.0;

/**
 * `_VALUE_ANCHORS` per position — keep in sync with `backend/agent/market_value_curves.py` VALUE_BY_POSITION.
 */
export const FA_VALUE_ANCHORS = {
  QB: [0.91, 3.64, 20.93, 25.48, 30.94, 38.22, 50.05, 52.78, 54.6, 56.42, 58.24, 60.06],
  HB: [1.2, 2.4, 5.83, 7.77, 9.72, 11.65, 15.53, 18.12, 19.41, 22.02, 24.59, 27.18],
  WR: [1.49, 3.73, 7.46, 13.0, 20.0, 31.0, 39.5, 42.5, 44.0, 45.0, 46.0, 46.5],
  TE: [1.94, 4.15, 9.66, 13.12, 14.5, 15.88, 22.1, 24.17, 24.86, 25.55, 26.25, 26.65],
  T: [1.21, 3.03, 7.25, 13.3, 18.13, 22.96, 26.58, 30.21, 32.62, 33.84, 35.06, 36.25],
  G: [0.74, 1.48, 3.2, 5.42, 8.38, 11.33, 14.29, 17.74, 20.2, 23.16, 26.61, 29.57],
  C: [0.74, 1.48, 3.03, 4.84, 6.04, 7.87, 10.28, 13.3, 15.12, 17.53, 20.54, 23.35],
  ED: [1.32, 3.3, 6.58, 10.53, 17.12, 23.69, 30.26, 35.54, 40.81, 47.38, 52.64, 56.58],
  DI: [1.28, 3.19, 7.64, 14.01, 19.1, 24.19, 29.28, 34.37, 36.28, 38.19, 42.01, 44.57],
  LB: [0.93, 1.86, 3.72, 6.2, 8.68, 11.16, 13.65, 16.76, 21.09, 24.8, 27.29, 29.78],
  CB: [1.24, 2.48, 6.2, 9.93, 13.65, 18.61, 23.58, 28.54, 31.02, 33.5, 36.0, 38.47],
  S: [0.99, 1.86, 3.72, 6.2, 8.68, 11.16, 14.89, 18.0, 22.33, 26.05, 28.53, 31.01],
};

/** Per-segment grade exponents — keep in sync with `SEGMENT_POWER_BY_POSITION` in market_value_curves.py */
export const FA_SEGMENT_POWERS = {
  QB: [0.95, 1.0, 1.0, 1.0, 1.02, 1.05, 1.1, 1.15, 1.28, 1.42, 1.55],
  HB: [1.0, 1.0, 1.0, 1.0, 1.0, 1.02, 1.05, 1.08, 1.12, 1.18, 1.22],
  WR: [0.98, 1.0, 1.0, 1.0, 1.0, 1.0, 1.03, 1.08, 1.22, 1.38, 1.5],
  TE: [1.0, 1.0, 1.0, 1.0, 1.0, 1.03, 1.08, 1.12, 1.22, 1.32, 1.38],
  T: [1.0, 1.0, 1.0, 1.0, 1.02, 1.04, 1.06, 1.1, 1.18, 1.22, 1.26],
  G: [1.0, 1.0, 1.0, 1.0, 1.02, 1.04, 1.06, 1.1, 1.15, 1.2, 1.24],
  C: [1.0, 1.0, 1.0, 1.0, 1.02, 1.04, 1.06, 1.1, 1.15, 1.2, 1.24],
  ED: [0.98, 1.0, 1.0, 1.0, 1.02, 1.05, 1.1, 1.15, 1.3, 1.48, 1.62],
  DI: [1.0, 1.0, 1.0, 1.02, 1.04, 1.06, 1.1, 1.14, 1.22, 1.3, 1.36],
  LB: [1.0, 1.0, 1.0, 1.0, 1.02, 1.04, 1.08, 1.12, 1.2, 1.26, 1.3],
  CB: [0.98, 1.0, 1.0, 1.0, 1.02, 1.05, 1.1, 1.14, 1.26, 1.4, 1.52],
  S: [1.0, 1.0, 1.0, 1.0, 1.02, 1.04, 1.08, 1.12, 1.18, 1.24, 1.28],
};

/** Cap-year scaling exponent — keep in sync with `CAP_EXPONENT_BY_POSITION` in market_value_curves.py */
export const FA_CAP_EXPONENT = {
  QB: 1.04,
  WR: 1.0,
  HB: 0.96,
  TE: 1.02,
  T: 1.05,
  G: 1.04,
  C: 1.04,
  ED: 1.12,
  DI: 1.07,
  LB: 1.03,
  CB: 1.08,
  S: 1.02,
};

export const FA_VALUE_ANCHOR_CALIBRATION_YEAR = 2026;

/**
 * Mirrors Python `cap_scale_for_year`: ratio of the selected year's cap to the
 * calibration year's cap, so dollar values scale proportionally with the cap.
 */
export function capScaleForYear(year) {
  return leagueCapMillions(year) / leagueCapMillions(FA_VALUE_ANCHOR_CALIBRATION_YEAR);
}

function piecewiseAavCalibrationYear(grade, valueAnchors, segmentPowers) {
  const xs = FA_GRADE_ANCHORS;
  const g = Math.max(45, Math.min(100, Number(grade)));
  if (!Number.isFinite(g)) return valueAnchors[0];
  if (g <= xs[0]) return valueAnchors[0];
  if (g >= xs[xs.length - 1]) return valueAnchors[xs.length - 1];
  let i = 0;
  for (let k = 1; k < xs.length; k += 1) {
    if (g <= xs[k]) {
      i = k - 1;
      break;
    }
  }
  const g0 = xs[i];
  const g1 = xs[i + 1];
  const v0 = valueAnchors[i];
  const v1 = valueAnchors[i + 1];
  const p = segmentPowers[i] ?? 1;
  const t = (g - g0) / (g1 - g0);
  const w = t ** p;
  return v0 + (v1 - v0) * w;
}

/**
 * Fair AAV ($M) for ``analysisYear`` — piecewise power-law between grade knots
 * (same as backend `fair_market_aav_millions`).
 */
export function gradeToMarketAav(grade, positionKey, analysisYear = FA_VALUE_ANCHOR_CALIBRATION_YEAR) {
  const pk = positionKey && FA_VALUE_ANCHORS[positionKey] ? positionKey : 'WR';
  const anchors = FA_VALUE_ANCHORS[pk];
  const powers = FA_SEGMENT_POWERS[pk] || FA_SEGMENT_POWERS.WR;
  const base =
    Math.round(
      piecewiseAavCalibrationYear(grade, anchors, powers) * FA_MARKET_CALIBRATION_FACTOR * 100,
    ) / 100;
  const exp = FA_CAP_EXPONENT[pk] ?? 1;
  const capY = leagueCapMillions(analysisYear);
  const capR = leagueCapMillions(FA_VALUE_ANCHOR_CALIBRATION_YEAR);
  const factor = (capY / capR) ** exp;
  return Math.round(base * factor * 100) / 100;
}

function fmtM(v) {
  const r = Math.round(v * 100) / 100;
  if (Number.isInteger(r)) return String(r);
  const s = r.toFixed(2);
  if (s.endsWith('0')) return r.toFixed(1).replace(/\.0$/, '');
  return s;
}

/**
 * Fair AAV bands by agent tier (same thresholds as `_grade_to_tier` in Python).
 * - Elite: 80–100 → m(80)–m(legend max); WR uses grade 92 for the upper $ (not 100).
 * - Good: 74–80 → m(74)–m(80)
 * - Starter: 62–74 → m(62)–m(74)
 * - Rotation/backup: &lt;62 → below fair AAV at grade 62
 */
export function buildMarketTierLegend(positionKey, extraNote = '') {
  const raw = (g) => gradeToMarketAav(g, positionKey);
  const m = (g) =>
    Math.round(raw(g) * FA_LEGEND_AAV_DISPLAY_FACTOR * 100) / 100;
  const m62 = m(62);
  const m74 = m(74);
  const m80 = m(80);
  const eliteTopG = legendEliteUpperGrade(positionKey);
  const mEliteTop = m(eliteTopG);

  const tierNote =
    'Tier cutoffs match the backend agent (<code>_grade_to_tier</code>: Elite ≥80, Good ≥74, Starter ≥62; Rotation/backup &lt;62). ' +
    'Dollar amounts use the same piecewise grade→AAV curve and cap-year exponent as the backend (<code>market_value_curves.py</code>). ' +
    (eliteTopG < 100
      ? `Elite fair AAV in this legend runs from grade 80 to ${eliteTopG} (not 100 — realistic composite ceiling for ${positionKey}). `
      : 'Reference grades for the bands are 62 / 74 / 80 / 100. ');

  return {
    title: `${FA_VALUE_ANCHOR_CALIBRATION_YEAR} Fair AAV by Tier (${positionKey})`,
    tiers: [
      { cls: 'elite', label: 'Elite', range: `$${fmtM(m80)}–${fmtM(mEliteTop)}M` },
      { cls: 'good', label: 'Good', range: `$${fmtM(m74)}–${fmtM(m80)}M` },
      { cls: 'starter', label: 'Starter', range: `$${fmtM(m62)}–${fmtM(m74)}M` },
      { cls: 'rotation', label: 'Rotation / backup', range: `<$${fmtM(m62)}M` },
    ],
    note: `${tierNote}${extraNote ? `<br/>${extraNote}` : ''}<br/>${NOTE_STD}`,
  };
}

/** Stat row: key (career), label, fmt id, optional projKey for projection column, noDelta skips heat */
const FMT = {
  pct: 'pct',
  pct0: 'pct0',
  btt: 'btt',
  rate4: 'rate4',
  skRate: 'sacks_rate',
};

export const FA_POSITION_ORDER = [
  'QB',
  'HB',
  'WR',
  'TE',
  'T',
  'G',
  'C',
  'ED',
  'DI',
  'LB',
  'CB',
  'S',
];

/**
 * Fair AAV ($M) tier bands for every position — same interpolation as `buildMarketTierLegend`
 * (grades 62 / 74 / 80 / 100; Elite / Good / Starter / Rotation).
 */
export function fairAavTierBandsForAllPositions(analysisYear = FA_VALUE_ANCHOR_CALIBRATION_YEAR) {
  const fmtM = (v) => {
    const r = Math.round(v * 100) / 100;
    if (Number.isInteger(r)) return String(r);
    const s = r.toFixed(2);
    if (s.endsWith('0')) return r.toFixed(1).replace(/\.0$/, '');
    return s;
  };
  return FA_POSITION_ORDER.map((pk) => {
    const va = FA_VALUE_ANCHORS[pk];
    if (!va) {
      return { pos: pk, elite: '—', good: '—', starter: '—', rotation: '—' };
    }
    const m = (g) =>
      Math.round(gradeToMarketAav(g, pk, analysisYear) * FA_LEGEND_AAV_DISPLAY_FACTOR * 100) / 100;
    const eliteTopG = legendEliteUpperGrade(pk);
    return {
      pos: pk,
      elite: `$${fmtM(m(80))}–${fmtM(m(eliteTopG))}M`,
      good: `$${fmtM(m(74))}–${fmtM(m(80))}M`,
      starter: `$${fmtM(m(62))}–${fmtM(m(74))}M`,
      rotation: `<$${fmtM(m(62))}M`,
    };
  });
}

export const POSITION_FREE_AGENCY = {
  QB: {
    fullName: 'Quarterback',
    port: 8004,
    playersPath: '/qb-players',
    panelTitle: 'Quarterback Scout',
    chatTitle: 'Quarterback',
    positionLabel: 'Quarterback',
    welcome:
      'Welcome to the Quarterback Free Agency Evaluator. Select a player, set contract AAV and length, then Analyze. Toggle Team Simulation Mode to factor roster strength, positional need, and cap space.',
    legend: buildMarketTierLegend(
      'QB',
      'Composite: 40% model pass grade + 60% stats grade. Stats: 35% passer rating, 30% YPA, 25% EPA/db, 10% comp%. Last 1–3 seasons use recency × dropback weights (higher-snap years count more).'
    ),
    statRows: [
      { key: 'yards', label: 'Pass Yards' },
      { key: 'touchdowns', label: 'Pass TDs' },
      { key: 'interceptions', label: 'INTs' },
      { key: 'ypa', label: 'YPA' },
      { key: 'qb_rating', label: 'Passer Rating' },
      { key: 'completion_pct', label: 'Comp %', fmt: FMT.pct },
      { key: 'run_grade', label: 'Run Grade' },
      { key: 'pass_grade', label: 'Pass Grade' },
      { key: 'overall_grade', label: 'Overall Grade', projKey: 'projected_grade', noDelta: true },
    ],
    statsNote:
      'Career columns are per season. With Team Simulation on, projected workload is starter vs fringe vs backup from roster + need; without a team, volume assumes a full-time starter role. INTs and TDs use stabilized per–dropback rates × that workload (backup seasons do not zero out turnover modeling). Yards/TDs/rating scale with the grade path; INT risk nudges when projected grade trails the composite.',
  },
  HB: {
    fullName: 'Running Back',
    port: 8005,
    playersPath: '/hb-players',
    panelTitle: 'Running Back Scout',
    chatTitle: 'Running Back',
    positionLabel: 'Running Back',
    welcome:
      'Welcome to the Running Back Free Agency Evaluator. Select a player, set contract AAV and length, then Analyze. Toggle Team Simulation Mode for team-specific need and cap context.',
    legend: buildMarketTierLegend('HB'),
    statRows: [
      { key: 'yards', label: 'Rush Yards' },
      { key: 'attempts', label: 'Attempts' },
      { key: 'ypc', label: 'YPC' },
      { key: 'touchdowns', label: 'Rush TDs' },
      { key: 'receptions', label: 'Rec' },
      { key: 'rec_yards', label: 'Rec Yards' },
      { key: 'elusive_rating', label: 'Elusive' },
      { key: 'run_grade', label: 'Run Grade' },
      { key: 'overall_grade', label: 'Overall Grade', projKey: 'projected_grade', noDelta: true },
    ],
    statsNote: 'Career columns are per season. Projections scale receiving and rushing volume by composite grade path.',
  },
  WR: {
    fullName: 'Wide Receiver',
    port: 8006,
    playersPath: '/wr-players',
    panelTitle: 'Wide Receiver Scout',
    chatTitle: 'Wide Receiver',
    positionLabel: 'Wide Receiver',
    welcome:
      'Welcome to the Wide Receiver Free Agency Evaluator. Select a player, set contract AAV and length, then Analyze. Toggle Team Simulation Mode for roster and cap context.',
    legend: buildMarketTierLegend('WR'),
    statRows: [
      { key: 'receptions', label: 'Rec' },
      { key: 'targets', label: 'Targets' },
      { key: 'yards', label: 'Yards' },
      { key: 'yards_per_rec', label: 'YPR' },
      { key: 'yac_per_rec', label: 'YAC/Rec' },
      { key: 'yprr', label: 'YPRR' },
      { key: 'drop_rate', label: 'Drop Rate', fmt: FMT.rate4 },
      { key: 'route_grade', label: 'Route Grade' },
      { key: 'overall_grade', label: 'Overall Grade', projKey: 'projected_grade', noDelta: true },
    ],
    statsNote: 'Projections scale volume and efficiency vs last-season baseline using the composite grade trajectory.',
  },
  TE: {
    fullName: 'Tight End',
    port: 8007,
    playersPath: '/te-players',
    panelTitle: 'Tight End Scout',
    chatTitle: 'Tight End',
    positionLabel: 'Tight End',
    welcome:
      'Welcome to the Tight End Free Agency Evaluator. Select a player, set contract AAV and length, then Analyze. Toggle Team Simulation Mode for roster and cap context.',
    legend: buildMarketTierLegend('TE'),
    statRows: [
      { key: 'receptions', label: 'Rec' },
      { key: 'yards', label: 'Yards' },
      { key: 'yards_per_rec', label: 'YPR' },
      { key: 'yprr', label: 'YPRR' },
      { key: 'drop_rate', label: 'Drop Rate', fmt: FMT.rate4 },
      { key: 'pass_block_grade', label: 'Pass Block Gr.' },
      { key: 'overall_grade', label: 'Overall Grade', projKey: 'projected_grade', noDelta: true },
    ],
    statsNote: 'Receiving and pass-block grades drive value; projections follow the composite grade curve.',
  },
  T: {
    fullName: 'Tackle',
    port: 8008,
    playersPath: '/t-players',
    panelTitle: 'Offensive Tackle Scout',
    chatTitle: 'Offensive Tackle',
    positionLabel: 'Tackle',
    welcome:
      'Welcome to the Tackle Free Agency Evaluator. Pass protection is weighted heavily. Use Team Simulation Mode for OL depth and cap.',
    legend: buildMarketTierLegend('T', 'Tackle market curve (pass-block premium) from <code>ol_agent_graph</code>.'),
    statRows: [
      { key: 'sacks_allowed', label: 'Sacks Allowed' },
      { key: 'hits_allowed', label: 'Hits Allowed' },
      { key: 'hurries_allowed', label: 'Hurries Allowed' },
      { key: 'sacks_rate', label: 'Sack Rate', fmt: FMT.skRate },
      { key: 'pbe', label: 'PBE' },
      { key: 'pass_block_grade', label: 'Pass Block Gr.' },
      { key: 'run_block_grade', label: 'Run Block Gr.' },
      { key: 'overall_grade', label: 'Overall Grade', projKey: 'projected_grade', noDelta: true },
    ],
    statsNote: 'Lower pressures allowed are better; projected allowed stats scale inversely with grade improvement.',
  },
  G: {
    fullName: 'Guard',
    port: 8009,
    playersPath: '/g-players',
    panelTitle: 'Guard Scout',
    chatTitle: 'Guard',
    positionLabel: 'Guard',
    welcome:
      'Welcome to the Guard Free Agency Evaluator. Select a player, set contract AAV and length, then Analyze. Toggle Team Simulation Mode for OL room and cap.',
    legend: buildMarketTierLegend('G'),
    statRows: [
      { key: 'sacks_allowed', label: 'Sacks Allowed' },
      { key: 'hits_allowed', label: 'Hits Allowed' },
      { key: 'hurries_allowed', label: 'Hurries Allowed' },
      { key: 'sacks_rate', label: 'Sack Rate', fmt: FMT.skRate },
      { key: 'pbe', label: 'PBE' },
      { key: 'pass_block_grade', label: 'Pass Block Gr.' },
      { key: 'run_block_grade', label: 'Run Block Gr.' },
      { key: 'overall_grade', label: 'Overall Grade', projKey: 'projected_grade', noDelta: true },
    ],
    statsNote: 'Same projection logic as tackle; market curve uses guard anchors.',
  },
  C: {
    fullName: 'Center',
    port: 8010,
    playersPath: '/c-players',
    panelTitle: 'Center Scout',
    chatTitle: 'Center',
    positionLabel: 'Center',
    welcome:
      'Welcome to the Center Free Agency Evaluator. Select a player, set contract AAV and length, then Analyze. Toggle Team Simulation Mode for OL room and cap.',
    legend: buildMarketTierLegend('C'),
    statRows: [
      { key: 'sacks_allowed', label: 'Sacks Allowed' },
      { key: 'hits_allowed', label: 'Hits Allowed' },
      { key: 'hurries_allowed', label: 'Hurries Allowed' },
      { key: 'sacks_rate', label: 'Sack Rate', fmt: FMT.skRate },
      { key: 'pbe', label: 'PBE' },
      { key: 'pass_block_grade', label: 'Pass Block Gr.' },
      { key: 'run_block_grade', label: 'Run Block Gr.' },
      { key: 'overall_grade', label: 'Overall Grade', projKey: 'projected_grade', noDelta: true },
    ],
    statsNote: 'Same projection logic as guard; market curve uses center anchors.',
  },
  ED: {
    fullName: 'Edge Defender',
    port: 8002,
    playersPath: '/ed-players',
    panelTitle: 'Edge Defender Scout',
    chatTitle: 'Edge Defender',
    positionLabel: 'Edge Defender',
    welcome:
      'Welcome to the Edge Defender Free Agency Evaluator. Select a player, set the contract AAV and length, then click Analyze for a recommendation. Toggle Team Simulation Mode to evaluate signings from a specific team\'s perspective — factoring in roster strength, positional need, and cap space.',
    legend: buildMarketTierLegend(
      'ED',
      'Composite = 40% model PFF grade + 60% stats (pressure %, sack rate, stops).'
    ),
    statRows: [
      { key: 'sacks', label: 'Sacks' },
      { key: 'pressures', label: 'Pressures' },
      { key: 'pressure_pct', label: 'Pressure %', fmt: FMT.pct0 },
      { key: 'stops', label: 'Stops' },
      { key: 'pass_rush_grade', label: 'PR Grade' },
      { key: 'run_def_grade', label: 'RD Grade' },
      { key: 'overall_grade', label: 'Overall Grade', projKey: 'projected_grade', noDelta: true },
    ],
    statsNote:
      'Career columns show actual per-season stats. Projected years assume 17 healthy games, scaled by the composite grade trajectory. Composite = 40% model PFF grade + 60% stats-based grade (pressure %, sack rate, stops).',
  },
  DI: {
    fullName: 'Defensive Interior',
    port: 8003,
    playersPath: '/di-players',
    panelTitle: 'Defensive Interior Scout',
    chatTitle: 'Defensive Interior',
    positionLabel: 'Defensive Interior',
    welcome:
      'Welcome to the Defensive Interior Free Agency Evaluator. Run-stopping is the primary value driver. Select a player, set the contract AAV and length, then click Analyze for a recommendation. Toggle Team Simulation Mode to evaluate signings from a specific team\'s perspective — factoring in roster strength, positional need, and cap space.',
    legend: buildMarketTierLegend(
      'DI',
      'Stop rate weighted heavily in stats grade.'
    ),
    statRows: [
      { key: 'stops', label: 'Stops' },
      { key: 'tfl', label: 'TFL' },
      { key: 'pressures', label: 'Pressures' },
      { key: 'sacks', label: 'Sacks' },
      { key: 'stop_rate', label: 'Stop Rate %', fmt: FMT.pct },
      { key: 'pass_rush_grade', label: 'PR Grade' },
      { key: 'run_def_grade', label: 'RD Grade' },
      { key: 'overall_grade', label: 'Overall Grade', projKey: 'projected_grade', noDelta: true },
    ],
    statsNote:
      'Career columns show actual per-season stats. Projected years assume 17 healthy games. Composite = 40% model PFF grade + 60% stats-based grade (stop rate 40%, TFL 20%, pressure 20%, sacks 20%).',
  },
  LB: {
    fullName: 'Linebacker',
    port: 8011,
    playersPath: '/lb-players',
    panelTitle: 'Linebacker Scout',
    chatTitle: 'Linebacker',
    positionLabel: 'Linebacker',
    welcome:
      'Welcome to the Linebacker Free Agency Evaluator. Coverage, run D, and tackle production drive the model. Use Team Simulation Mode for depth chart and cap.',
    legend: buildMarketTierLegend('LB'),
    statRows: [
      { key: 'tackles', label: 'Tackles' },
      { key: 'assists', label: 'Assists' },
      { key: 'tfl', label: 'TFL' },
      { key: 'stops', label: 'Stops' },
      { key: 'sacks', label: 'Sacks' },
      { key: 'interceptions', label: 'INTs' },
      { key: 'pass_breakups', label: 'PBUs' },
      { key: 'coverage_grade', label: 'Coverage Gr.' },
      { key: 'run_def_grade', label: 'Run Def Gr.' },
      { key: 'tackle_grade', label: 'Tackle Gr.' },
      { key: 'overall_grade', label: 'Overall Grade', projKey: 'projected_grade', noDelta: true },
    ],
    statsNote: 'Projection uses 17-game scaled box stats; tackle grade is shown for career seasons only where available.',
  },
  CB: {
    fullName: 'Cornerback',
    port: 8012,
    playersPath: '/cb-players',
    panelTitle: 'Cornerback Scout',
    chatTitle: 'Cornerback',
    positionLabel: 'Cornerback',
    welcome:
      'Welcome to the Cornerback Free Agency Evaluator. Coverage and ball production drive value. Use Team Simulation Mode for CB room strength and cap.',
    legend: buildMarketTierLegend('CB'),
    statRows: [
      { key: 'interceptions', label: 'INTs' },
      { key: 'pass_breakups', label: 'PBUs' },
      { key: 'tackles', label: 'Tackles' },
      { key: 'tackles_for_loss', label: 'TFL', projKey: 'tfl' },
      { key: 'qb_rating_against', label: 'QBR Against' },
      { key: 'coverage_grade', label: 'Coverage Gr.' },
      { key: 'tackle_grade', label: 'Tackle Gr.' },
      { key: 'overall_grade', label: 'Overall Grade', projKey: 'projected_grade', noDelta: true },
    ],
    statsNote: 'Career TFL vs projected TFL: same row for continuity. Overall projection follows composite grade.',
  },
  S: {
    fullName: 'Safety',
    port: 8013,
    playersPath: '/s-players',
    panelTitle: 'Safety Scout',
    chatTitle: 'Safety',
    positionLabel: 'Safety',
    welcome:
      'Welcome to the Safety Free Agency Evaluator. Coverage, tackling, and ball skills drive the model. Use Team Simulation Mode for depth and cap.',
    legend: buildMarketTierLegend('S'),
    statRows: [
      { key: 'interceptions', label: 'INTs' },
      { key: 'pass_breakups', label: 'PBUs' },
      { key: 'tackles', label: 'Tackles' },
      { key: 'tackles_for_loss', label: 'TFL', projKey: 'tfl' },
      { key: 'coverage_grade', label: 'Coverage Gr.' },
      { key: 'tackle_grade', label: 'Tackle Gr.' },
      { key: 'defense_grade', label: 'Overall Grade', projKey: 'projected_grade', noDelta: true },
    ],
    statsNote: 'Safeties use combined defense grade in career columns; projected years show trajectory via projected_grade.',
  },
};

export const FA_PICKER_POSITIONS = FA_POSITION_ORDER.map((label) => ({
  label,
  name: POSITION_FREE_AGENCY[label].fullName,
}));
