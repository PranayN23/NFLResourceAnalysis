/**
 * Free Agency evaluator: API routes, copy, market legend, and stats table columns per position.
 * Backend ports must match backend/agent/*_main_api.py
 *
 * Market tier bands are NOT hand-tuned: they use the same grade→AAV curve as each
 * `*_agent_graph.py` (`grade_to_market_value`: np.interp on GRADE_ANCHORS / VALUE_ANCHORS,
 * grade clamped to [45, 100], result rounded to 2 decimals). Tier thresholds match
 * `_grade_to_tier`: Elite ≥80, Good ≥74, Starter ≥62, else Rotation/backup.
 */

const NOTE_STD =
  'Calibrated to 2026 OTC-style contracts. Age curve from position season data where applicable. Future years discounted at 8%/yr.';

/** Keep legend ranges exactly aligned with backend valuation curve. */
export const FA_LEGEND_AAV_DISPLAY_FACTOR = 1.0;

/** Same as all backend `*_agent_graph.py` files */
export const FA_GRADE_ANCHORS = [45, 55, 60, 65, 70, 75, 80, 85, 88, 92, 96, 100];
export const FA_MARKET_CALIBRATION_FACTOR = 0.88;

/**
 * `_VALUE_ANCHORS` per position — keep in sync with:
 * agent_graph (QB), rb_agent_graph (HB), wr, te, ol_agent_graph (T/G/C), ed, di, lb, cb, s
 */
export const FA_VALUE_ANCHORS = {
  QB: [1.0, 4.0, 10.0, 20.0, 30.0, 40.0, 48.0, 54.0, 58.0, 62.0, 68.0, 75.0],
  HB: [0.75, 1.5, 3.0, 5.5, 8.5, 11.5, 14.5, 17.5, 20.0, 23.0, 26.0, 30.0],
  WR: [0.75, 2.0, 4.5, 8.5, 12.5, 16.5, 21.5, 27.0, 31.0, 36.0, 42.0, 50.0],
  TE: [0.75, 1.5, 3.5, 6.5, 9.5, 12.5, 16.0, 20.0, 23.0, 27.0, 31.0, 35.0],
  T: [0.75, 2.0, 4.5, 8.5, 13.0, 18.0, 23.5, 28.0, 31.0, 35.0, 39.0, 42.0],
  G: [0.75, 1.5, 3.25, 5.5, 8.5, 11.5, 14.5, 18.0, 20.5, 23.5, 27.0, 30.0],
  C: [0.75, 1.5, 3.25, 5.5, 8.5, 11.5, 14.5, 18.0, 20.5, 23.5, 27.0, 30.0],
  ED: [0.75, 2.0, 4.5, 9.5, 15.0, 22.0, 40.0, 44.0, 47.0, 50.0, 53.0, 56.0],
  DI: [0.5, 1.5, 2.75, 5.0, 8.0, 11.0, 14.0, 18.0, 20.5, 23.0, 25.5, 28.0],
  LB: [0.75, 2.0, 3.5, 6.5, 9.5, 12.5, 16.0, 20.0, 22.5, 25.5, 28.0, 30.0],
  CB: [0.75, 2.0, 4.5, 8.5, 13.0, 17.5, 22.5, 27.5, 30.5, 33.0, 35.0, 36.0],
  S: [0.75, 1.5, 3.5, 6.5, 9.5, 12.0, 15.0, 18.5, 21.0, 24.0, 26.0, 28.0],
};

function interp(x, xs, ys) {
  if (x <= xs[0]) return ys[0];
  if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
  for (let i = 1; i < xs.length; i += 1) {
    if (x <= xs[i]) {
      const t = (x - xs[i - 1]) / (xs[i] - xs[i - 1]);
      return ys[i - 1] + t * (ys[i] - ys[i - 1]);
    }
  }
  return ys[ys.length - 1];
}

/** Mirrors Python `grade_to_market_value` */
export function gradeToMarketAav(grade, valueAnchors) {
  const g = Math.max(45, Math.min(100, Number(grade)));
  return Math.round(interp(g, FA_GRADE_ANCHORS, valueAnchors) * FA_MARKET_CALIBRATION_FACTOR * 100) / 100;
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
 * - Elite: 80–100 → m(80)–m(100)
 * - Good: 74–80 → m(74)–m(80)
 * - Starter: 62–74 → m(62)–m(74)
 * - Rotation/backup: &lt;62 → below fair AAV at grade 62
 */
export function buildMarketTierLegend(abbr, valueAnchors, extraNote = '') {
  const raw = (g) => gradeToMarketAav(g, valueAnchors);
  const m = (g) =>
    Math.round(raw(g) * FA_LEGEND_AAV_DISPLAY_FACTOR * 100) / 100;
  const m62 = m(62);
  const m74 = m(74);
  const m80 = m(80);
  const m100 = m(100);

  const tierNote =
    'Tier cutoffs match the backend agent (<code>_grade_to_tier</code>: Elite ≥80, Good ≥74, Starter ≥62; Rotation/backup &lt;62). ' +
    'Dollar amounts are the same OTC-interpolated fair AAV curve used by the backend at grades 62 / 74 / 80 / 100.';

  return {
    title: `2026 Fair AAV by Tier (${abbr})`,
    tiers: [
      { cls: 'elite', label: 'Elite', range: `$${fmtM(m80)}–${fmtM(m100)}M` },
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
    legend: buildMarketTierLegend('QB', FA_VALUE_ANCHORS.QB, 'Composite: 45% model pass grade + 55% stats (rating, YPA, BTT%, comp%, EPA/db), with extra weight on rating and YPA.'),
    statRows: [
      { key: 'yards', label: 'Pass Yards' },
      { key: 'touchdowns', label: 'Pass TDs' },
      { key: 'interceptions', label: 'INTs' },
      { key: 'ypa', label: 'YPA' },
      { key: 'qb_rating', label: 'Passer Rating' },
      { key: 'completion_pct', label: 'Comp %', fmt: FMT.pct },
      { key: 'btt_rate', label: 'BTT Rate', fmt: FMT.btt },
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
    legend: buildMarketTierLegend('HB', FA_VALUE_ANCHORS.HB),
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
    legend: buildMarketTierLegend('WR', FA_VALUE_ANCHORS.WR),
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
    legend: buildMarketTierLegend('TE', FA_VALUE_ANCHORS.TE),
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
    legend: buildMarketTierLegend('T', FA_VALUE_ANCHORS.T, 'Tackle market curve (pass-block premium) from <code>ol_agent_graph</code>.'),
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
    legend: buildMarketTierLegend('G', FA_VALUE_ANCHORS.G),
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
    legend: buildMarketTierLegend('C', FA_VALUE_ANCHORS.C),
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
      FA_VALUE_ANCHORS.ED,
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
      FA_VALUE_ANCHORS.DI,
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
    legend: buildMarketTierLegend('LB', FA_VALUE_ANCHORS.LB),
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
    legend: buildMarketTierLegend('CB', FA_VALUE_ANCHORS.CB),
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
    legend: buildMarketTierLegend('S', FA_VALUE_ANCHORS.S),
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
