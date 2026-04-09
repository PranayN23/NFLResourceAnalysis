import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import './FreeAgency.css';
import {
  POSITION_FREE_AGENCY,
  FA_PICKER_POSITIONS,
  FA_VALUE_ANCHORS,
  gradeToMarketAav,
} from '../config/freeAgencyPositionConfig';

const FA_FMT = {
  pct: (v) => `${Number(v).toFixed(1)}%`,
  pct0: (v) => `${Math.round(Number(v))}%`,
  btt: (v) => `${(Number(v) * 100).toFixed(2)}%`,
  rate4: (v) => Number(v).toFixed(4),
  sacks_rate: (v) => `${(Number(v) * 100).toFixed(2)}%`,
};

/* ─── Searchable Select (combobox) ─── */
function SearchableSelect({ options, value, onChange, placeholder = 'Search…' }) {
  const [query, setQuery] = useState('');
  const [open, setOpen] = useState(false);
  const [highlightIdx, setHighlightIdx] = useState(0);
  const wrapperRef = useRef(null);
  const listRef = useRef(null);

  const filtered = useMemo(() => {
    if (!query) return options;
    const q = query.toLowerCase();
    return options.filter(o => o.toLowerCase().includes(q));
  }, [options, query]);

  useEffect(() => { setHighlightIdx(0); }, [filtered]);

  useEffect(() => {
    const handler = (e) => {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target)) setOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  useEffect(() => {
    if (open && listRef.current) {
      const el = listRef.current.children[highlightIdx];
      if (el) el.scrollIntoView({ block: 'nearest' });
    }
  }, [highlightIdx, open]);

  const pick = (val) => { onChange(val); setQuery(''); setOpen(false); };

  const handleKeyDown = (e) => {
    if (!open && (e.key === 'ArrowDown' || e.key === 'Enter')) { setOpen(true); return; }
    if (e.key === 'ArrowDown') setHighlightIdx(i => Math.min(i + 1, filtered.length - 1));
    else if (e.key === 'ArrowUp') setHighlightIdx(i => Math.max(i - 1, 0));
    else if (e.key === 'Enter' && filtered[highlightIdx]) { pick(filtered[highlightIdx]); e.preventDefault(); }
    else if (e.key === 'Escape') setOpen(false);
  };

  return (
    <div className="fa-searchable-select" ref={wrapperRef}>
      <input
        className={`fa-ss-input${value && !open ? ' fa-ss-has-value' : ''}`}
        placeholder={open || !value ? placeholder : ''}
        value={open ? query : value || ''}
        readOnly={!open}
        onChange={e => { setQuery(e.target.value); if (!open) setOpen(true); }}
        onFocus={() => { setOpen(true); setQuery(''); }}
        onKeyDown={handleKeyDown}
      />
      <span className="fa-ss-arrow">{open ? '▲' : '▼'}</span>
      {open && (
        <ul className="fa-ss-list" ref={listRef}>
          {filtered.length === 0 && <li className="fa-ss-empty">No matches</li>}
          {filtered.map((o, i) => (
            <li key={o}
              className={`fa-ss-item ${o === value ? 'fa-ss-selected' : ''} ${i === highlightIdx ? 'fa-ss-highlight' : ''}`}
              onMouseEnter={() => setHighlightIdx(i)}
              onMouseDown={() => pick(o)}>
              {o}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

const POSITIONS = FA_PICKER_POSITIONS;

/* ─── Decision Tier Legend (shared) ─── */
const TIER_LADDER_BASE = [
  { key: 'exceptional', label: 'Exceptional Value', range: '> 20% surplus',   color: '#3de87a' },
  { key: 'good',        label: 'Good Signing',       range: '5 – 20% surplus', color: '#5dbb6e' },
  { key: 'fair',        label: 'Fair Deal',           range: '± 5% of value',  color: '#d4c94a' },
  { key: 'slight-overpay', label: 'Slight Overpay',  range: '5 – 15% over',   color: '#e8a44a' },
  { key: 'overpay',     label: 'Overpay',             range: '15 – 30% over',  color: '#e07030' },
  { key: 'poor',        label: 'Poor Signing',        range: '> 30% over',     color: '#e05555' },
];

const TIER_LADDER_TEAM = [
  { key: 'must-sign',          label: 'Must Sign',            color: '#00e5ff' },
  { key: 'priority',           label: 'Priority Target',      color: '#00bcd4' },
  { key: 'exceptional',        label: 'Exceptional Value',    color: '#3de87a' },
  { key: 'luxury-great',       label: 'Luxury — Great Value', color: '#66ffb2' },
  { key: 'good',               label: 'Good Signing',         color: '#5dbb6e' },
  { key: 'fill-gap',           label: 'Fill the Gap',         color: '#8bc34a' },
  { key: 'luxury',             label: 'Luxury Add',           color: '#a5d6a7' },
  { key: 'fair',               label: 'Fair Deal',            color: '#d4c94a' },
  { key: 'justifiable',        label: 'Justifiable Overpay',  color: '#cddc39' },
  { key: 'slight-overpay',     label: 'Slight Overpay',       color: '#e8a44a' },
  { key: 'unnecessary',        label: 'Unnecessary Spend',    color: '#ff9800' },
  { key: 'overpay-consider',   label: 'Overpay — Consider',   color: '#ef6c00' },
  { key: 'overpay',            label: 'Overpay',              color: '#e07030' },
  { key: 'wasteful',           label: 'Wasteful Overpay',     color: '#d84315' },
  { key: 'desperation',        label: 'Desperation Overpay',  color: '#c62828' },
  { key: 'poor',               label: 'Poor Signing',         color: '#e05555' },
  { key: 'cap-mismanage',      label: 'Cap Mismanagement',    color: '#b71c1c' },
  { key: 'exceeds-cap',        label: 'Exceeds Cap',          color: '#ff1744' },
];

const TIER_DESCRIPTION = {
  'Must Sign':                     'Elite value at a position of need — cap-friendly deal you can\'t pass up.',
  'Priority Target':               'Good value at a weak position group — fits the cap and fills a real need.',
  'Exceptional Value':             'More than 20% below fair value — a steal regardless of team context.',
  'Luxury Add — Great Value':      'Elite value, but the position is already strong — a luxury the team can afford.',
  'Good Signing':                  'Solid 5–20% surplus — good player at a fair price for a moderate need.',
  'Fill the Gap':                  'Market rate at a weak position — worth paying to address a clear hole.',
  'Luxury Add':                    'Good value at an already-strong position — nice depth if cap allows.',
  'Fair Deal':                     'Right at market value — reasonable deal for a moderate positional need.',
  'Justifiable Overpay':           'Slight premium, but the team badly needs help here — cap is manageable.',
  'Slight Overpay':                '5–15% above fair value — moderate need doesn\'t fully justify the premium.',
  'Unnecessary Spend':             'Paying market rate at a stacked position — cap dollars better spent elsewhere.',
  'Overpay — But Consider':        'Significant overpay with heavy cap impact, but positional weakness may warrant it.',
  'Overpay':                       '15–30% above fair value — strains the cap with no strong positional justification.',
  'Wasteful Overpay':              'Overpaying at an already-strong position — wastes cap space on redundancy.',
  'Desperation Overpay':           'Severe overpay even for a position of need — cripples future cap flexibility.',
  'Poor Signing':                  'More than 30% over fair value — no positional need or cap room justifies this.',
  'Cap Mismanagement':             'Massive overpay at a position that\'s already stacked — destructive to the cap.',
  'Exceeds Cap':                   'This signing doesn\'t fit under the team\'s available salary cap.',
};

function DecisionTierLegend({ teamMode }) {
  const [open, setOpen] = useState(false);
  const ladder = teamMode ? TIER_LADDER_TEAM : TIER_LADDER_BASE;
  return (
    <div className="fa-tier-legend">
      <p className="fa-legend-title fa-legend-toggle" onClick={() => setOpen(o => !o)}>
        {teamMode ? 'Team-Aware Decision Tiers' : 'Decision Tiers'}
        <span className={`fa-toggle-arrow ${open ? 'open' : ''}`}>▸</span>
      </p>
      {open && (
        <>
          {ladder.map((t, i) => (
            <div key={t.key} className="fa-tier-row">
              <span className="fa-tier-rank">{i + 1}</span>
              <span className="fa-tier-dot" style={{ background: t.color }} />
              <span className="fa-tier-label" style={{ color: t.color }}>{t.label}</span>
            </div>
          ))}
          {teamMode && (
            <p className="fa-legend-note">
              Accounts for value, positional strength, and cap space.<br/>
              Full explanation shown below each result.
            </p>
          )}
        </>
      )}
    </div>
  );
}

/* ─── Signing Grade ─── */
const _SURPLUS_ANCHORS = [-60, -30, -15,  -5,   0,   5,  15,  25,  40,  60];
const _GRADE_ANCHORS   = [  0,  20,  35,  50,  58,  68,  80,  88,  95, 100];

function _lerp(xs, ys, x) {
  if (x <= xs[0]) return ys[0];
  if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
  for (let i = 1; i < xs.length; i++) {
    if (x <= xs[i]) {
      const t = (x - xs[i - 1]) / (xs[i] - xs[i - 1]);
      return ys[i - 1] + t * (ys[i] - ys[i - 1]);
    }
  }
  return ys[ys.length - 1];
}

function signingGradeFromData(fair_aav, cap_burden, teamCtx) {
  const surplus_pct = (fair_aav - cap_burden) / Math.max(fair_aav, 0.01) * 100;
  let base = _lerp(_SURPLUS_ANCHORS, _GRADE_ANCHORS, surplus_pct);

  if (teamCtx && teamCtx.need_label) {
    const strength = teamCtx.positional_need || 50;
    const teamAdj = (50 - strength) * 0.24;

    const yr1Pct = (teamCtx.signing_cap_pcts || [])[0] || 0;
    const availPct = teamCtx.available_cap_pct || 100;
    const capRatio = yr1Pct / Math.max(availPct, 0.01);

    // Cap flexibility adjustment — continuous scale:
    //   capRatio ~0.10 (tiny % of cap) → +6 pts  (team can easily absorb this)
    //   capRatio ~0.25                  → +3 pts
    //   capRatio ~0.35                  →  0 pts  (neutral)
    //   capRatio ~0.50                  → -5 pts  (eating half the remaining cap)
    //   capRatio ~0.75                  → -12 pts (almost no room left after)
    //   capRatio ~1.00+                 → -18 pts (barely fits or exceeds cap)
    const capAdj = _lerp(
      [0.0,  0.10, 0.25, 0.35, 0.50, 0.75, 1.0],
      [8,    6,    3,    0,    -5,   -12,  -18],
      capRatio
    );

    // Absolute cap room bonus: teams with lots of space can afford overpays
    //   availPct >= 25% → +4 pts (flush with cash)
    //   availPct ~15%   → +2 pts
    //   availPct ~10%   →  0 pts (neutral)
    //   availPct <= 5%  → -4 pts (cap-strapped, every dollar matters)
    const roomAdj = _lerp(
      [2,   5,   10,  15,  25],
      [-4,  -3,   0,   2,   4],
      availPct
    );

    base = base + teamAdj + capAdj + roomAdj;
  }

  return Math.round(Math.max(0, Math.min(100, base)));
}

function gradeColor(g) {
  if (g >= 80) return '#3de87a';
  if (g >= 68) return '#5dbb6e';
  if (g >= 55) return '#d4c94a';
  if (g >= 40) return '#e8a44a';
  if (g >= 22) return '#e07030';
  return '#e05555';
}

function gradeLetter(g) {
  if (g >= 97) return 'A+';
  if (g >= 90) return 'A';
  if (g >= 83) return 'A-';
  if (g >= 76) return 'B+';
  if (g >= 68) return 'B';
  if (g >= 60) return 'B-';
  if (g >= 52) return 'C+';
  if (g >= 44) return 'C';
  if (g >= 36) return 'C-';
  if (g >= 28) return 'D+';
  if (g >= 20) return 'D';
  if (g >= 12) return 'D-';
  return 'F';
}

function SigningGrade({ grade }) {
  const color = gradeColor(grade);
  const letter = gradeLetter(grade);
  const pct = grade;
  return (
    <div className="fa-signing-grade">
      <svg className="fa-grade-ring" viewBox="0 0 80 80">
        <circle cx="40" cy="40" r="34" fill="none" stroke="#2a2a2a" strokeWidth="7" />
        <circle cx="40" cy="40" r="34" fill="none" stroke={color} strokeWidth="7"
          strokeDasharray={`${2 * Math.PI * 34 * pct / 100} ${2 * Math.PI * 34 * (1 - pct / 100)}`}
          strokeDashoffset={2 * Math.PI * 34 * 0.25}
          strokeLinecap="round" />
        <text x="40" y="37" textAnchor="middle" fill={color} fontSize="16" fontWeight="bold">{grade}</text>
        <text x="40" y="52" textAnchor="middle" fill={color} fontSize="10">{letter}</text>
      </svg>
      <p className="fa-grade-label">Signing Grade</p>
    </div>
  );
}

/* ─── Need Badge ─── */
function NeedBadge({ label, score }) {
  const color = label === 'Weak' ? '#e05555'
              : label === 'Well-Stocked' ? '#3de87a'
              : '#d4c94a';
  return (
    <span className="fa-need-badge" style={{ borderColor: color, color }}>
      {label} ({Math.round(score)}/100)
    </span>
  );
}

/* ─── Roster Preview ─── */
const SALARY_CAP_M = 255.4;
function pctToDollars(pct) { return (pct / 100 * SALARY_CAP_M).toFixed(1); }
function capDisplay(pct) { return `${pct.toFixed(1)}% ($${pctToDollars(pct)}M)`; }

function RosterPreview({ roster, needLabel, needScore, allocatedPct, availablePct, positionLabel }) {
  const top = roster.slice(0, 5);
  return (
    <div className="fa-roster-card">
      <div className="fa-roster-header">
        <span className="fa-roster-title">{positionLabel} Roster</span>
        <NeedBadge label={needLabel} score={needScore} />
      </div>
      <p className="fa-roster-note">Strength at {positionLabel} only — not overall team.</p>
      <div className="fa-cap-bar-row">
        <span className="fa-cap-label">Cap Used</span>
        <div className="fa-cap-bar">
          <div className="fa-cap-fill" style={{ width: `${Math.min(allocatedPct, 100)}%` }} />
        </div>
        <span className="fa-cap-text">{allocatedPct.toFixed(1)}% (${pctToDollars(allocatedPct)}M)</span>
      </div>
      <div className="fa-cap-bar-row">
        <span className="fa-cap-label">Available</span>
        <span className="fa-cap-text fa-cap-avail">{availablePct.toFixed(1)}% (${pctToDollars(availablePct)}M)</span>
      </div>
      {top.length > 0 && (
        <table className="fa-roster-tbl">
          <thead>
            <tr><th>Player</th><th>Age</th><th>Grade</th><th>Snaps</th><th>Cap Hit</th></tr>
          </thead>
          <tbody>
            {top.map((p, i) => (
              <tr key={i}>
                <td className="fa-roster-name">{p.player}</td>
                <td>{p.age}</td>
                <td>
                  <div className="fa-roster-grade-cell">
                    <span>{p.grade}</span>
                    <span className={`fa-roster-grade-bar ${rosterGradeTierClass(p.grade)}`} aria-hidden="true" />
                  </div>
                </td>
                <td>{p.snaps}</td>
                <td>{p.cap_pct}% (${pctToDollars(p.cap_pct)}M)</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

/* ─── Team Fit Section (in result card) ─── */
function TeamFitSection({ teamCtx, signingPcts, positionLabel }) {
  if (!teamCtx) return null;
  const yr1 = signingPcts?.[0] || 0;
  const yrLast = signingPcts?.[signingPcts.length - 1] || yr1;
  const capAfter = Math.max(0, (teamCtx.available_cap_pct || 0) - yr1);
  return (
    <div className="fa-team-fit-section">
      <div className="fa-stat-section-hdr">Team Fit — {positionLabel}</div>
      <div className="fa-stat-row">
        <span className="fa-stat-label">Team</span>
        <span className="fa-stat-value">
          {teamCtx.team}
          {teamCtx.is_re_signing && <span className="fa-re-sign-tag">RE-SIGNING</span>}
        </span>
      </div>
      {teamCtx.is_re_signing && (
        <div className="fa-re-sign-note">
          Player is already on this roster. Need is calculated without them (what if they leave?),
          and their current cap hit ({teamCtx.freed_cap_pct?.toFixed(1)}% / ${pctToDollars(teamCtx.freed_cap_pct || 0)}M) is freed up.
        </div>
      )}
      <div className="fa-stat-row">
        <span className="fa-stat-label">{positionLabel} Need{teamCtx.is_re_signing ? ' (without player)' : ''}</span>
        <span className="fa-stat-value">
          <NeedBadge label={teamCtx.need_label} score={teamCtx.positional_need} />
        </span>
      </div>
      <div className="fa-stat-row">
        <span className="fa-stat-label">Available Cap{teamCtx.is_re_signing ? ' (after freeing)' : ''}</span>
        <span className="fa-stat-value">{capDisplay(teamCtx.available_cap_pct || 0)}</span>
      </div>
      <div className="fa-stat-row">
        <span className="fa-stat-label">Yr 1 Cap Hit</span>
        <span className="fa-stat-value">{capDisplay(yr1)}</span>
      </div>
      {signingPcts?.length > 1 && (
        <div className="fa-stat-row">
          <span className="fa-stat-label">Yr {signingPcts.length} Cap Hit</span>
          <span className="fa-stat-value">{yrLast.toFixed(1)}% (${pctToDollars(yrLast)}M) — shrinks with cap growth</span>
        </div>
      )}
      <div className="fa-stat-row">
        <span className="fa-stat-label">Cap After Signing</span>
        <span className="fa-stat-value">{capDisplay(capAfter)}</span>
      </div>
      {teamCtx.fit_summary && (
        <div className="fa-stat-row">
          <span className="fa-stat-label">Fit Summary</span>
          <span className="fa-stat-value fa-fit-note">{teamCtx.fit_summary}</span>
        </div>
      )}
    </div>
  );
}

/* ─── Position Picker ─── */
function PositionPicker({ onSelect }) {
  return (
    <div className="fa-picker-page">
      <div className="fa-picker-card">
        <h1 className="fa-picker-title">Free Agency Evaluator</h1>
        <p className="fa-picker-sub">Select a position to evaluate free agents</p>
        <div className="fa-picker-grid">
          {POSITIONS.map((pos) => (
            <button
              key={pos.label}
              type="button"
              className="fa-pos-btn fa-pos-btn--active"
              onClick={() => onSelect(pos.label)}
            >
              <span className="fa-pos-abbr">{pos.label}</span>
              <span className="fa-pos-name">{pos.name}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ─── Year Breakdown Table ─── */
function YearBreakdown({ rows }) {
  return (
    <div className="fa-breakdown">
      <p className="fa-breakdown-title">Year-by-Year Projection</p>
      <table className="fa-breakdown-table">
        <thead>
          <tr>
            <th>Yr</th>
            <th>Age</th>
            <th>Grade</th>
            <th>Mkt Val</th>
            <th>Cap-Adj Ask</th>
            <th>Δ</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const delta = r.market_value - r.cap_adj_ask;
            return (
              <tr key={r.year}>
                <td>{r.year}</td>
                <td>{r.age}</td>
                <td>{r.projected_grade}</td>
                <td>${r.market_value}M</td>
                <td>${r.cap_adj_ask}M</td>
                <td className={delta >= 0 ? 'fa-pos-delta' : 'fa-neg-delta'}>
                  {delta >= 0 ? '+' : ''}{delta.toFixed(2)}M
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="fa-breakdown-note">
        Cap-Adj Ask = fixed AAV deflated by 6.5%/yr cap growth — shows the real cap burden shrinks each year.
        Grade curve: rises for developing players (≤26), peaks at 27–29, then declines per empirical data.
        Δ = player market value minus cap-adjusted ask.
      </p>
    </div>
  );
}

/* ─── Stats Panel (position-specific columns from config) ─── */
function PositionStatsPanel({ careerStats, projectedStats, statRows, note }) {
  const lastCareer = careerStats[careerStats.length - 1];

  const fmtVal = (row, raw) => {
    if (raw == null) return null;
    const fn = row.fmt && FA_FMT[row.fmt];
    return fn ? fn(raw) : raw;
  };

  const projRaw = (yr, row) => {
    if (row.projKey) return yr[row.projKey];
    if (row.key === 'interceptions') {
      const v = yr.interceptions ?? yr.ints ?? yr.ints_17g ?? null;
      return (typeof v === 'number' && !Number.isFinite(v)) ? null : v;
    }
    if (row.key === 'overall_grade' || row.key === 'defense_grade')
      return yr[row.key] ?? yr.projected_grade;
    const v = yr[row.key];
    return (typeof v === 'number' && !Number.isFinite(v)) ? null : v;
  };

  return (
    <div className="fa-stats-panel">
      <p className="fa-stats-panel-title">Career Stats + Projection</p>
      <div className="fa-stats-scroll">
        <table className="fa-stats-tbl">
          <thead>
            <tr>
              <th className="fa-stats-row-hdr">Stat</th>
              {careerStats.map(s => (
                <th key={s.season} className="fa-stats-col-career">
                  <div className="fa-stats-col-hdr">{s.season}</div>
                  <div className="fa-stats-col-sub">{s.games_played}/{s.max_games}g</div>
                </th>
              ))}
              {projectedStats.map(yr => (
                <th key={`proj-${yr.year}`} className="fa-stats-col-proj">
                  <div className="fa-stats-col-hdr">Yr {yr.year}</div>
                  <div className="fa-stats-col-sub">Age {yr.age}</div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {statRows.map((row) => (
              <tr key={row.key + (row.projKey || '')}>
                <td className="fa-stats-row-hdr">{row.label}</td>
                {careerStats.map(s => (
                  <td key={s.season} className="fa-stats-career-cell">
                    {(row.key === 'interceptions'
                      ? (s.interceptions ?? s.ints ?? null)
                      : s[row.key]) != null
                      ? fmtVal(row, row.key === 'interceptions' ? (s.interceptions ?? s.ints) : s[row.key])
                      : '—'}
                  </td>
                ))}
                {projectedStats.map(yr => {
                  const raw = projRaw(yr, row);
                  const val = fmtVal(row, raw);
                  const last = lastCareer?.[row.key];
                  const rawProj = projRaw(yr, row);
                  let delta = null;
                  if (!row.noDelta && rawProj != null && last != null && Number(last) !== 0) {
                    delta = Number(rawProj) - Number(last);
                  }
                  return (
                    <td key={yr.year} className={
                      delta == null ? '' : delta > 0.05 ? 'fa-stat-up' : delta < -0.05 ? 'fa-stat-down' : ''
                    }>
                      {raw != null ? val : '—'}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {note && <p className="fa-breakdown-note">{note}</p>}
    </div>
  );
}

/* ─── Decision class map (shared) ─── */
const DECISION_CLASS = {
  'Must Sign — Elite Value + Need': 'must-sign',
  'Priority Target':                'priority',
  'Exceptional Value':              'exceptional',
  'Luxury Add — Great Value':       'luxury-great',
  'Good Signing':                   'good',
  'Fill the Gap':                   'fill-gap',
  'Luxury Add':                     'luxury',
  'Fair Deal':                      'fair',
  'Justifiable Overpay':            'justifiable',
  'Slight Overpay':                 'slight-overpay',
  'Unnecessary Spend':              'unnecessary',
  'Overpay — But Consider':         'overpay-consider',
  'Overpay':                        'overpay',
  'Wasteful Overpay':               'wasteful',
  'Desperation Overpay':            'desperation',
  'Poor Signing':                   'poor',
  'Cap Mismanagement':              'cap-mismanage',
  'Exceeds Cap':                    'exceeds-cap',
};

/** Maps agent `predicted_tier` to sidebar legend `.tier-badge.*` classes. */
function predictedTierToBadgeClass(tier) {
  if (tier == null || tier === '') return null;
  const t = String(tier).trim().toLowerCase();
  if (t.startsWith('elite')) return 'elite';
  if (t.startsWith('good')) return 'good';
  if (t.startsWith('starter')) return 'starter';
  if (t.startsWith('rotation') || t.includes('backup')) return 'rotation';
  if (t.startsWith('reserve') || t.includes('poor')) return 'reserve';
  return null;
}

function rosterGradeTierClass(grade) {
  const g = Number(grade);
  if (!Number.isFinite(g)) return 'reserve';
  if (g >= 80) return 'elite';
  if (g >= 74) return 'good';
  if (g >= 62) return 'starter';
  if (g >= 50) return 'rotation';
  return 'reserve';
}

function tierFromFairAav(positionKey, fairAav) {
  const anchors = FA_VALUE_ANCHORS[positionKey];
  if (!anchors || fairAav == null || !Number.isFinite(Number(fairAav))) return null;
  const a = Number(fairAav);
  const t62 = gradeToMarketAav(62, anchors);
  const t74 = gradeToMarketAav(74, anchors);
  const t80 = gradeToMarketAav(80, anchors);
  if (a >= t80) return 'Elite';
  if (a >= t74) return 'Good';
  if (a >= t62) return 'Starter';
  return 'Rotation/backup';
}

function buildStructuredFreeAgent(result, ask, years, positionKey) {
  const { decision, reasoning, data, team_context } = result;
  const {
    predicted_tier, current_age, effective_fair_aav, effective_cap_burden,
    total_nominal_value, total_ask, confidence, year_breakdown,
    projected_stats, career_stats,
  } = data;
  const { model_grade, stats_grade, composite_grade, health_factor, avg_availability,
    transformer_grade, xgb_grade, age_adjustment } = confidence || {};

  const salaryAlignedTier = tierFromFairAav(positionKey, Number(effective_fair_aav));
  const displayedTier = salaryAlignedTier || predicted_tier;

  const statRows = [
    { divider: true, title: 'Player Profile' },
    {
      label: 'Projected Tier',
      value: displayedTier || 'N/A',
      tierBadgeClass: predictedTierToBadgeClass(displayedTier),
    },
    { label: 'Current Age', value: current_age },
    { divider: true, title: 'Grade Breakdown' },
    { label: 'Model Grade', value: model_grade != null ? `${Number(model_grade).toFixed(1)} / 100` : 'N/A' },
    { label: 'Stats Grade', value: stats_grade != null ? `${Number(stats_grade).toFixed(1)} / 100` : 'N/A' },
    { label: 'Composite Grade', value: composite_grade != null ? `${Number(composite_grade).toFixed(1)} / 100` : 'N/A' },
  ];
  if (transformer_grade != null) {
    statRows.push({ label: 'Transformer Grade', value: Number(transformer_grade).toFixed(1) });
  }
  if (xgb_grade != null) {
    statRows.push({ label: 'XGBoost Grade', value: Number(xgb_grade).toFixed(1) });
  }
  if (age_adjustment != null && age_adjustment !== 0) {
    statRows.push({ label: 'Age Penalty (applied)', value: `-${Number(age_adjustment).toFixed(1)} pts` });
  }

  statRows.push(
    { divider: true, title: 'Health & Availability' },
    { label: 'Availability (3yr)', value: avg_availability != null ? `${Math.round(avg_availability * 100)}%` : 'N/A' },
    { label: 'Health Factor', value: health_factor != null ? `${health_factor >= 0 ? '+' : ''}${Number(health_factor).toFixed(1)} pts` : 'N/A' },
    { divider: true, title: 'Contract Valuation' },
    { label: 'Contract', value: `$${ask}M/yr × ${years} yr  =  $${total_ask}M total` },
    { label: 'Fair AAV (cap-adj PV)', value: `$${effective_fair_aav}M / yr` },
    { label: 'Real Cap Burden (PV)', value: `$${effective_cap_burden}M / yr` },
    { label: 'Total Nominal Value', value: `$${total_nominal_value}M` },
  );

  const normalizedProjected = (projected_stats || []).map((yr) => ({
    ...yr,
    interceptions: yr?.interceptions ?? yr?.ints ?? yr?.int ?? yr?.ints_17g ?? null,
  }));
  const normalizedCareer = (career_stats || []).map((s) => ({
    ...s,
    interceptions: s?.interceptions ?? s?.ints ?? s?.int ?? null,
  }));

  return {
    decision,
    highlight: DECISION_CLASS[decision] || 'fair',
    signing_grade: signingGradeFromData(effective_fair_aav, effective_cap_burden, team_context),
    tier_description: TIER_DESCRIPTION[decision] || '',
    stats: statRows,
    reasoning,
    year_breakdown,
    projected_stats: normalizedProjected,
    career_stats: normalizedCareer,
    team_context: team_context || null,
    meta: {
      positionKey,
      ask: Number(ask),
      years: Number(years),
      fairAav: Number(effective_fair_aav),
      burdenAav: Number(effective_cap_burden),
      playerName: result?.player || '',
      team: team_context?.team || '',
      analysisYear: Number(data?.analysis_year || 2025),
    },
  };
}

function PositionEvaluator({ positionKey, onBack, onSwitchPosition, pendingPick, clearPendingPick }) {
  const cfg = POSITION_FREE_AGENCY[positionKey];
  const apiBase = `http://127.0.0.1:${cfg.port}`;
  const contractMax = 7;
  const latestAnalysisYear = 2025;

  const [players, setPlayers] = useState([]);
  const [selectedPlayer, setSelectedPlayer] = useState('');
  const [salaryAsk, setSalaryAsk] = useState('');
  const [contractYears, setContractYears] = useState(1);
  const [loading, setLoading] = useState(false);
  const [fetchingPlayers, setFetchingPlayers] = useState(true);
  const [messages, setMessages] = useState([{ role: 'assistant', content: cfg.welcome }]);
  const [error, setError] = useState('');
  const [statsOpen, setStatsOpen] = useState({});

  const [teamMode, setTeamMode] = useState(false);
  const [teams, setTeams] = useState([]);
  const [selectedTeam, setSelectedTeam] = useState('');
  const [teamRoster, setTeamRoster] = useState(null);
  const [capOverride, setCapOverride] = useState('');
  const [capOverrideDirty, setCapOverrideDirty] = useState(false);
  const [fetchingTeam, setFetchingTeam] = useState(false);
  const [summarizingTeam, setSummarizingTeam] = useState(false);
  const [showRankingsDialog, setShowRankingsDialog] = useState(false);
  const [loadingRankings, setLoadingRankings] = useState(false);
  const [teamRankings, setTeamRankings] = useState([]);
  const [classBuilderOn, setClassBuilderOn] = useState(false);
  const [classSignings, setClassSignings] = useState([]);
  const [showClassDialog, setShowClassDialog] = useState(false);
  const [classStartCapPct, setClassStartCapPct] = useState(null);
  const [classStartCapInput, setClassStartCapInput] = useState('');
  const [classCapLocked, setClassCapLocked] = useState(false);
  const [departuresOn, setDeparturesOn] = useState(false);
  const [classDepartures, setClassDepartures] = useState([]);
  const [selectedDeparturePlayer, setSelectedDeparturePlayer] = useState('');
  const [playerDirectory, setPlayerDirectory] = useState([]);
  const [classSearchPlayer, setClassSearchPlayer] = useState('');
  const [classQuickAsk, setClassQuickAsk] = useState('');
  const [classQuickYears, setClassQuickYears] = useState(3);
  const [classQuickLoading, setClassQuickLoading] = useState(false);
  const [analysisYear, setAnalysisYear] = useState(latestAnalysisYear);
  const [analysisYearMin, setAnalysisYearMin] = useState(2010);

  const chatEndRef = useRef(null);

  const toggleStats = useCallback((i) => {
    setStatsOpen((prev) => ({ ...prev, [i]: !prev[i] }));
  }, []);

  useEffect(() => {
    const c = POSITION_FREE_AGENCY[positionKey];
    if (!messages.length) setMessages([{ role: 'assistant', content: c.welcome }]);
    setPlayers([]);
    setSelectedPlayer('');
    setError('');
    setStatsOpen({});
  }, [positionKey]);

  useEffect(() => {
    setFetchingPlayers(true);
    fetch(`${apiBase}${cfg.playersPath}`)
      .then((r) => r.json())
      .then((data) => {
        const nextPlayers = data.players || [];
        setPlayers(nextPlayers);
        setSelectedPlayer((prev) => {
          if (prev && nextPlayers.includes(prev)) return prev;
          return nextPlayers[0] || '';
        });
        const minYr = Number(data.analysis_year_min);
        setAnalysisYearMin(Number.isFinite(minYr) ? minYr : 2010);
      })
      .catch(() =>
        setError(`Could not load player list. Start the ${positionKey} API (port ${cfg.port}): uvicorn backend.agent…`)
      )
      .finally(() => setFetchingPlayers(false));

    fetch(`${apiBase}/player-directory?analysis_year=${encodeURIComponent(analysisYear)}`)
      .then((r) => r.json())
      .then((data) => {
        setPlayerDirectory(data.players || []);
      })
      .catch(() => setPlayerDirectory([]));

    fetch(`${apiBase}/teams?analysis_year=${encodeURIComponent(analysisYear)}`)
      .then((r) => r.json())
      .then((data) => {
        const nextTeams = data.teams || [];
        setTeams(nextTeams);
        setSelectedTeam((prev) => {
          if (!prev) return '';
          return nextTeams.includes(prev) ? prev : '';
        });
      })
      .catch(() => {});
  }, [apiBase, cfg.playersPath, cfg.port, positionKey, analysisYear]);

  useEffect(() => {
    if (!pendingPick || pendingPick.positionKey !== positionKey) return;
    if (pendingPick.playerName) setSelectedPlayer(pendingPick.playerName);
    if (clearPendingPick) clearPendingPick();
  }, [pendingPick, positionKey, clearPendingPick]);

  useEffect(() => {
    if (!selectedTeam || !teamMode) return;
    setFetchingTeam(true);
    fetch(`${apiBase}/team-roster?team=${encodeURIComponent(selectedTeam)}&analysis_year=${encodeURIComponent(analysisYear)}`)
      .then((r) => r.json())
      .then((data) => {
        setTeamRoster(data);
        if (!capOverrideDirty) {
          setCapOverride(pctToDollars(data.available_cap_pct || 0));
        }
      })
      .catch(() => setTeamRoster(null))
      .finally(() => setFetchingTeam(false));
  }, [selectedTeam, teamMode, apiBase, capOverrideDirty, analysisYear]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleAnalyze = async () => {
    if (!selectedPlayer) return;
    const ask = parseFloat(salaryAsk);
    if (isNaN(ask) || ask <= 0) {
      setError('Please enter a valid salary (positive number in $M).');
      return;
    }
    setError('');

    const teamLabel = teamMode && selectedTeam ? ` as ${selectedTeam}` : '';
    setMessages((prev) => [
      ...prev,
      {
        role: 'user',
        content: `Evaluate ${selectedPlayer} — $${ask}M/yr × ${contractYears} yr contract${teamLabel}.`,
      },
    ]);
    setLoading(true);

    try {
      const body = {
        player_name: selectedPlayer,
        salary_ask: ask,
        contract_years: contractYears,
        analysis_year: analysisYear,
      };
      if (teamMode && selectedTeam) {
        body.team = selectedTeam;
        const capValM = parseFloat(capOverride);
        if (!isNaN(capValM) && capValM > 0) {
          body.cap_available_pct = (capValM / SALARY_CAP_M) * 100;
        }
      }

      const resp = await fetch(`${apiBase}/evaluate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!resp.ok) {
        let detail = 'Evaluation failed.';
        try {
          const err = await resp.json();
          detail = err.detail || detail;
        } catch { /* ignore */ }
        throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
      }

      const result = await resp.json();
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: null,
          structured: buildStructuredFreeAgent(result, ask, contractYears, positionKey),
        },
      ]);
    } catch (e) {
      setMessages((prev) => [...prev, { role: 'assistant', content: `Error: ${e.message}` }]);
    } finally {
      setLoading(false);
    }
  };

  const ticks = useMemo(() => Array.from({ length: contractMax }, (_, i) => i + 1), [contractMax]);
  const analysisYearOptions = useMemo(() => {
    const minY = Math.max(1900, Math.min(latestAnalysisYear, Number(analysisYearMin) || latestAnalysisYear));
    const out = [];
    for (let y = latestAnalysisYear; y >= minY; y -= 1) out.push(y);
    return out;
  }, [analysisYearMin]);
  const sortedTeamRankings = useMemo(
    () => [...teamRankings].sort((a, b) => Number(a.rank) - Number(b.rank)),
    [teamRankings]
  );
  const classGradeSummary = useMemo(() => {
    if (!classSignings.length) return null;
    const POS_IMPORTANCE = {
      QB: 1.45, ED: 1.25, WR: 1.18, CB: 1.15, T: 1.12, DI: 1.08,
      C: 1.03, G: 1.02, LB: 1.0, S: 0.98, TE: 0.95, HB: 0.9,
    };
    let num = 0;
    let den = 0;
    const rows = classSignings.map((s) => {
      const posW = POS_IMPORTANCE[s.positionKey] || 1.0;
      const profileW = Math.max(0.8, Math.min(1.8, 0.8 + (Number(s.ask) || 0) / 30));
      const w = posW * profileW;
      const g = Number(s.signingGrade) || 0;
      num += g * w;
      den += w;
      return { ...s, weight: w, weightedScore: g * w };
    });
    const grade = den > 0 ? Math.round(num / den) : 0;
    return { grade, letter: gradeLetter(grade), rows };
  }, [classSignings]);
  const classUsedCapPct = useMemo(
    () => classSignings.reduce((acc, s) => acc + Number(s.yr1CapPct || 0), 0),
    [classSignings]
  );
  const classFreedCapPct = useMemo(
    () => (departuresOn ? classDepartures.reduce((acc, d) => acc + Number(d.freedCapPct || 0), 0) : 0),
    [classDepartures, departuresOn]
  );
  const classNetCapPct = useMemo(
    () => classUsedCapPct - classFreedCapPct,
    [classUsedCapPct, classFreedCapPct]
  );
  const classRemainingCapPct = useMemo(() => {
    if (classStartCapPct == null) return null;
    return Number(classStartCapPct) - classNetCapPct;
  }, [classStartCapPct, classNetCapPct]);
  const rosterDepartureOptions = useMemo(
    () => (teamRoster?.roster || []).map((p) => p.player).filter(Boolean),
    [teamRoster]
  );
  const handleTeamChange = useCallback((team) => {
    setSelectedTeam(team);
    setCapOverrideDirty(false);
  }, []);

  const handleCapOverrideChange = useCallback((raw) => {
    // Allow temporary empty/partial numeric text while editing.
    if (raw === '' || /^(\d+(\.\d*)?)?$/.test(raw)) {
      setCapOverride(raw);
      setCapOverrideDirty(true);
    }
  }, []);

  const handleTeamYearSummary = useCallback(async () => {
    if (!selectedTeam) return;
    setSummarizingTeam(true);
    setError('');
    const seasonYear = Math.max(1900, Number(analysisYear) - 1);
    setMessages((prev) => [
      ...prev,
      { role: 'user', content: `Summarize the ${selectedTeam} in ${seasonYear}.` },
    ]);
    try {
      const resp = await fetch(
        `${apiBase}/team-summary?team=${encodeURIComponent(selectedTeam)}&analysis_year=${encodeURIComponent(analysisYear)}`
      );
      if (!resp.ok) throw new Error('Failed to fetch team-year summary.');
      const data = await resp.json();
      const summary = data?.summary || `No summary available for ${selectedTeam} in ${analysisYear}.`;
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: summary,
          teamSummaryMeta: { team: selectedTeam, analysisYear },
        },
      ]);
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: `Error: ${e.message}` },
      ]);
    } finally {
      setSummarizingTeam(false);
    }
  }, [selectedTeam, analysisYear, apiBase]);

  const handleOpenRankingsDialog = useCallback(async (teamArg, yearArg) => {
    const teamName = teamArg || selectedTeam;
    const yearVal = Number.isFinite(Number(yearArg)) ? Number(yearArg) : analysisYear;
    if (!teamName) return;
    setLoadingRankings(true);
    setShowRankingsDialog(true);
    try {
      const resp = await fetch(
        `${apiBase}/team-rankings?team=${encodeURIComponent(teamName)}&analysis_year=${encodeURIComponent(yearVal)}`
      );
      if (!resp.ok) throw new Error('Failed to fetch team rankings.');
      const data = await resp.json();
      setTeamRankings(data?.rankings || []);
    } catch (e) {
      setTeamRankings([]);
      setError(e.message || 'Failed to fetch team rankings.');
    } finally {
      setLoadingRankings(false);
    }
  }, [selectedTeam, analysisYear, apiBase]);

  const handleAddToClass = useCallback((structured) => {
    if (!structured?.meta) return;
    const m = structured.meta;
    if (!m.team) {
      setError('Enable Team Mode with a selected team to add class signings.');
      return;
    }
    const key = `${m.playerName}::${m.positionKey}::${m.team}::${m.analysisYear}`;
    setClassSignings((prev) => {
      if (prev.some((x) => x.key === key)) return prev;
      const contexts = new Set(prev.map((x) => `${x.team}::${x.analysisYear}`));
      if (contexts.size > 0 && !contexts.has(`${m.team}::${m.analysisYear}`)) {
        setError(`Class builder is currently scoped to ${[...contexts][0].replace('::', ' / ')}. Clear class to switch context.`);
        return prev;
      }
      return [
        ...prev,
        {
          key,
          playerName: m.playerName,
          positionKey: m.positionKey,
          team: m.team,
          analysisYear: m.analysisYear,
          ask: m.ask,
          years: m.years,
          signingGrade: structured.signing_grade,
          decision: structured.decision,
          yr1CapPct: Number(structured?.team_context?.signing_cap_pcts?.[0] || 0),
        },
      ];
    });
  }, []);

  const handleClearClass = useCallback(() => {
    setClassSignings([]);
    setClassStartCapPct(null);
    setClassStartCapInput('');
    setClassCapLocked(false);
    setClassDepartures([]);
    setSelectedDeparturePlayer('');
    setDeparturesOn(false);
  }, []);
  const handleAddDeparture = useCallback(() => {
    if (!departuresOn || !selectedDeparturePlayer) return;
    const roster = teamRoster?.roster || [];
    const found = roster.find((p) => p.player === selectedDeparturePlayer);
    if (!found) return;
    const key = `${selectedDeparturePlayer}::${analysisYear}`;
    setClassDepartures((prev) => {
      if (prev.some((d) => d.key === key)) return prev;
      return [
        ...prev,
        {
          key,
          playerName: selectedDeparturePlayer,
          freedCapPct: Number(found.cap_pct || 0),
        },
      ];
    });
    setSelectedDeparturePlayer('');
  }, [departuresOn, selectedDeparturePlayer, teamRoster, analysisYear]);
  const handleRemoveDeparture = useCallback((key) => {
    setClassDepartures((prev) => prev.filter((d) => d.key !== key));
  }, []);
  const classSearchOptions = useMemo(
    () => playerDirectory.map((p) => p.player),
    [playerDirectory]
  );
  const handleGoToClassPlayer = useCallback(() => {
    if (!classSearchPlayer) return;
    const hit = playerDirectory.find((p) => p.player === classSearchPlayer);
    if (!hit) return;
    onSwitchPosition(hit.position_key, hit.player);
    setShowClassDialog(false);
  }, [classSearchPlayer, playerDirectory, onSwitchPosition]);
  const handleQuickEvaluateAndAdd = useCallback(async () => {
    if (!classSearchPlayer) {
      setError('Pick a player first.');
      return;
    }
    const ask = Number(classQuickAsk);
    if (!Number.isFinite(ask) || ask <= 0) {
      setError('Enter a valid class contract AAV.');
      return;
    }
    const years = Number(classQuickYears);
    if (!Number.isFinite(years) || years < 1 || years > 7) {
      setError('Contract years must be 1-7.');
      return;
    }
    const hit = playerDirectory.find((p) => p.player === classSearchPlayer);
    if (!hit) {
      setError('Could not resolve selected player position.');
      return;
    }
    if (!selectedTeam) {
      setError('Select a team before adding to free agency class.');
      return;
    }
    const posCfg = POSITION_FREE_AGENCY[hit.position_key];
    if (!posCfg) {
      setError('Unsupported player position for evaluation.');
      return;
    }
    setClassQuickLoading(true);
    try {
      const body = {
        player_name: classSearchPlayer,
        salary_ask: ask,
        contract_years: years,
        analysis_year: analysisYear,
        team: selectedTeam,
      };
      const capValM = parseFloat(capOverride);
      if (!isNaN(capValM) && capValM > 0) {
        body.cap_available_pct = (capValM / SALARY_CAP_M) * 100;
      }
      const resp = await fetch(`http://127.0.0.1:${posCfg.port}/evaluate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!resp.ok) {
        let detail = 'Class quick evaluation failed.';
        try {
          const err = await resp.json();
          detail = err.detail || detail;
        } catch { /* ignore */ }
        throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
      }
      const result = await resp.json();
      const structured = buildStructuredFreeAgent(result, ask, years, hit.position_key);
      handleAddToClass(structured);
      setMessages((prev) => [
        ...prev,
        { role: 'user', content: `Evaluate ${classSearchPlayer} — $${ask}M/yr × ${years} yr contract as ${selectedTeam}.` },
        { role: 'assistant', content: null, structured },
      ]);
      setClassSearchPlayer('');
      setClassQuickAsk('');
    } catch (e) {
      setError(e.message || 'Class quick evaluation failed.');
    } finally {
      setClassQuickLoading(false);
    }
  }, [
    classSearchPlayer, classQuickAsk, classQuickYears, playerDirectory,
    selectedTeam, analysisYear, capOverride, handleAddToClass,
  ]);

  return (
    <div className="fa-evaluator-wrap">
      <div className="fa-top-pos-bar">
        {POSITIONS.map((pos) => (
          <button
            key={pos.label}
            type="button"
            className={`fa-top-pos-btn ${pos.label === positionKey ? 'active' : ''}`}
            onClick={() => onSwitchPosition(pos.label)}
          >
            {pos.label}
          </button>
        ))}
      </div>
    <div className="fa-page">
      <div className="fa-panel">
        <button type="button" className="fa-back-btn" onClick={onBack}>← Change Position</button>
        <h2 className="fa-panel-title">{cfg.panelTitle}</h2>

        <div className="fa-team-toggle">
          <label className="fa-toggle-label">
            <input type="checkbox" checked={teamMode} onChange={(e) => setTeamMode(e.target.checked)} />
            <span className="fa-toggle-slider" />
            <span className="fa-toggle-text">Team Simulation Mode</span>
          </label>
        </div>

        <div className="fa-field">
          <label className="fa-label">Analysis Year</label>
          <select
            className="fa-select"
            value={analysisYear}
            onChange={(e) => {
              const y = Number(e.target.value);
              setAnalysisYear(Number.isFinite(y) ? y : latestAnalysisYear);
            }}
          >
            {analysisYearOptions.map((y) => (
              <option key={y} value={y}>{y}</option>
            ))}
          </select>
        </div>

        {teamMode && (
          <div className="fa-team-section">
            <div className="fa-field">
              <label className="fa-label">Select Team</label>
              <SearchableSelect
                options={teams}
                value={selectedTeam}
                onChange={handleTeamChange}
                placeholder="Search teams…"
              />
            </div>

            {fetchingTeam && <p className="fa-hint">Loading team data…</p>}

            {!!selectedTeam && (
              <>
                <button
                  type="button"
                  className="fa-btn"
                  onClick={handleTeamYearSummary}
                  disabled={summarizingTeam}
                >
                  {summarizingTeam ? 'Running Team Evaluation…' : `Smart Team Evaluation (${analysisYear})`}
                </button>
                <label className="fa-toggle-label">
                  <input type="checkbox" checked={classBuilderOn} onChange={(e) => setClassBuilderOn(e.target.checked)} />
                  <span className="fa-toggle-slider" />
                  <span className="fa-toggle-text">Free Agency Class Builder</span>
                </label>
                <button
                  type="button"
                  className="fa-btn"
                  onClick={() => {
                    if (!classBuilderOn) setClassBuilderOn(true);
                    if (classStartCapPct == null) {
                      const capPct = !isNaN(parseFloat(capOverride)) && parseFloat(capOverride) >= 0
                        ? (parseFloat(capOverride) / SALARY_CAP_M) * 100
                        : Number(teamRoster?.available_cap_pct || 0);
                      setClassStartCapPct(capPct);
                      setClassStartCapInput((capPct || 0).toFixed(1));
                    }
                    setShowClassDialog(true);
                  }}
                >
                  {classBuilderOn
                    ? `Open Free Agency Class (${classSignings.length})`
                    : 'Enable + Open Free Agency Class'}
                </button>
                {classBuilderOn && (
                  <p className="fa-hint">
                    Analyze players, then click <strong>Add this signing to class</strong> in each result card.
                  </p>
                )}
              </>
            )}

            {teamRoster && !fetchingTeam && (
              <>
                <RosterPreview
                  roster={teamRoster.roster}
                  needLabel={teamRoster.need_label}
                  needScore={teamRoster.positional_need}
                  allocatedPct={teamRoster.allocated_cap_pct}
                  availablePct={
                    !isNaN(parseFloat(capOverride)) && parseFloat(capOverride) >= 0
                      ? (parseFloat(capOverride) / SALARY_CAP_M) * 100
                      : teamRoster.available_cap_pct
                  }
                  positionLabel={cfg.positionLabel}
                />
                <div className="fa-field">
                  <label className="fa-label">Available Cap ($M, editable)</label>
                  <input
                    type="number"
                    min="0"
                    step="0.1"
                    className="fa-input fa-cap-override-input"
                    value={capOverride}
                    onChange={(e) => handleCapOverrideChange(e.target.value)}
                  />
                </div>
              </>
            )}
          </div>
        )}

        <div className="fa-field">
          <label className="fa-label">Player</label>
          {fetchingPlayers ? (
            <p className="fa-hint">Loading players…</p>
          ) : (
            <SearchableSelect
              options={players}
              value={selectedPlayer}
              onChange={setSelectedPlayer}
              placeholder="Search players…"
            />
          )}
        </div>

        <div className="fa-field">
          <label className="fa-label">Contract AAV ($M / yr)</label>
          <div className="fa-price-row">
            <span className="fa-dollar">$</span>
            <input
              type="number"
              min="0"
              step="0.5"
              className="fa-input"
              placeholder="e.g. 18.5"
              value={salaryAsk}
              onChange={(e) => setSalaryAsk(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleAnalyze()}
            />
            <span className="fa-million">M</span>
          </div>
        </div>

        <div className="fa-field">
          <label className="fa-label">Contract Length — {contractYears} yr</label>
          <input
            type="range"
            min="1"
            max={contractMax}
            step="1"
            className="fa-slider"
            value={contractYears}
            onChange={(e) => setContractYears(Number(e.target.value))}
          />
          <div className="fa-slider-ticks">
            {ticks.map((n) => (
              <span
                key={n}
                className={n === contractYears ? 'fa-tick fa-tick--active' : 'fa-tick'}
                onClick={() => setContractYears(n)}
              >
                {n}
              </span>
            ))}
          </div>
        </div>

        {error && <p className="fa-error">{error}</p>}

        <button
          type="button"
          className="fa-btn"
          onClick={handleAnalyze}
          disabled={loading || fetchingPlayers || !selectedPlayer}
        >
          {loading ? 'Analyzing…' : teamMode ? 'Analyze as Team' : 'Analyze Player'}
        </button>

        <div className="fa-legend">
          <p className="fa-legend-title">{cfg.legend.title}</p>
          {cfg.legend.tiers.map((t) => (
            <div key={t.cls} className="fa-legend-row">
              <span className={`tier-badge ${t.cls}`}>{t.label}</span>
              <span>{t.range}</span>
            </div>
          ))}
          <p className="fa-legend-note" dangerouslySetInnerHTML={{ __html: cfg.legend.note }} />
        </div>

        <DecisionTierLegend teamMode={teamMode} />
      </div>

      <div className="fa-chat">
        <div className="fa-chat-header">
          GM Decision Feed — {cfg.chatTitle}
          {teamMode && selectedTeam && <span className="fa-chat-team-tag">{selectedTeam}</span>}
        </div>
        <div className="fa-chat-body">
          {messages.map((msg, i) => (
            <div key={i} className={`fa-msg fa-msg--${msg.role}`}>
              <div className="fa-msg-label">{msg.role === 'user' ? 'You' : 'GM Agent'}</div>

              {msg.content != null ? (
                <div>
                  <div className="fa-msg-text">{msg.content}</div>
                  {msg.teamSummaryMeta && (
                    <button
                      type="button"
                      className="fa-summary-link-btn"
                      onClick={() => handleOpenRankingsDialog(msg.teamSummaryMeta.team, msg.teamSummaryMeta.analysisYear)}
                    >
                      Click for a more detailed view
                    </button>
                  )}
                </div>
              ) : (
                <div className="fa-msg-card">
                  <div className={`fa-decision-badge ${msg.structured.highlight}`}>
                    {msg.structured.decision}
                  </div>
                  {classBuilderOn && msg.structured?.meta?.team && (
                    <button
                      type="button"
                      className="fa-summary-link-btn"
                      onClick={() => handleAddToClass(msg.structured)}
                    >
                      Add this signing to class
                    </button>
                  )}

                  <SigningGrade grade={msg.structured.signing_grade} />

                  {msg.structured.tier_description && (
                    <div className="fa-tier-desc">{msg.structured.tier_description}</div>
                  )}

                  <div className="fa-stats-grid">
                    {msg.structured.stats.map((s, j) =>
                      s.divider ? (
                        <div key={j} className="fa-stat-section-hdr">{s.title}</div>
                      ) : (
                        <div key={j} className="fa-stat-row">
                          <span className="fa-stat-label">{s.label}</span>
                          <span className="fa-stat-value">
                            {s.tierBadgeClass ? (
                              <span className={`tier-badge ${s.tierBadgeClass} fa-projected-tier-badge`}>
                                {s.value}
                              </span>
                            ) : (
                              s.value
                            )}
                          </span>
                        </div>
                      )
                    )}
                  </div>

                  {msg.structured.team_context && (
                    <TeamFitSection
                      teamCtx={msg.structured.team_context}
                      signingPcts={msg.structured.team_context.signing_cap_pcts}
                      positionLabel={cfg.positionLabel}
                    />
                  )}

                  {msg.structured.projected_stats?.length > 0 && (
                    <div className="fa-stats-toggle-row">
                      <button type="button" className="fa-stats-toggle-btn" onClick={() => toggleStats(i)}>
                        {statsOpen[i] ? '▲ Hide Stats Projection' : '▼ View Stats Projection'}
                      </button>
                    </div>
                  )}

                  {statsOpen[i] && msg.structured.career_stats?.length > 0 && (
                    <PositionStatsPanel
                      careerStats={msg.structured.career_stats}
                      projectedStats={msg.structured.projected_stats}
                      statRows={cfg.statRows}
                      note={cfg.statsNote}
                    />
                  )}

                  {msg.structured.year_breakdown?.length > 0 && (
                    <YearBreakdown rows={msg.structured.year_breakdown} />
                  )}

                  <div className="fa-reasoning">
                    <p className="fa-reasoning-title">Reasoning</p>
                    <p className="fa-reasoning-text">{msg.structured.reasoning}</p>
                  </div>
                </div>
              )}
            </div>
          ))}

          {loading && (
            <div className="fa-msg fa-msg--assistant">
              <div className="fa-msg-label">GM Agent</div>
              <div className="fa-typing"><span /><span /><span /></div>
            </div>
          )}

          <div ref={chatEndRef} />
        </div>
      </div>

      {showRankingsDialog && (
        <div className="fa-rankings-overlay" onClick={() => setShowRankingsDialog(false)}>
          <div className="fa-rankings-dialog" onClick={(e) => e.stopPropagation()}>
            <div className="fa-rankings-header">
              <h3>{selectedTeam} Position Power Rankings ({analysisYear})</h3>
              <button type="button" className="fa-back-btn" onClick={() => setShowRankingsDialog(false)}>Close</button>
            </div>
            {loadingRankings ? (
              <p className="fa-hint">Loading rankings…</p>
            ) : (
              <div>
                <div className="fa-rankings-legend">
                  <span className="fa-rank-chip top">Top 12</span>
                  <span className="fa-rank-chip mid">13-22</span>
                  <span className="fa-rank-chip low">23-32</span>
                </div>
                <div className="fa-rankings-grid">
                  {sortedTeamRankings.map((r) => {
                    const rank = Number(r.rank);
                    const cls = rank <= 12 ? 'top' : rank <= 22 ? 'mid' : 'low';
                    const widthPct = Math.max(4, Math.min(100, ((33 - rank) / 32) * 100));
                    return (
                      <div key={r.position_key} className="fa-ranking-card">
                        <div className="fa-ranking-row">
                          <span className="fa-ranking-pos">{r.position_key}</span>
                          <span className={`fa-ranking-badge ${cls}`}>#{rank}</span>
                        </div>
                        <div className="fa-ranking-bar-track">
                          <div className={`fa-ranking-bar-fill ${cls}`} style={{ width: `${widthPct}%` }} />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
      {showClassDialog && (
        <div className="fa-rankings-overlay" onClick={() => setShowClassDialog(false)}>
          <div className="fa-rankings-dialog" onClick={(e) => e.stopPropagation()}>
            <div className="fa-rankings-header">
              <h3>Free Agency Class Grade</h3>
              <button type="button" className="fa-back-btn" onClick={() => setShowClassDialog(false)}>Close</button>
            </div>
            {!classGradeSummary ? (
              <div>
                <p className="fa-hint">No signings added yet.</p>
                <div className="fa-field" style={{ marginTop: 8 }}>
                  <label className="fa-label">Find player by name (all positions)</label>
                  <SearchableSelect
                    options={classSearchOptions}
                    value={classSearchPlayer}
                    onChange={setClassSearchPlayer}
                    placeholder="Search players..."
                  />
                </div>
                <button type="button" className="fa-btn" onClick={handleGoToClassPlayer} disabled={!classSearchPlayer}>
                  Go to Player Evaluation
                </button>
              </div>
            ) : (
              <div>
                <div className="fa-field" style={{ marginBottom: 10 }}>
                  <label className="fa-label">Starting Cap for Class (%)</label>
                  <div className="fa-price-row">
                    <input
                      type="number"
                      min="0"
                      step="0.1"
                      className="fa-input"
                      value={classStartCapInput}
                      disabled={classCapLocked}
                      onChange={(e) => setClassStartCapInput(e.target.value)}
                    />
                    <span className="fa-million">%</span>
                  </div>
                  <div style={{ marginTop: 8, display: 'flex', gap: 8 }}>
                    <button
                      type="button"
                      className="fa-btn"
                      onClick={() => {
                        const v = Number(classStartCapInput);
                        if (Number.isFinite(v) && v >= 0) {
                          setClassStartCapPct(v);
                          setClassCapLocked(true);
                        }
                      }}
                      disabled={classCapLocked}
                    >
                      {classCapLocked ? 'Starting Cap Locked' : 'Lock Starting Cap'}
                    </button>
                    {classCapLocked && (
                      <button
                        type="button"
                        className="fa-btn"
                        onClick={() => setClassCapLocked(false)}
                      >
                        Unlock
                      </button>
                    )}
                  </div>
                </div>
                <p className="fa-msg-text">
                  Class Grade: <strong>{classGradeSummary.grade} ({classGradeSummary.letter})</strong>
                </p>
                <p className="fa-msg-text">
                  Cap Used (Yr 1): <strong>{classUsedCapPct.toFixed(1)}%</strong>
                  {departuresOn && (
                    <>
                      {' '}· Cap Freed: <strong style={{ color: '#3de87a' }}>{classFreedCapPct.toFixed(1)}%</strong>
                      {' '}· Net Used: <strong>{classNetCapPct.toFixed(1)}%</strong>
                    </>
                  )}
                  {' '}· Remaining:{' '}
                  <strong style={{ color: classRemainingCapPct != null && classRemainingCapPct < 0 ? '#e05555' : '#3de87a' }}>
                    {classRemainingCapPct == null ? 'N/A' : `${classRemainingCapPct.toFixed(1)}%`}
                  </strong>
                </p>
                <label className="fa-toggle-label" style={{ marginBottom: 8 }}>
                  <input type="checkbox" checked={departuresOn} onChange={(e) => setDeparturesOn(e.target.checked)} />
                  <span className="fa-toggle-slider" />
                  <span className="fa-toggle-text">Account for Departures</span>
                </label>
                {departuresOn && (
                  <div style={{ marginBottom: 10 }}>
                    <div className="fa-field">
                      <label className="fa-label">Add Departure (search roster)</label>
                      <SearchableSelect
                        options={rosterDepartureOptions}
                        value={selectedDeparturePlayer}
                        onChange={setSelectedDeparturePlayer}
                        placeholder="Search roster players..."
                      />
                    </div>
                    <button type="button" className="fa-btn" onClick={handleAddDeparture} disabled={!selectedDeparturePlayer}>
                      Add Departure
                    </button>
                    {!!classDepartures.length && (
                      <div className="fa-rankings-grid" style={{ marginTop: 8 }}>
                        {classDepartures.map((d) => (
                          <div key={d.key} className="fa-ranking-card">
                            <div className="fa-ranking-row">
                              <span className="fa-ranking-pos">{d.playerName}</span>
                              <span className="fa-ranking-badge top">+{Number(d.freedCapPct).toFixed(1)}%</span>
                            </div>
                            <button type="button" className="fa-summary-link-btn" onClick={() => handleRemoveDeparture(d.key)}>
                              Remove departure
                            </button>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
                <div className="fa-field" style={{ marginBottom: 10 }}>
                  <label className="fa-label">Find player by name (all positions)</label>
                  <SearchableSelect
                    options={classSearchOptions}
                    value={classSearchPlayer}
                    onChange={setClassSearchPlayer}
                    placeholder="Search players..."
                  />
                </div>
                <button type="button" className="fa-btn" onClick={handleGoToClassPlayer} disabled={!classSearchPlayer}>
                  Go to Player Evaluation
                </button>
                <div className="fa-rankings-grid">
                  {classGradeSummary.rows.map((r) => (
                    <div key={r.key} className="fa-ranking-card">
                      <div className="fa-ranking-row">
                        <span className="fa-ranking-pos">{r.positionKey} — {r.playerName}</span>
                        <span className="fa-ranking-badge mid">{Math.round(r.signingGrade)}</span>
                      </div>
                      <div className="fa-hint">
                        ${r.ask}M x {r.years}y · Yr1 cap {Number(r.yr1CapPct || 0).toFixed(1)}% · weight {r.weight.toFixed(2)}
                      </div>
                    </div>
                  ))}
                </div>
                <div style={{ marginTop: 10 }}>
                  <button type="button" className="fa-btn" onClick={handleClearClass}>Clear Class</button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
    </div>
  );
}

/* ─── Root ─── */
function FreeAgency() {
  const [selectedPosition, setSelectedPosition] = useState(null);
  const [pendingPick, setPendingPick] = useState(null);

  const handleSwitchPosition = useCallback((positionKey, playerName = '') => {
    setSelectedPosition(positionKey);
    if (playerName) {
      setPendingPick({ positionKey, playerName });
    }
  }, []);

  if (!selectedPosition) {
    return <PositionPicker onSelect={(k) => handleSwitchPosition(k)} />;
  }

  return (
    <PositionEvaluator
      positionKey={selectedPosition}
      onBack={() => setSelectedPosition(null)}
      onSwitchPosition={handleSwitchPosition}
      pendingPick={pendingPick}
      clearPendingPick={() => setPendingPick(null)}
    />
  );
}

export default FreeAgency;
