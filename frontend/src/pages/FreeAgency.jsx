import React, { useState, useEffect, useRef, useCallback, useMemo, useId } from 'react';
import './FreeAgency.css';
import {
  POSITION_FREE_AGENCY,
  FA_POSITION_ORDER,
  FA_VALUE_ANCHORS,
  NOTE_STD,
  fairAavTierBandsForAllPositions,
  gradeToMarketAav,
} from '../config/freeAgencyPositionConfig';

const UNIFIED_WELCOME =
  'Welcome to the Free Agency Assistant. Search any player (all positions), set contract AAV and length, then Analyze. Position is detected from your player pick and shown on each result.';

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

  // Prevent tiny-dollar deals from auto-scoring in the 90s/100s.
  // Even cheap deals consume cap and should not grade like elite-impact signings.
  const lowValuePenalty = _lerp(
    [0,   1,   2,   3,   5,   8,   12,  20],
    [14, 12,  10,   8,   5,   3,    1,   0],
    Number(fair_aav) || 0
  );
  const capFootprintPenalty = _lerp(
    [0,   0.5, 1,   2,   3,   5,   8,   12],
    [14, 12,  10,   8,   6,   3,   1,    0],
    Number(cap_burden) || 0
  );
  base = base - (0.7 * lowValuePenalty + 0.6 * capFootprintPenalty);

  if (teamCtx && teamCtx.need_label) {
    const strength = teamCtx.positional_need || 50;
    const teamAdj = (50 - strength) * 0.24;

    const yr1Pct = (teamCtx.signing_cap_pcts || [])[0] || 0;
    const availPct = teamCtx.available_cap_pct || 100;
    const capRatio = yr1Pct / Math.max(availPct, 0.01);

    // Cap flexibility adjustment — continuous scale:
    //   capRatio ~0.10 (tiny % of cap) →  0 pts
    //   capRatio ~0.25                  →  0 pts
    //   capRatio ~0.35                  → -2 pts
    //   capRatio ~0.50                  → -5 pts  (eating half the remaining cap)
    //   capRatio ~0.75                  → -12 pts (almost no room left after)
    //   capRatio ~1.00+                 → -18 pts (barely fits or exceeds cap)
    const capAdj = _lerp(
      [0.0,  0.10, 0.25, 0.35, 0.50, 0.75, 1.0],
      [0,    0,    0,   -2,    -5,   -12,  -18],
      capRatio
    );

    // Absolute cap room adjustment (no upside bonus for "cheap" signings):
    //   availPct >= 25% →  0 pts
    //   availPct ~15%   →  0 pts
    //   availPct ~10%   → -1 pts
    //   availPct <= 5%  → -4 pts (cap-strapped, every dollar matters)
    const roomAdj = _lerp(
      [2,   5,   10,  15,  25],
      [-4,  -3,  -1,   0,   0],
      availPct
    );

    base = base + teamAdj + capAdj + roomAdj;
  }

  // Hard ceiling by player value tier: low-value deals should not hit top-end grades.
  const valueCeiling = _lerp(
    [0,  2,  4,  6,  8,  12, 20, 35],
    [70, 74, 78, 82, 86, 90, 95, 100],
    Number(fair_aav) || 0
  );
  base = Math.min(base, valueCeiling);

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

/** Compact ring for FA class summary: deals-only vs roster net (supports empty signing class). */
function ClassMetricRing({ title, grade, letter, subtitle, emptyHint, variant = 'signing' }) {
  const gid = useId().replace(/:/g, '');
  const empty = grade == null || Number.isNaN(Number(grade));
  const g = empty ? 0 : Math.round(Number(grade));
  const pct = empty ? 0 : Math.max(0, Math.min(100, g));
  const color = empty ? '#5c5c5c' : gradeColor(g);
  const R = 34;
  const arc = 2 * Math.PI * R * pct / 100;
  const rest = 2 * Math.PI * R * (1 - pct / 100);
  const gradId = `fa-ring-grad-${variant}-${gid}`;
  return (
    <div className={`fa-class-metric-card fa-class-metric-card--${variant}${empty ? ' fa-class-metric-card--empty' : ''}`}>
      <div className="fa-class-metric-card-inner">
        <span className="fa-class-metric-title">{title}</span>
        <svg className="fa-class-metric-ring" viewBox="0 0 80 80" aria-hidden>
          <defs>
            <linearGradient id={gradId} x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor={empty ? '#444' : color} stopOpacity="1" />
              <stop offset="100%" stopColor={empty ? '#333' : color} stopOpacity="0.75" />
            </linearGradient>
          </defs>
          <circle cx="40" cy="40" r={R} fill="none" stroke="#1a1a1f" strokeWidth="6" />
          <circle
            cx="40"
            cy="40"
            r={R}
            fill="none"
            stroke={empty ? '#3a3a42' : `url(#${gradId})`}
            strokeWidth="6"
            strokeDasharray={empty ? `${2 * Math.PI * R * 0.08} ${2 * Math.PI * R * 0.92}` : `${arc} ${rest}`}
            strokeDashoffset={2 * Math.PI * R * 0.25}
            strokeLinecap="round"
            className={empty ? '' : 'fa-class-metric-ring-arc'}
          />
          <text x="40" y="38" textAnchor="middle" fill={empty ? '#777' : color} fontSize="17" fontWeight="bold">
            {empty ? '—' : g}
          </text>
          <text x="40" y="52" textAnchor="middle" fill={empty ? '#666' : color} fontSize="11">
            {empty ? '—' : letter}
          </text>
        </svg>
        {subtitle && <span className="fa-class-metric-sub">{subtitle}</span>}
        {empty && emptyHint && <span className="fa-class-metric-hint">{emptyHint}</span>}
      </div>
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
function pctToDollarsNum(pct) { return Number(((pct / 100) * SALARY_CAP_M).toFixed(1)); }
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

function PositionEvaluator({ positionKey, pendingPick, clearPendingPick }) {
  const [resolvedPositionKey, setResolvedPositionKey] = useState(positionKey);
  const cfg = POSITION_FREE_AGENCY[resolvedPositionKey];
  const apiBase = `http://127.0.0.1:${cfg.port}`;
  const directoryApiBase = `http://127.0.0.1:${POSITION_FREE_AGENCY[FA_POSITION_ORDER[0]].port}`;
  const contractMax = 7;
  const latestAnalysisYear = 2025;

  const [selectedPlayer, setSelectedPlayer] = useState('');
  const [salaryAsk, setSalaryAsk] = useState('');
  const [contractYears, setContractYears] = useState(1);
  const [loading, setLoading] = useState(false);
  const [fetchingPlayers, setFetchingPlayers] = useState(true);
  const [messages, setMessages] = useState([{ role: 'assistant', content: UNIFIED_WELCOME }]);
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
  const [fullTeamRosterForDepartures, setFullTeamRosterForDepartures] = useState([]);
  const [selectedDeparturePlayer, setSelectedDeparturePlayer] = useState('');
  const [playerDirectory, setPlayerDirectory] = useState([]);
  const [classSearchPlayer, setClassSearchPlayer] = useState('');
  const [classQuickAsk, setClassQuickAsk] = useState('');
  const [classQuickYears, setClassQuickYears] = useState(3);
  const [classQuickLoading, setClassQuickLoading] = useState(false);
  const [showSigningClassDialog, setShowSigningClassDialog] = useState(false);
  const [showRosterNetDialog, setShowRosterNetDialog] = useState(false);
  const [analysisYear, setAnalysisYear] = useState(latestAnalysisYear);
  const [analysisYearMin, setAnalysisYearMin] = useState(2010);

  const chatEndRef = useRef(null);
  const messageRefs = useRef({});

  const toggleStats = useCallback((i) => {
    setStatsOpen((prev) => ({ ...prev, [i]: !prev[i] }));
  }, []);

  useEffect(() => {
    setResolvedPositionKey(positionKey);
  }, [positionKey]);

  const directoryCfg = POSITION_FREE_AGENCY[FA_POSITION_ORDER[0]];

  useEffect(() => {
    setFetchingPlayers(true);
    setError('');
    fetch(`${directoryApiBase}${directoryCfg.playersPath}`)
      .then((r) => r.json())
      .then((data) => {
        const minYr = Number(data.analysis_year_min);
        setAnalysisYearMin(Number.isFinite(minYr) ? minYr : 2010);
      })
      .catch(() =>
        setError(`Could not load player list. Start the ${directoryCfg.chatTitle} API (port ${directoryCfg.port}): uvicorn backend.agent…`)
      )
      .finally(() => setFetchingPlayers(false));

    fetch(`${directoryApiBase}/player-directory?analysis_year=${encodeURIComponent(analysisYear)}`)
      .then((r) => r.json())
      .then((data) => {
        setPlayerDirectory(data.players || []);
      })
      .catch(() => setPlayerDirectory([]));

    fetch(`${directoryApiBase}/teams?analysis_year=${encodeURIComponent(analysisYear)}`)
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
  }, [directoryApiBase, directoryCfg.playersPath, directoryCfg.port, directoryCfg.chatTitle, analysisYear]);

  useEffect(() => {
    if (!pendingPick) return;
    if (pendingPick.playerName) setSelectedPlayer(pendingPick.playerName);
    if (pendingPick.positionKey) setResolvedPositionKey(pendingPick.positionKey);
    if (clearPendingPick) clearPendingPick();
  }, [pendingPick, clearPendingPick]);

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
    if (!selectedTeam || !teamMode) {
      setFullTeamRosterForDepartures([]);
      return;
    }
    const host = window.location.hostname || 'localhost';
    const posEntries = Object.entries(POSITION_FREE_AGENCY);
    const reqs = posEntries.map(([, pcfg]) =>
      fetch(`http://${host}:${pcfg.port}/team-roster?team=${encodeURIComponent(selectedTeam)}&analysis_year=${encodeURIComponent(analysisYear)}`)
        .then((r) => (r.ok ? r.json() : null))
        .catch(() => null)
    );
    Promise.all(reqs)
      .then((all) => {
        const merged = new Map();
        all.forEach((res, idx) => {
          const positionKey = posEntries[idx][0];
          const roster = Array.isArray(res?.roster) ? res.roster : [];
          roster.forEach((p) => {
            const name = String(p?.player || '').trim();
            if (!name) return;
            const key = name.toLowerCase();
            const capPct = Number(p?.cap_pct || 0);
            const grade = Number(p?.grade || 0);
            const snaps = Number(p?.snaps || 0);
            const prev = merged.get(key);
            if (!prev || capPct > Number(prev.cap_pct || 0)) {
              merged.set(key, { player: name, cap_pct: capPct, grade, snaps, position_key: positionKey });
            }
          });
        });
        setFullTeamRosterForDepartures(
          Array.from(merged.values()).sort((a, b) => a.player.localeCompare(b.player))
        );
      })
      .catch(() => setFullTeamRosterForDepartures([]));
  }, [selectedTeam, teamMode, analysisYear]);

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
        content: `Evaluate ${selectedPlayer} (${resolvedPositionKey}) — $${ask}M/yr × ${contractYears} yr contract${teamLabel}.`,
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
          structured: buildStructuredFreeAgent(result, ask, contractYears, resolvedPositionKey),
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
  const allPositionsTierBands = useMemo(() => fairAavTierBandsForAllPositions(), []);
  const playerPickOptions = useMemo(
    () => [...new Set(
      playerDirectory.map((p) => `${p.player} (${p.position_key})`)
    )].sort((a, b) => a.localeCompare(b)),
    [playerDirectory]
  );
  const selectedPlayerPickValue = useMemo(() => {
    if (!selectedPlayer || !resolvedPositionKey) return selectedPlayer || '';
    return `${selectedPlayer} (${resolvedPositionKey})`;
  }, [selectedPlayer, resolvedPositionKey]);
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

  /** Signings-only grade is in classGradeSummary. Net roster score penalizes losing higher-grade / higher-cap departures (weighted by position). */
  const classRosterNetSummary = useMemo(() => {
    const POS_IMPORTANCE = {
      QB: 1.45, ED: 1.25, WR: 1.18, CB: 1.15, T: 1.12, DI: 1.08,
      C: 1.03, G: 1.02, LB: 1.0, S: 0.98, TE: 0.95, HB: 0.9,
    };
    const hasSignings = classSignings.length > 0;
    const hasDepartures = departuresOn && classDepartures.length > 0;
    if (!hasSignings && !hasDepartures) return null;
    if (!hasDepartures && hasSignings) {
      if (!classGradeSummary) return null;
      const sg = classGradeSummary.grade;
      return {
        grade: sg,
        letter: gradeLetter(sg),
        hasSignings: true,
        signingGrade: sg,
        hasDepartures: false,
        avgDepartureGrade: null,
        lossPenalty: 0,
        unfilledReplacementPenalty: 0,
        coverageGapPenalty: 0,
        signingEmphasis: 1,
        adjustedSigningBaseline: sg,
        replacementCoveragePct: null,
        explanation:
          'Roster net equals signing class until you model departures. Turn on “Account for Departures” and add players to estimate talent (and cap) walking out the door.',
      };
    }

    let lossNum = 0;
    let lossDen = 0;
    classDepartures.forEach((d) => {
      const g = Number(d.grade) || 60;
      const posW = POS_IMPORTANCE[d.positionKey] || 1;
      const capW = Math.max(0.5, Math.min(1.8, 0.5 + Number(d.freedCapPct || 0) / 25));
      const w = posW * capW;
      lossNum += g * w;
      lossDen += w;
    });
    const avgDep = lossDen > 0 ? lossNum / lossDen : 0;
    const lossPenalty = Math.max(0, (avgDep - 58) * 0.55);
    const stressFromPenalty = Math.min(1, lossPenalty / 24);
    const stressFromCount = Math.min(1, classDepartures.length / 7);
    const departureStress = Math.min(1, 0.55 * stressFromPenalty + 0.45 * stressFromCount);

    let signingGrade = 0;
    let sigW = 0;
    if (hasSignings && classGradeSummary) {
      signingGrade = classGradeSummary.grade;
      sigW = classGradeSummary.rows.reduce((acc, r) => acc + (Number(r.weight) || 0), 0);
    }
    const signingEmphasis = hasSignings ? 1 + 0.12 * departureStress : 1;
    const adjustedSigningBaseline = hasSignings ? Math.min(100, signingGrade * signingEmphasis) : 0;

    const replacementCoveragePct =
      lossDen > 0 ? Math.round(Math.min(150, (sigW / lossDen) * 100)) : 100;

    let unfilledReplacementPenalty = 0;
    let coverageGapPenalty = 0;
    if (hasDepartures && !hasSignings) {
      unfilledReplacementPenalty = 14 + 0.5 * lossPenalty + 2.5 * Math.min(classDepartures.length, 12);
    } else if (hasDepartures && hasSignings && lossDen > 0) {
      const cov = Math.min(1.25, sigW / lossDen);
      const shortfall = Math.max(0, 1 - cov);
      coverageGapPenalty = Math.min(26, shortfall * 28 * (0.55 + 0.45 * departureStress));
    }

    let net =
      adjustedSigningBaseline -
      lossPenalty -
      unfilledReplacementPenalty -
      coverageGapPenalty;
    net = Math.round(Math.max(0, Math.min(100, net)));

    const round1 = (x) => Math.round(x * 10) / 10;
    const parts = [];
    if (hasSignings) {
      parts.push(
        `Signing class ${signingGrade}/100 is scaled to ~${round1(adjustedSigningBaseline)} (×${Math.round(signingEmphasis * 1000) / 1000}) under departure stress.`
      );
    } else {
      parts.push('No free-agent signings in this class — net is driven by talent walking out and unfilled roster holes.');
    }
    parts.push(`Weighted departures average ~${Math.round(avgDep)} grade; talent-out penalty: −${round1(lossPenalty)}.`);
    if (unfilledReplacementPenalty > 0) {
      parts.push(`Unfilled replacement penalty: −${round1(unfilledReplacementPenalty)} (no signings to offset losses).`);
    }
    if (coverageGapPenalty > 0) {
      parts.push(
        `Replacement coverage ~${Math.min(100, replacementCoveragePct)}% of weighted departure mass; coverage gap: −${round1(coverageGapPenalty)}.`
      );
    }

    return {
      grade: net,
      letter: gradeLetter(net),
      hasSignings,
      signingGrade: hasSignings ? signingGrade : null,
      hasDepartures: true,
      avgDepartureGrade: Math.round(avgDep),
      lossPenalty: round1(lossPenalty),
      unfilledReplacementPenalty: round1(unfilledReplacementPenalty),
      coverageGapPenalty: round1(coverageGapPenalty),
      signingEmphasis: hasSignings ? Math.round(signingEmphasis * 1000) / 1000 : null,
      adjustedSigningBaseline: hasSignings ? round1(adjustedSigningBaseline) : null,
      replacementCoveragePct: lossDen > 0 ? Math.min(100, replacementCoveragePct) : null,
      explanation: parts.join(' '),
    };
  }, [classSignings, classDepartures, departuresOn, classGradeSummary]);
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
    () => fullTeamRosterForDepartures.map((p) => p.player).filter(Boolean),
    [fullTeamRosterForDepartures]
  );
  const classStorageKey = useMemo(() => {
    if (!selectedTeam || !analysisYear) return null;
    return `faClass::${selectedTeam}::${analysisYear}`;
  }, [selectedTeam, analysisYear]);
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
          fullEvaluation: structured,
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
    if (classStorageKey) {
      try { localStorage.removeItem(classStorageKey); } catch { /* ignore */ }
    }
  }, [classStorageKey]);
  const handleAddDeparture = useCallback(() => {
    if (!departuresOn || !selectedDeparturePlayer) return;
    const roster = fullTeamRosterForDepartures || [];
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
          grade: Number(found.grade || 0),
          snaps: Number(found.snaps || 0),
          positionKey: found.position_key || resolvedPositionKey,
        },
      ];
    });
    setSelectedDeparturePlayer('');
  }, [departuresOn, selectedDeparturePlayer, fullTeamRosterForDepartures, analysisYear, resolvedPositionKey]);
  const handleRemoveDeparture = useCallback((key) => {
    setClassDepartures((prev) => prev.filter((d) => d.key !== key));
  }, []);
  const handleOpenClassSigningEvaluation = useCallback(async (row) => {
    if (!row) return;
    setShowClassDialog(false);
    setResolvedPositionKey(row.positionKey);
    setSelectedPlayer(row.playerName);
    const existingIdx = messages.findIndex((m) => {
      const meta = m?.structured?.meta;
      if (!meta) return false;
      return (
        String(meta.playerName || '').toLowerCase() === String(row.playerName || '').toLowerCase() &&
        String(meta.positionKey || '') === String(row.positionKey || '') &&
        String(meta.team || '') === String(row.team || '') &&
        Number(meta.analysisYear || 0) === Number(row.analysisYear || 0)
      );
    });
    if (existingIdx >= 0) {
      setTimeout(() => {
        const el = messageRefs.current[existingIdx];
        if (el?.scrollIntoView) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }, 120);
      return;
    }
    if (row.fullEvaluation) {
      setMessages((prev) => [
        ...prev,
        { role: 'user', content: `Evaluate ${row.playerName} — $${row.ask}M/yr × ${row.years} yr contract as ${row.team}.` },
        { role: 'assistant', content: null, structured: row.fullEvaluation },
      ]);
      return;
    }
    try {
      const cfgForPos = POSITION_FREE_AGENCY[row.positionKey];
      if (!cfgForPos) throw new Error('No API config for that position.');
      const host = window.location.hostname || 'localhost';
      const url = `http://${host}:${cfgForPos.port}/evaluate`;
      const body = {
        player_name: row.playerName,
        salary_ask: Number(row.ask),
        contract_years: Number(row.years),
        analysis_year: Number(row.analysisYear),
        team: row.team,
      };
      const resp = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!resp.ok) throw new Error(`Failed to evaluate ${row.playerName}.`);
      const result = await resp.json();
      const structured = buildStructuredFreeAgent(result, Number(row.ask), Number(row.years), row.positionKey);
      setClassSignings((prev) => prev.map((x) => (x.key === row.key ? { ...x, fullEvaluation: structured } : x)));
      setMessages((prev) => [
        ...prev,
        { role: 'user', content: `Evaluate ${row.playerName} — $${row.ask}M/yr × ${row.years} yr contract as ${row.team}.` },
        { role: 'assistant', content: null, structured },
      ]);
    } catch (e) {
      setError(e.message || `Could not load full evaluation for ${row.playerName}.`);
    }
  }, [messages]);
  const classSearchOptions = useMemo(
    () => playerDirectory.map((p) => p.player),
    [playerDirectory]
  );
  const classGradeBadgeStyle = useCallback((g) => {
    const c = gradeColor(Number(g));
    return {
      color: c,
      borderColor: c,
      background: 'rgba(255, 255, 255, 0.02)',
    };
  }, []);
  const handleGoToClassPlayer = useCallback(() => {
    if (!classSearchPlayer) return;
    const hit = playerDirectory.find((p) => p.player === classSearchPlayer);
    if (!hit) return;
    setResolvedPositionKey(hit.position_key);
    setSelectedPlayer(hit.player);
    setShowClassDialog(false);
  }, [classSearchPlayer, playerDirectory]);
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

  useEffect(() => {
    if (!classStorageKey) return;
    try {
      const raw = localStorage.getItem(classStorageKey);
      if (!raw) {
        setClassSignings([]);
        setClassDepartures([]);
        setClassStartCapPct(null);
        setClassStartCapInput('');
        setClassCapLocked(false);
        setDeparturesOn(false);
        return;
      }
      const data = JSON.parse(raw);
      setClassSignings(Array.isArray(data.classSignings) ? data.classSignings : []);
      setClassDepartures(Array.isArray(data.classDepartures) ? data.classDepartures : []);
      const startPct = Number(data.classStartCapPct);
      if (Number.isFinite(startPct)) {
        setClassStartCapPct(startPct);
        setClassStartCapInput(pctToDollars(startPct));
      } else {
        setClassStartCapPct(null);
        setClassStartCapInput('');
      }
      setClassCapLocked(Boolean(data.classCapLocked));
      setDeparturesOn(Boolean(data.departuresOn));
    } catch {
      setClassSignings([]);
      setClassDepartures([]);
      setClassStartCapPct(null);
      setClassStartCapInput('');
      setClassCapLocked(false);
      setDeparturesOn(false);
    }
  }, [classStorageKey]);

  useEffect(() => {
    if (!classStorageKey) return;
    try {
      localStorage.setItem(
        classStorageKey,
        JSON.stringify({
          classSignings,
          classDepartures,
          classStartCapPct,
          classCapLocked,
          departuresOn,
        })
      );
    } catch {
      // ignore storage errors
    }
  }, [classStorageKey, classSignings, classDepartures, classStartCapPct, classCapLocked, departuresOn]);

  return (
    <div className="fa-evaluator-wrap">
    <div className="fa-page">
      <div className="fa-panel">
        <h2 className="fa-panel-title">Free Agency Assistant</h2>
        <p className="fa-hint" style={{ marginTop: 4 }}>
          Pick any player below — your selection sets position, model, and stat columns for each run.
        </p>

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
                      setClassStartCapInput(pctToDollars(capPct || 0));
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
          <label className="fa-label">Player (all positions)</label>
          {fetchingPlayers ? (
            <p className="fa-hint">Loading player directory…</p>
          ) : (
            <SearchableSelect
              options={playerPickOptions}
              value={selectedPlayerPickValue}
              onChange={(val) => {
                const hit = playerDirectory.find((p) => `${p.player} (${p.position_key})` === val);
                if (hit) {
                  setSelectedPlayer(hit.player);
                  setResolvedPositionKey(hit.position_key);
                }
              }}
              placeholder="Search players…"
            />
          )}
          {!!selectedPlayer && (
            <p className="fa-hint">Evaluating as: <strong>{resolvedPositionKey}</strong> ({cfg.positionLabel})</p>
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
          <p className="fa-legend-title">2026 fair AAV by tier (all positions)</p>
          <div className="fa-legend-table-wrap">
            <table className="fa-legend-mini-table">
              <thead>
                <tr>
                  <th scope="col">Pos</th>
                  <th scope="col" className="fa-legend-th-elite">Elite</th>
                  <th scope="col" className="fa-legend-th-good">Good</th>
                  <th scope="col" className="fa-legend-th-starter">Starter</th>
                  <th scope="col" className="fa-legend-th-rot">Rot / Bkup</th>
                </tr>
              </thead>
              <tbody>
                {allPositionsTierBands.map((row) => (
                  <tr key={row.pos}>
                    <td className="fa-legend-pos">{row.pos}</td>
                    <td>{row.elite}</td>
                    <td>{row.good}</td>
                    <td>{row.starter}</td>
                    <td>{row.rotation}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="fa-legend-note">
            Grade cutoffs match the backend: Elite ≥80, Good ≥74, Starter ≥62; rotation/backup below 62.
            Dollar ranges use each position’s grade→AAV curve (same as the valuation engine).{' '}
            {NOTE_STD}
          </p>
        </div>

        <DecisionTierLegend teamMode={teamMode} />
      </div>

      <div className="fa-chat">
        <div className="fa-chat-header">
          GM Decision Feed
          {teamMode && selectedTeam && <span className="fa-chat-team-tag">{selectedTeam}</span>}
        </div>
        <div className="fa-chat-body">
          {messages.map((msg, i) => {
            const msgPosKey = msg.structured?.meta?.positionKey;
            const msgCfg = (msgPosKey && POSITION_FREE_AGENCY[msgPosKey]) ? POSITION_FREE_AGENCY[msgPosKey] : cfg;
            return (
            <div key={i} ref={(el) => { messageRefs.current[i] = el; }} className={`fa-msg fa-msg--${msg.role}`}>
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
                      positionLabel={msgCfg.positionLabel}
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
                      statRows={msgCfg.statRows}
                      note={msgCfg.statsNote}
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
            );
          })}

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
                <div className="fa-class-grades-row">
                  <ClassMetricRing
                    title="Signing class"
                    subtitle="Incoming deals only"
                    grade={null}
                    letter={null}
                    emptyHint="Evaluate + add players"
                    variant="signing"
                  />
                  <ClassMetricRing
                    title="Roster net"
                    subtitle={
                      classRosterNetSummary?.hasDepartures
                        ? (classRosterNetSummary.hasSignings
                          ? `Coverage ${classRosterNetSummary.replacementCoveragePct ?? 0}% vs losses`
                          : 'Departures with no FA class')
                        : 'Signings vs departures'
                    }
                    grade={classRosterNetSummary?.grade ?? null}
                    letter={classRosterNetSummary?.letter ?? null}
                    emptyHint={!classRosterNetSummary ? 'Turn on departures or add signings' : undefined}
                    variant="net"
                  />
                </div>
                <p className="fa-hint" style={{ marginTop: 10 }}>
                  No signings in this class yet — you can still model <strong>departures</strong> below; roster net will show the cost of talent leaving without replacements.
                </p>
                <label className="fa-toggle-label" style={{ marginBottom: 8, marginTop: 12 }}>
                  <input type="checkbox" checked={departuresOn} onChange={(e) => setDeparturesOn(e.target.checked)} />
                  <span className="fa-toggle-slider" />
                  <span className="fa-toggle-text">Account for Departures</span>
                </label>
                {departuresOn && (
                  <div style={{ marginBottom: 12 }}>
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
                              <span className="fa-ranking-badge top">+${pctToDollarsNum(Number(d.freedCapPct || 0)).toFixed(1)}M</span>
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
                <div className="fa-field" style={{ marginTop: 8 }}>
                  <label className="fa-label">Find player by name (all positions)</label>
                  <SearchableSelect
                    options={classSearchOptions}
                    value={classSearchPlayer}
                    onChange={setClassSearchPlayer}
                    placeholder="Search players..."
                  />
                </div>
                <div className="fa-field" style={{ marginBottom: 8 }}>
                  <label className="fa-label">Class Contract AAV ($M)</label>
                  <div className="fa-price-row">
                    <input
                      type="number"
                      min="0"
                      step="0.5"
                      className="fa-input"
                      value={classQuickAsk}
                      onChange={(e) => setClassQuickAsk(e.target.value)}
                    />
                    <span className="fa-million">M</span>
                  </div>
                </div>
                <div className="fa-field" style={{ marginBottom: 10 }}>
                  <label className="fa-label">Contract Years</label>
                  <select
                    className="fa-select"
                    value={classQuickYears}
                    onChange={(e) => setClassQuickYears(Number(e.target.value))}
                  >
                    {[1, 2, 3, 4, 5, 6, 7].map((y) => (
                      <option key={y} value={y}>{y}</option>
                    ))}
                  </select>
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                  <button
                    type="button"
                    className="fa-btn"
                    onClick={handleQuickEvaluateAndAdd}
                    disabled={classQuickLoading || !classSearchPlayer}
                  >
                    {classQuickLoading ? 'Evaluating…' : 'Evaluate + Add to Class'}
                  </button>
                  <button type="button" className="fa-btn" onClick={handleGoToClassPlayer} disabled={!classSearchPlayer}>
                    Go to Player Page
                  </button>
                  <button
                    type="button"
                    className="fa-btn fa-btn--ghost"
                    onClick={() => setShowRosterNetDialog(true)}
                    disabled={!classRosterNetSummary}
                  >
                    Explain roster net
                  </button>
                </div>
              </div>
            ) : (
              <div>
                <div className="fa-field" style={{ marginBottom: 10 }}>
                  <label className="fa-label">Starting Cap for Class ($M)</label>
                  <div className="fa-price-row">
                    <span className="fa-dollar">$</span>
                    <input
                      type="number"
                      min="0"
                      step="0.1"
                      className="fa-input"
                      value={classStartCapInput}
                      disabled={classCapLocked}
                      onChange={(e) => setClassStartCapInput(e.target.value)}
                    />
                    <span className="fa-million">M</span>
                  </div>
                  <div style={{ marginTop: 8, display: 'flex', gap: 8 }}>
                    <button
                      type="button"
                      className="fa-btn"
                      onClick={() => {
                        const vM = Number(classStartCapInput);
                        if (Number.isFinite(vM) && vM >= 0) {
                          const vPct = (vM / SALARY_CAP_M) * 100;
                          setClassStartCapPct(vPct);
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
                <div className="fa-class-grades-row">
                  <ClassMetricRing
                    title="Signing class"
                    subtitle="Incoming deals only"
                    grade={classGradeSummary.grade}
                    letter={classGradeSummary.letter}
                    variant="signing"
                  />
                  <ClassMetricRing
                    title="Roster net"
                    subtitle={
                      !classRosterNetSummary?.hasDepartures
                        ? 'Matches signing until departures are on'
                        : classRosterNetSummary.hasSignings
                          ? `Coverage ${classRosterNetSummary.replacementCoveragePct ?? 0}% vs losses · need ×${classRosterNetSummary.signingEmphasis ?? 1}`
                          : 'Departures with no FA class in this builder'
                    }
                    grade={classRosterNetSummary?.grade ?? null}
                    letter={classRosterNetSummary?.letter ?? null}
                    emptyHint={!classRosterNetSummary ? 'Add signings or departures' : undefined}
                    variant="net"
                  />
                </div>
                {classRosterNetSummary && (
                  <p className="fa-class-net-hint">
                    {!classRosterNetSummary.hasDepartures && (
                      <span>Add departures to separate roster net from signing-only.</span>
                    )}
                    {classRosterNetSummary.hasDepartures && classRosterNetSummary.hasSignings && classRosterNetSummary.signingEmphasis != null && (
                      <span>Under stress, signing quality is scaled up to ×{classRosterNetSummary.signingEmphasis} before applying losses.</span>
                    )}
                    {classRosterNetSummary.hasDepartures && !classRosterNetSummary.hasSignings && (
                      <span>Roster net reflects departures with <strong>no</strong> free-agent signings here — unfilled replacement penalty applies.</span>
                    )}
                  </p>
                )}
                <p className="fa-msg-text">
                  Cap Used (Yr 1): <strong>${pctToDollars(classUsedCapPct)}M</strong>
                  {departuresOn && (
                    <>
                      {' '}· Cap Freed: <strong style={{ color: '#3de87a' }}>${pctToDollars(classFreedCapPct)}M</strong>
                      {' '}· Net Used: <strong>${pctToDollars(classNetCapPct)}M</strong>
                    </>
                  )}
                  {' '}· Remaining:{' '}
                  <strong style={{ color: classRemainingCapPct != null && classRemainingCapPct < 0 ? '#e05555' : '#3de87a' }}>
                    {classRemainingCapPct == null ? 'N/A' : `$${pctToDollars(classRemainingCapPct)}M`}
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
                              <span className="fa-ranking-badge top">+${pctToDollarsNum(Number(d.freedCapPct || 0)).toFixed(1)}M</span>
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
                <div className="fa-field" style={{ marginBottom: 8 }}>
                  <label className="fa-label">Class Contract AAV ($M)</label>
                  <div className="fa-price-row">
                    <input
                      type="number"
                      min="0"
                      step="0.5"
                      className="fa-input"
                      value={classQuickAsk}
                      onChange={(e) => setClassQuickAsk(e.target.value)}
                    />
                    <span className="fa-million">M</span>
                  </div>
                </div>
                <div className="fa-field" style={{ marginBottom: 10 }}>
                  <label className="fa-label">Contract Years</label>
                  <select
                    className="fa-select"
                    value={classQuickYears}
                    onChange={(e) => setClassQuickYears(Number(e.target.value))}
                  >
                    {[1, 2, 3, 4, 5, 6, 7].map((y) => (
                      <option key={y} value={y}>{y}</option>
                    ))}
                  </select>
                </div>
                <div style={{ display: 'flex', gap: 8, marginBottom: 10 }}>
                  <button
                    type="button"
                    className="fa-btn"
                    onClick={handleQuickEvaluateAndAdd}
                    disabled={classQuickLoading || !classSearchPlayer}
                  >
                    {classQuickLoading ? 'Evaluating…' : 'Evaluate + Add to Class'}
                  </button>
                  <button type="button" className="fa-btn" onClick={handleGoToClassPlayer} disabled={!classSearchPlayer}>
                    Go to Player Page
                  </button>
                </div>
                <div className="fa-rankings-grid">
                  {classGradeSummary.rows.map((r) => (
                    <button
                      key={r.key}
                      type="button"
                      className="fa-ranking-card"
                      style={{ textAlign: 'left', cursor: 'pointer' }}
                      onClick={() => handleOpenClassSigningEvaluation(r)}
                      title="Open full evaluation"
                    >
                      <div className="fa-ranking-row">
                        <span className="fa-ranking-pos">{r.positionKey} — {r.playerName}</span>
                        <span className="fa-ranking-badge" style={classGradeBadgeStyle(r.signingGrade)}>
                          {Math.round(r.signingGrade)}
                        </span>
                      </div>
                      <div className="fa-hint">
                        ${r.ask}M x {r.years}y · Yr1 cap ${pctToDollarsNum(Number(r.yr1CapPct || 0)).toFixed(1)}M · weight {r.weight.toFixed(2)}
                      </div>
                    </button>
                  ))}
                </div>
                <div style={{ marginTop: 10, display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                  <button
                    type="button"
                    className="fa-btn"
                    onClick={() => setShowSigningClassDialog(true)}
                    disabled={!classGradeSummary?.rows?.length}
                  >
                    Explain signing class
                  </button>
                  <button
                    type="button"
                    className="fa-btn fa-btn--ghost"
                    onClick={() => setShowRosterNetDialog(true)}
                    disabled={!classRosterNetSummary}
                  >
                    Explain roster net impact
                  </button>
                  <button type="button" className="fa-btn" onClick={handleClearClass}>Clear Class</button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
      {showSigningClassDialog && classGradeSummary && (
        <div className="fa-rankings-overlay" onClick={() => setShowSigningClassDialog(false)}>
          <div className="fa-rankings-dialog" onClick={(e) => e.stopPropagation()}>
            <div className="fa-rankings-header">
              <h3>Signing class — {selectedTeam} ({analysisYear})</h3>
              <button type="button" className="fa-back-btn" onClick={() => setShowSigningClassDialog(false)}>Close</button>
            </div>
            <p className="fa-msg-text">
              Score: <strong>{classGradeSummary.grade}/100</strong> ({classGradeSummary.letter})
            </p>
            <p className="fa-hint" style={{ marginTop: 10 }}>
              This measures only how good the <strong>incoming contracts</strong> are (signing grades), weighted by position importance and deal size.
              It does <strong>not</strong> change when you add departures — use “Roster net impact” for that.
            </p>
          </div>
        </div>
      )}
      {showRosterNetDialog && classRosterNetSummary && (
        <div className="fa-rankings-overlay" onClick={() => setShowRosterNetDialog(false)}>
          <div className="fa-rankings-dialog" onClick={(e) => e.stopPropagation()}>
            <div className="fa-rankings-header">
              <h3>Roster net impact — {selectedTeam} ({analysisYear})</h3>
              <button type="button" className="fa-back-btn" onClick={() => setShowRosterNetDialog(false)}>Close</button>
            </div>
            <p className="fa-msg-text">
              Net score: <strong>{classRosterNetSummary.grade}/100</strong> ({classRosterNetSummary.letter})
            </p>
            <p className="fa-msg-text" style={{ marginTop: 8 }}>
              Signing class baseline: <strong>{classRosterNetSummary.signingGrade}/100</strong>
              {classRosterNetSummary.hasDepartures && (
                <>
                  {' '}· Adjusted for need (×<strong>{classRosterNetSummary.signingEmphasis}</strong>): <strong>{classRosterNetSummary.adjustedSigningBaseline}/100</strong>
                  {' '}· Avg departure grade (weighted): ~<strong>{classRosterNetSummary.avgDepartureGrade}</strong>
                  {' '}· Talent-out penalty: <strong>−{classRosterNetSummary.lossPenalty}</strong> pts
                </>
              )}
            </p>
            <p className="fa-hint" style={{ marginTop: 10 }}>
              {classRosterNetSummary.explanation}
            </p>
          </div>
        </div>
      )}
    </div>
    </div>
  );
}

/* ─── Root ─── */
function FreeAgency() {
  return (
    <PositionEvaluator
      positionKey={FA_POSITION_ORDER[0]}
      pendingPick={null}
      clearPendingPick={() => {}}
    />
  );
}

export default FreeAgency;
