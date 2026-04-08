import React, { useState, useEffect, useRef, useCallback } from 'react';
import './FreeAgency.css';

const ED_API = 'http://127.0.0.1:8002';
const DI_API = 'http://127.0.0.1:8003';

const POSITIONS = [
  { label: 'QB',  name: 'Quarterback',       available: false },
  { label: 'HB',  name: 'Running Back',       available: false },
  { label: 'WR',  name: 'Wide Receiver',      available: false },
  { label: 'TE',  name: 'Tight End',          available: false },
  { label: 'T',   name: 'Tackle',             available: false },
  { label: 'G',   name: 'Guard',              available: false },
  { label: 'C',   name: 'Center',             available: false },
  { label: 'ED',  name: 'Edge Defender',      available: true  },
  { label: 'DI',  name: 'Defensive Interior', available: true  },
  { label: 'LB',  name: 'Linebacker',         available: false },
  { label: 'CB',  name: 'Cornerback',         available: false },
  { label: 'S',   name: 'Safety',             available: false },
];

/* ─── Decision Tier Legend (shared) ─── */
const TIER_LADDER = [
  { key: 'exceptional', label: 'Exceptional Value', range: '> 20% surplus',   color: '#3de87a' },
  { key: 'good',        label: 'Good Signing',       range: '5 – 20% surplus', color: '#5dbb6e' },
  { key: 'fair',        label: 'Fair Deal',           range: '± 5% of value',  color: '#d4c94a' },
  { key: 'slight-overpay', label: 'Slight Overpay',  range: '5 – 15% over',   color: '#e8a44a' },
  { key: 'overpay',     label: 'Overpay',             range: '15 – 30% over',  color: '#e07030' },
  { key: 'poor',        label: 'Poor Signing',        range: '> 30% over',     color: '#e05555' },
];

function DecisionTierLegend() {
  return (
    <div className="fa-tier-legend">
      <p className="fa-legend-title">Decision Tiers</p>
      {TIER_LADDER.map((t, i) => (
        <div key={t.key} className="fa-tier-row">
          <span className="fa-tier-rank">{i + 1}</span>
          <span className="fa-tier-dot" style={{ background: t.color }} />
          <span className="fa-tier-label" style={{ color: t.color }}>{t.label}</span>
          <span className="fa-tier-range">{t.range}</span>
        </div>
      ))}
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

function signingGradeFromData(fair_aav, cap_burden) {
  const surplus_pct = (fair_aav - cap_burden) / Math.max(fair_aav, 0.01) * 100;
  return Math.round(_lerp(_SURPLUS_ANCHORS, _GRADE_ANCHORS, surplus_pct));
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
              className={`fa-pos-btn ${pos.available ? 'fa-pos-btn--active' : 'fa-pos-btn--disabled'}`}
              onClick={() => pos.available && onSelect(pos.label)}
              disabled={!pos.available}
            >
              <span className="fa-pos-abbr">{pos.label}</span>
              <span className="fa-pos-name">{pos.name}</span>
              {!pos.available && <span className="fa-pos-soon">Coming Soon</span>}
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

/* ─── Stats Panel ─── */
const STAT_ROWS = [
  { key: 'sacks',          label: 'Sacks' },
  { key: 'pressures',      label: 'Pressures' },
  { key: 'pressure_pct',   label: 'Pressure %',   fmt: v => `${v}%` },
  { key: 'stops',          label: 'Stops' },
  { key: 'pass_rush_grade',label: 'PR Grade' },
  { key: 'run_def_grade',  label: 'RD Grade' },
  { key: 'overall_grade',  label: 'Overall Grade' },
];

function StatsPanel({ careerStats, projectedStats }) {
  const lastCareer = careerStats[careerStats.length - 1];
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
            {STAT_ROWS.map(({ key, label, fmt }) => (
              <tr key={key}>
                <td className="fa-stats-row-hdr">{label}</td>
                {careerStats.map(s => (
                  <td key={s.season} className="fa-stats-career-cell">
                    {s[key] != null ? (fmt ? fmt(s[key]) : s[key]) : '—'}
                  </td>
                ))}
                {projectedStats.map(yr => {
                  const val = yr[key];
                  const last = lastCareer?.[key];
                  const delta = (val != null && last != null && last !== 0 && key !== 'overall_grade')
                    ? (val - last) : null;
                  return (
                    <td key={yr.year} className={
                      delta == null ? '' : delta > 0.05 ? 'fa-stat-up' : delta < -0.05 ? 'fa-stat-down' : ''
                    }>
                      {val != null ? (fmt ? fmt(val) : val) : '—'}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="fa-breakdown-note">
        Career columns show actual per-season stats. Projected years assume 17 healthy games, scaled by the composite grade trajectory.
        Composite = 30% model PFF grade + 70% stats-based grade (pressure %, sack rate, stops).
      </p>
    </div>
  );
}

/* ─── ED Evaluator ─── */
function EDEvaluator({ onBack }) {
  const [players, setPlayers] = useState([]);
  const [selectedPlayer, setSelectedPlayer] = useState('');
  const [salaryAsk, setSalaryAsk] = useState('');
  const [contractYears, setContractYears] = useState(1);
  const [loading, setLoading] = useState(false);
  const [fetchingPlayers, setFetchingPlayers] = useState(true);
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content:
        'Welcome to the Edge Defender Free Agency Evaluator. Select a player, set the contract AAV and length, then click Analyze to get a SIGN / PASS recommendation with a year-by-year projection.',
    },
  ]);
  const [error, setError] = useState('');
  const [statsOpen, setStatsOpen] = useState({});  // msgIndex → bool
  const chatEndRef = useRef(null);

  const toggleStats = useCallback((i) => {
    setStatsOpen(prev => ({ ...prev, [i]: !prev[i] }));
  }, []);

  useEffect(() => {
    fetch(`${ED_API}/ed-players`)
      .then((r) => r.json())
      .then((data) => {
        setPlayers(data.players || []);
        setSelectedPlayer(data.players?.[0] || '');
      })
      .catch(() =>
        setError('Could not load player list. Make sure the ED API is running on port 8002.')
      )
      .finally(() => setFetchingPlayers(false));
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const buildStructured = (result, ask, years) => {
    const { decision, reasoning, data } = result;
    const {
      predicted_tier, current_age, effective_fair_aav, effective_cap_burden,
      total_nominal_value, total_ask, confidence, year_breakdown,
      last_season_stats, projected_stats, career_stats,
    } = data;
    const { model_grade, stats_grade, composite_grade, health_factor, avg_availability,
            transformer_grade, xgb_grade, age_adjustment } = confidence || {};

    const statRows = [
      { divider: true, title: 'Player Profile' },
      { label: 'Projected Tier',        value: predicted_tier },
      { label: 'Current Age',           value: current_age },
      { divider: true, title: 'Grade Breakdown' },
      { label: 'Model Grade',           value: model_grade != null ? `${Number(model_grade).toFixed(1)} / 100` : 'N/A' },
      { label: 'Stats Grade',           value: stats_grade != null ? `${Number(stats_grade).toFixed(1)} / 100` : 'N/A' },
      { label: 'Composite Grade',       value: composite_grade != null ? `${Number(composite_grade).toFixed(1)} / 100` : 'N/A' },
    ];
    if (transformer_grade != null)
      statRows.push({ label: 'Transformer Grade', value: Number(transformer_grade).toFixed(1) });
    if (xgb_grade != null)
      statRows.push({ label: 'XGBoost Grade', value: Number(xgb_grade).toFixed(1) });
    if (age_adjustment != null && age_adjustment !== 0)
      statRows.push({ label: 'Age Penalty (applied)', value: `-${Number(age_adjustment).toFixed(1)} pts` });

    statRows.push(
      { divider: true, title: 'Health & Availability' },
      { label: 'Availability (3yr)',    value: avg_availability != null ? `${Math.round(avg_availability * 100)}%` : 'N/A' },
      { label: 'Health Factor',         value: health_factor != null ? `${health_factor >= 0 ? '+' : ''}${Number(health_factor).toFixed(1)} pts` : 'N/A' },
      { divider: true, title: 'Contract Valuation' },
      { label: 'Contract',              value: `$${ask}M/yr × ${years} yr  =  $${total_ask}M total` },
      { label: 'Fair AAV (cap-adj PV)', value: `$${effective_fair_aav}M / yr` },
      { label: 'Real Cap Burden (PV)',  value: `$${effective_cap_burden}M / yr` },
      { label: 'Total Nominal Value',   value: `$${total_nominal_value}M` },
    );

    const DECISION_CLASS = {
      'Exceptional Value': 'exceptional',
      'Good Signing':      'good',
      'Fair Deal':         'fair',
      'Slight Overpay':    'slight-overpay',
      'Overpay':           'overpay',
      'Poor Signing':      'poor',
    };

    return {
      decision,
      highlight: DECISION_CLASS[decision] || 'fair',
      signing_grade: signingGradeFromData(effective_fair_aav, effective_cap_burden),
      stats: statRows,
      reasoning,
      year_breakdown,
      last_season_stats,
      projected_stats,
      career_stats: career_stats || [],
    };
  };

  const handleAnalyze = async () => {
    if (!selectedPlayer) return;
    const ask = parseFloat(salaryAsk);
    if (isNaN(ask) || ask <= 0) {
      setError('Please enter a valid salary (positive number in $M).');
      return;
    }
    setError('');

    setMessages((prev) => [
      ...prev,
      {
        role: 'user',
        content: `Evaluate ${selectedPlayer} — $${ask}M/yr × ${contractYears} yr contract.`,
      },
    ]);
    setLoading(true);

    try {
      const resp = await fetch(`${ED_API}/evaluate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          player_name:    selectedPlayer,
          salary_ask:     ask,
          contract_years: contractYears,
        }),
      });

      if (!resp.ok) {
        const err = await resp.json();
        throw new Error(err.detail || 'Evaluation failed.');
      }

      const result = await resp.json();
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: null,
          structured: buildStructured(result, ask, contractYears),
        },
      ]);
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: `Error: ${e.message}` },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fa-page">
      {/* ── Left panel ── */}
      <div className="fa-panel">
        <button className="fa-back-btn" onClick={onBack}>← Change Position</button>
        <h2 className="fa-panel-title">Edge Defender Scout</h2>

        <div className="fa-field">
          <label className="fa-label">Player</label>
          {fetchingPlayers ? (
            <p className="fa-hint">Loading players…</p>
          ) : (
            <select
              className="fa-select"
              value={selectedPlayer}
              onChange={(e) => setSelectedPlayer(e.target.value)}
            >
              {players.map((p) => (
                <option key={p} value={p}>{p}</option>
              ))}
            </select>
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
            max="7"
            step="1"
            className="fa-slider"
            value={contractYears}
            onChange={(e) => setContractYears(Number(e.target.value))}
          />
          <div className="fa-slider-ticks">
            {[1,2,3,4,5,6,7].map((n) => (
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
          className="fa-btn"
          onClick={handleAnalyze}
          disabled={loading || fetchingPlayers || !selectedPlayer}
        >
          {loading ? 'Analyzing…' : 'Analyze Player'}
        </button>

        <div className="fa-legend">
          <p className="fa-legend-title">2026 Market Ranges</p>
          <div className="fa-legend-row"><span className="tier-badge elite">Elite</span><span>$33–50M</span></div>
          <div className="fa-legend-row"><span className="tier-badge starter">Starter</span><span>$13–33M</span></div>
          <div className="fa-legend-row"><span className="tier-badge rotation">Rotation</span><span>$4–13M</span></div>
          <div className="fa-legend-row"><span className="tier-badge reserve">Reserve</span><span>&lt;$4M</span></div>
          <p className="fa-legend-note">Calibrated to 2026 OTC contracts.<br/>Age curve derived from 1,433 ED seasons.<br/>Future years discounted at 8%/yr.</p>
        </div>

        <DecisionTierLegend />
      </div>

      {/* ── Right chat panel ── */}
      <div className="fa-chat">
        <div className="fa-chat-header">GM Decision Feed — Edge Defender</div>
        <div className="fa-chat-body">
          {messages.map((msg, i) => (
            <div key={i} className={`fa-msg fa-msg--${msg.role}`}>
              <div className="fa-msg-label">{msg.role === 'user' ? 'You' : 'GM Agent'}</div>

              {msg.content != null ? (
                <div className="fa-msg-text">{msg.content}</div>
              ) : (
                <div className="fa-msg-card">
                  <div className={`fa-decision-badge ${msg.structured.highlight}`}>
                    {msg.structured.decision}
                  </div>

                  <SigningGrade grade={msg.structured.signing_grade} />

                  <div className="fa-stats-grid">
                    {msg.structured.stats.map((s, j) =>
                      s.divider ? (
                        <div key={j} className="fa-stat-section-hdr">{s.title}</div>
                      ) : (
                        <div key={j} className="fa-stat-row">
                          <span className="fa-stat-label">{s.label}</span>
                          <span className="fa-stat-value">{s.value}</span>
                        </div>
                      )
                    )}
                  </div>

                  {msg.structured.projected_stats?.length > 0 && (
                    <div className="fa-stats-toggle-row">
                      <button
                        className="fa-stats-toggle-btn"
                        onClick={() => toggleStats(i)}
                      >
                        {statsOpen[i] ? '▲ Hide Stats Projection' : '▼ View Stats Projection'}
                      </button>
                    </div>
                  )}

                  {statsOpen[i] && msg.structured.career_stats?.length > 0 && (
                    <StatsPanel
                      careerStats={msg.structured.career_stats}
                      projectedStats={msg.structured.projected_stats}
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
    </div>
  );
}

/* ─── DI Evaluator ─── */
const DI_STAT_ROWS = [
  { key: 'stops',          label: 'Stops' },
  { key: 'tfl',            label: 'TFL' },
  { key: 'pressures',      label: 'Pressures' },
  { key: 'sacks',          label: 'Sacks' },
  { key: 'stop_rate',      label: 'Stop Rate %',  fmt: v => `${v}%` },
  { key: 'pass_rush_grade',label: 'PR Grade' },
  { key: 'run_def_grade',  label: 'RD Grade' },
  { key: 'overall_grade',  label: 'Overall Grade' },
];

function DIStatsPanel({ careerStats, projectedStats }) {
  const lastCareer = careerStats[careerStats.length - 1];
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
            {DI_STAT_ROWS.map(({ key, label, fmt }) => (
              <tr key={key}>
                <td className="fa-stats-row-hdr">{label}</td>
                {careerStats.map(s => (
                  <td key={s.season} className="fa-stats-career-cell">
                    {s[key] != null ? (fmt ? fmt(s[key]) : s[key]) : '—'}
                  </td>
                ))}
                {projectedStats.map(yr => {
                  const val = yr[key];
                  const last = lastCareer?.[key];
                  const delta = (val != null && last != null && last !== 0 && key !== 'overall_grade')
                    ? (val - last) : null;
                  return (
                    <td key={yr.year} className={
                      delta == null ? '' : delta > 0.05 ? 'fa-stat-up' : delta < -0.05 ? 'fa-stat-down' : ''
                    }>
                      {val != null ? (fmt ? fmt(val) : val) : '—'}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="fa-breakdown-note">
        Career columns show actual per-season stats. Projected years assume 17 healthy games.
        Composite = 30% model PFF grade + 70% stats-based grade (stop rate 40%, TFL 20%, pressure 20%, sacks 20%).
      </p>
    </div>
  );
}

function DIEvaluator({ onBack }) {
  const [players, setPlayers] = useState([]);
  const [selectedPlayer, setSelectedPlayer] = useState('');
  const [salaryAsk, setSalaryAsk] = useState('');
  const [contractYears, setContractYears] = useState(1);
  const [loading, setLoading] = useState(false);
  const [fetchingPlayers, setFetchingPlayers] = useState(true);
  const [messages, setMessages] = useState([{
    role: 'assistant',
    content: 'Welcome to the Defensive Interior Free Agency Evaluator. Run-stopping is the primary value driver. Select a player, set the contract AAV and length, then click Analyze to get a recommendation.',
  }]);
  const [error, setError] = useState('');
  const [statsOpen, setStatsOpen] = useState({});
  const chatEndRef = useRef(null);

  const toggleStats = useCallback((i) => {
    setStatsOpen(prev => ({ ...prev, [i]: !prev[i] }));
  }, []);

  useEffect(() => {
    fetch(`${DI_API}/di-players`)
      .then(r => r.json())
      .then(data => {
        setPlayers(data.players || []);
        setSelectedPlayer(data.players?.[0] || '');
      })
      .catch(() => setError('Could not load player list. Make sure the DI API is running on port 8003.'))
      .finally(() => setFetchingPlayers(false));
  }, []);

  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  const buildStructured = (result, ask, years) => {
    const { decision, reasoning, data } = result;
    const {
      predicted_tier, current_age, effective_fair_aav, effective_cap_burden,
      total_nominal_value, total_ask, confidence, year_breakdown,
      last_season_stats, projected_stats, career_stats,
    } = data;
    const { model_grade, stats_grade, composite_grade, health_factor, avg_availability,
            transformer_grade, xgb_grade, age_adjustment } = confidence || {};

    const statRows = [
      { divider: true, title: 'Player Profile' },
      { label: 'Projected Tier',        value: predicted_tier },
      { label: 'Current Age',           value: current_age },
      { divider: true, title: 'Grade Breakdown' },
      { label: 'Model Grade',           value: model_grade != null ? `${Number(model_grade).toFixed(1)} / 100` : 'N/A' },
      { label: 'Stats Grade',           value: stats_grade != null ? `${Number(stats_grade).toFixed(1)} / 100` : 'N/A' },
      { label: 'Composite Grade',       value: composite_grade != null ? `${Number(composite_grade).toFixed(1)} / 100` : 'N/A' },
    ];
    if (transformer_grade != null)
      statRows.push({ label: 'Transformer Grade', value: Number(transformer_grade).toFixed(1) });
    if (age_adjustment != null && age_adjustment !== 0)
      statRows.push({ label: 'Age Penalty (applied)', value: `-${Number(age_adjustment).toFixed(1)} pts` });

    statRows.push(
      { divider: true, title: 'Health & Availability' },
      { label: 'Availability (3yr)',    value: avg_availability != null ? `${Math.round(avg_availability * 100)}%` : 'N/A' },
      { label: 'Health Factor',         value: health_factor != null ? `${health_factor >= 0 ? '+' : ''}${Number(health_factor).toFixed(1)} pts` : 'N/A' },
      { divider: true, title: 'Contract Valuation' },
      { label: 'Contract',              value: `$${ask}M/yr × ${years} yr  =  $${total_ask}M total` },
      { label: 'Fair AAV (cap-adj PV)', value: `$${effective_fair_aav}M / yr` },
      { label: 'Real Cap Burden (PV)',  value: `$${effective_cap_burden}M / yr` },
      { label: 'Total Nominal Value',   value: `$${total_nominal_value}M` },
    );

    const DECISION_CLASS = {
      'Exceptional Value': 'exceptional',
      'Good Signing':      'good',
      'Fair Deal':         'fair',
      'Slight Overpay':    'slight-overpay',
      'Overpay':           'overpay',
      'Poor Signing':      'poor',
    };

    return {
      decision,
      highlight: DECISION_CLASS[decision] || 'fair',
      signing_grade: signingGradeFromData(effective_fair_aav, effective_cap_burden),
      stats: statRows,
      reasoning,
      year_breakdown,
      last_season_stats,
      projected_stats,
      career_stats: career_stats || [],
    };
  };

  const handleAnalyze = async () => {
    if (!selectedPlayer) return;
    const ask = parseFloat(salaryAsk);
    if (isNaN(ask) || ask <= 0) { setError('Please enter a valid salary.'); return; }
    setError('');
    setMessages(prev => [...prev, {
      role: 'user',
      content: `Evaluate ${selectedPlayer} — $${ask}M/yr × ${contractYears} yr contract.`,
    }]);
    setLoading(true);
    try {
      const resp = await fetch(`${DI_API}/evaluate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ player_name: selectedPlayer, salary_ask: ask, contract_years: contractYears }),
      });
      if (!resp.ok) { const err = await resp.json(); throw new Error(err.detail || 'Evaluation failed.'); }
      const result = await resp.json();
      setMessages(prev => [...prev, {
        role: 'assistant', content: null,
        structured: buildStructured(result, ask, contractYears),
      }]);
    } catch (e) {
      setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${e.message}` }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fa-page">
      <div className="fa-panel">
        <button className="fa-back-btn" onClick={onBack}>← Change Position</button>
        <h2 className="fa-panel-title">Defensive Interior Scout</h2>

        <div className="fa-field">
          <label className="fa-label">Player</label>
          {fetchingPlayers ? <p className="fa-hint">Loading players…</p> : (
            <select className="fa-select" value={selectedPlayer} onChange={e => setSelectedPlayer(e.target.value)}>
              {players.map(p => <option key={p} value={p}>{p}</option>)}
            </select>
          )}
        </div>

        <div className="fa-field">
          <label className="fa-label">Contract AAV ($M / yr)</label>
          <div className="fa-price-row">
            <span className="fa-dollar">$</span>
            <input type="number" min="0" step="0.5" className="fa-input" placeholder="e.g. 14.0"
              value={salaryAsk} onChange={e => setSalaryAsk(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleAnalyze()} />
            <span className="fa-million">M</span>
          </div>
        </div>

        <div className="fa-field">
          <label className="fa-label">Contract Length — {contractYears} yr</label>
          <input type="range" min="1" max="7" step="1" className="fa-slider"
            value={contractYears} onChange={e => setContractYears(Number(e.target.value))} />
          <div className="fa-slider-ticks">
            {[1,2,3,4,5,6,7].map(n => (
              <span key={n} className={n === contractYears ? 'fa-tick fa-tick--active' : 'fa-tick'}
                onClick={() => setContractYears(n)}>{n}</span>
            ))}
          </div>
        </div>

        {error && <p className="fa-error">{error}</p>}

        <button className="fa-btn" onClick={handleAnalyze}
          disabled={loading || fetchingPlayers || !selectedPlayer}>
          {loading ? 'Analyzing…' : 'Analyze Player'}
        </button>

        <div className="fa-legend">
          <p className="fa-legend-title">2026 Market Ranges (DI)</p>
          <div className="fa-legend-row"><span className="tier-badge elite">Elite</span><span>$21–42M</span></div>
          <div className="fa-legend-row"><span className="tier-badge starter">Starter</span><span>$11–21M</span></div>
          <div className="fa-legend-row"><span className="tier-badge rotation">Rotation</span><span>$3–11M</span></div>
          <div className="fa-legend-row"><span className="tier-badge reserve">Reserve</span><span>&lt;$3M</span></div>
          <p className="fa-legend-note">Stop rate weighted 50% in stats grade.<br/>Age curve derived from DI season data.<br/>Future years discounted at 8%/yr.</p>
        </div>
      </div>

      <div className="fa-chat">
        <div className="fa-chat-header">GM Decision Feed — Defensive Interior</div>
        <div className="fa-chat-body">
          {messages.map((msg, i) => (
            <div key={i} className={`fa-msg fa-msg--${msg.role}`}>
              <div className="fa-msg-label">{msg.role === 'user' ? 'You' : 'GM Agent'}</div>
              {msg.content != null ? (
                <div className="fa-msg-text">{msg.content}</div>
              ) : (
                <div className="fa-msg-card">
                  <div className={`fa-decision-badge ${msg.structured.highlight}`}>
                    {msg.structured.decision}
                  </div>

                  <SigningGrade grade={msg.structured.signing_grade} />

                  <div className="fa-stats-grid">
                    {msg.structured.stats.map((s, j) =>
                      s.divider ? (
                        <div key={j} className="fa-stat-section-hdr">{s.title}</div>
                      ) : (
                        <div key={j} className="fa-stat-row">
                          <span className="fa-stat-label">{s.label}</span>
                          <span className="fa-stat-value">{s.value}</span>
                        </div>
                      )
                    )}
                  </div>
                  {msg.structured.projected_stats?.length > 0 && (
                    <div className="fa-stats-toggle-row">
                      <button className="fa-stats-toggle-btn" onClick={() => toggleStats(i)}>
                        {statsOpen[i] ? '▲ Hide Stats Projection' : '▼ View Stats Projection'}
                      </button>
                    </div>
                  )}
                  {statsOpen[i] && msg.structured.career_stats?.length > 0 && (
                    <DIStatsPanel
                      careerStats={msg.structured.career_stats}
                      projectedStats={msg.structured.projected_stats}
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
    </div>
  );
}

/* ─── Root ─── */
function FreeAgency() {
  const [selectedPosition, setSelectedPosition] = useState(null);

  if (!selectedPosition) {
    return <PositionPicker onSelect={setSelectedPosition} />;
  }
  if (selectedPosition === 'DI') {
    return <DIEvaluator onBack={() => setSelectedPosition(null)} />;
  }
  return <EDEvaluator onBack={() => setSelectedPosition(null)} />;
}

export default FreeAgency;
