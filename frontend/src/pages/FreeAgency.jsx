import React, { useState, useEffect, useRef } from 'react';
import './FreeAgency.css';

const ED_API = 'http://127.0.0.1:8002';

const POSITIONS = [
  { label: 'QB',   name: 'Quarterback',        available: false },
  { label: 'HB',   name: 'Running Back',        available: false },
  { label: 'WR',   name: 'Wide Receiver',       available: false },
  { label: 'TE',   name: 'Tight End',           available: false },
  { label: 'T',    name: 'Tackle',              available: false },
  { label: 'G',    name: 'Guard',               available: false },
  { label: 'C',    name: 'Center',              available: false },
  { label: 'ED',   name: 'Edge Defender',       available: true  },
  { label: 'DI',   name: 'Defensive Interior',  available: false },
  { label: 'LB',   name: 'Linebacker',          available: false },
  { label: 'CB',   name: 'Cornerback',          available: false },
  { label: 'S',    name: 'Safety',              available: false },
];

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

/* ─── ED Evaluator ─── */
function EDEvaluator({ onBack }) {
  const [players, setPlayers] = useState([]);
  const [selectedPlayer, setSelectedPlayer] = useState('');
  const [salaryAsk, setSalaryAsk] = useState('');
  const [loading, setLoading] = useState(false);
  const [fetchingPlayers, setFetchingPlayers] = useState(true);
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content:
        'Welcome to the Edge Defender Free Agency Evaluator. Select a player and enter a contract value (AAV in $M), then click Analyze to get a SIGN / PASS recommendation.',
    },
  ]);
  const [error, setError] = useState('');
  const chatEndRef = useRef(null);

  useEffect(() => {
    fetch(`${ED_API}/ed-players`)
      .then((r) => r.json())
      .then((data) => {
        setPlayers(data.players || []);
        setSelectedPlayer(data.players?.[0] || '');
      })
      .catch(() => setError('Could not load player list. Make sure the ED API is running on port 8002.'))
      .finally(() => setFetchingPlayers(false));
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const buildDetailLines = (result, ask) => {
    const { decision, reasoning, data } = result;
    const { predicted_tier, valuation_estimate, confidence } = data;
    const { predicted_grade, transformer_grade, xgb_grade, age_adjustment } = confidence || {};

    const lines = [
      { label: 'Decision',          value: decision, highlight: decision === 'SIGN' ? 'sign' : 'pass' },
      { label: 'Projected Tier',    value: predicted_tier },
      { label: 'Predicted Grade',   value: predicted_grade != null ? `${Number(predicted_grade).toFixed(1)} / 100` : 'N/A' },
      { label: 'Fair Market Value', value: `$${valuation_estimate}M / yr` },
      { label: 'Salary Ask',        value: `$${ask}M / yr` },
    ];

    if (transformer_grade != null)
      lines.push({ label: 'Transformer Grade', value: Number(transformer_grade).toFixed(1) });
    if (xgb_grade != null)
      lines.push({ label: 'XGBoost Grade', value: Number(xgb_grade).toFixed(1) });
    if (age_adjustment != null && age_adjustment !== 0)
      lines.push({ label: 'Age Adjustment', value: `-${Number(age_adjustment).toFixed(1)} pts` });

    return { lines, reasoning };
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
      { role: 'user', content: `Evaluate ${selectedPlayer} at $${ask}M/yr AAV.` },
    ]);
    setLoading(true);

    try {
      const resp = await fetch(`${ED_API}/evaluate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ player_name: selectedPlayer, salary_ask: ask }),
      });

      if (!resp.ok) {
        const err = await resp.json();
        throw new Error(err.detail || 'Evaluation failed.');
      }

      const result = await resp.json();
      const { lines, reasoning } = buildDetailLines(result, ask);

      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: null, structured: { lines, reasoning } },
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
      {/* Left panel */}
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

        {error && <p className="fa-error">{error}</p>}

        <button
          className="fa-btn"
          onClick={handleAnalyze}
          disabled={loading || fetchingPlayers || !selectedPlayer}
        >
          {loading ? 'Analyzing…' : 'Analyze Player'}
        </button>

        <div className="fa-legend">
          <p className="fa-legend-title">Tier Fair Values</p>
          <div className="fa-legend-row"><span className="tier-badge elite">Elite</span><span>$28M / yr</span></div>
          <div className="fa-legend-row"><span className="tier-badge starter">Starter</span><span>$16M / yr</span></div>
          <div className="fa-legend-row"><span className="tier-badge rotation">Rotation</span><span>$6M / yr</span></div>
          <div className="fa-legend-row"><span className="tier-badge reserve">Reserve</span><span>$1.5M / yr</span></div>
        </div>
      </div>

      {/* Right chat panel */}
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
                  <div className={`fa-decision-badge ${msg.structured.lines[0].highlight}`}>
                    {msg.structured.lines[0].value}
                  </div>
                  <div className="fa-stats-grid">
                    {msg.structured.lines.slice(1).map((line, j) => (
                      <div key={j} className="fa-stat-row">
                        <span className="fa-stat-label">{line.label}</span>
                        <span className="fa-stat-value">{line.value}</span>
                      </div>
                    ))}
                  </div>
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

  // Only ED is available right now
  return <EDEvaluator onBack={() => setSelectedPosition(null)} />;
}

export default FreeAgency;
