import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './DefenseAnalyst.css';

const API = 'http://127.0.0.1:5000';

const SESSION_ID = () => `session_${Date.now()}`;

const DefenseAnalyst = ({ position }) => {
  // ── Evaluate Panel ─────────────────────────────────────────────
  const [playerName, setPlayerName] = useState('');
  const [salaryAsk, setSalaryAsk] = useState('');
  const [evalResult, setEvalResult] = useState(null);
  const [evalLoading, setEvalLoading] = useState(false);
  const [evalError, setEvalError] = useState('');

  // ── Chat Panel ─────────────────────────────────────────────────
  const [messages, setMessages] = useState([
    { role: 'gm', text: `Welcome. I'm your ${position} analyst. Ask me about any ${position === 'CB' ? 'cornerback' : 'linebacker'} — evaluations, comparisons, contract decisions, 2025 projections.` }
  ]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const sessionId = useRef(SESSION_ID());
  const chatEndRef = useRef(null);

  const posLabel = position === 'CB' ? 'Cornerback' : 'Linebacker';
  const accentColor = position === 'CB' ? '#4f9eff' : '#ff7c4f';

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // ── Evaluate ───────────────────────────────────────────────────
  const handleEvaluate = async () => {
    if (!playerName.trim()) { setEvalError('Enter a player name.'); return; }
    setEvalLoading(true);
    setEvalError('');
    setEvalResult(null);
    try {
      const res = await axios.post(`${API}/evaluate-player`, {
        player_name: playerName.trim(),
        salary_ask: parseFloat(salaryAsk) || 0,
        position
      });
      setEvalResult(res.data);
    } catch (err) {
      setEvalError(err.response?.data?.error || 'Player not found.');
    } finally {
      setEvalLoading(false);
    }
  };

  // ── Chat ───────────────────────────────────────────────────────
  const handleChat = async () => {
    const msg = chatInput.trim();
    if (!msg) return;
    setChatInput('');
    setMessages(prev => [...prev, { role: 'user', text: msg }]);
    setChatLoading(true);
    try {
      const res = await axios.post(`${API}/gm-chat`, {
        message: msg,
        session_id: sessionId.current,
        position
      });
      setMessages(prev => [...prev, { role: 'gm', text: res.data.reply, agentData: res.data.agent_data }]);
    } catch (err) {
      setMessages(prev => [...prev, { role: 'gm', text: 'Sorry, something went wrong. Try again.' }]);
    } finally {
      setChatLoading(false);
    }
  };

  const handleChatKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleChat(); }
  };

  const handleReset = async () => {
    await axios.post(`${API}/gm-chat/reset`, { session_id: sessionId.current }).catch(() => {});
    sessionId.current = SESSION_ID();
    setMessages([{ role: 'gm', text: `Fresh start. Ask me anything about ${posLabel}s.` }]);
  };

  const tierColor = (tier) => {
    if (tier === 'Elite') return '#ffd700';
    if (tier === 'Starter') return '#4f9eff';
    return '#aaaaaa';
  };

  const decisionColor = (decision) => decision === 'SIGN' ? '#4caf50' : '#ff5c5c';

  return (
    <div className="da-page">
      <div className="da-header" style={{ borderBottomColor: accentColor }}>
        <h1>{posLabel} <span style={{ color: accentColor }}>GM Analyst</span></h1>
        <p className="da-subtitle">Time2Vec Transformer · LangGraph Agent · 2025 Projections</p>
      </div>

      <div className="da-body">
        {/* ── Left: Evaluate Panel ── */}
        <div className="da-panel">
          <h2 className="da-panel-title">Player Evaluation</h2>
          <div className="da-form">
            <input
              className="da-input"
              placeholder={`${posLabel} name (e.g. ${position === 'CB' ? 'Sauce Gardner' : 'Fred Warner'})`}
              value={playerName}
              onChange={e => setPlayerName(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleEvaluate()}
            />
            <div className="da-salary-row">
              <span className="da-salary-label">Salary ask ($M)</span>
              <input
                className="da-input da-salary-input"
                type="number"
                placeholder="e.g. 18"
                value={salaryAsk}
                onChange={e => setSalaryAsk(e.target.value)}
              />
            </div>
            <button
              className="da-btn"
              style={{ backgroundColor: accentColor }}
              onClick={handleEvaluate}
              disabled={evalLoading}
            >
              {evalLoading ? 'Evaluating…' : 'Evaluate'}
            </button>
          </div>

          {evalError && <p className="da-error">{evalError}</p>}

          {evalResult && (
            <div className="da-result-card">
              <div className="da-result-header">
                <span className="da-player-name">{evalResult.player}</span>
                <span className="da-tier-badge" style={{ color: tierColor(evalResult.predicted_tier), borderColor: tierColor(evalResult.predicted_tier) }}>
                  {evalResult.predicted_tier}
                </span>
              </div>

              <div className="da-grade-row">
                <span className="da-grade-label">Projected Grade</span>
                <span className="da-grade-value">{evalResult.predicted_grade}</span>
                <span className="da-grade-range">[{evalResult.conf_low} – {evalResult.conf_high}]</span>
              </div>

              <div className="da-stats-grid">
                <div className="da-stat">
                  <span className="da-stat-label">Fair Value</span>
                  <span className="da-stat-value">${evalResult.fair_value}M</span>
                </div>
                <div className="da-stat">
                  <span className="da-stat-label">Volatility</span>
                  <span className="da-stat-value">{evalResult.volatility_index}</span>
                </div>
                <div className="da-stat">
                  <span className="da-stat-label">Age Penalty</span>
                  <span className="da-stat-value">{evalResult.age_adjustment} pts</span>
                </div>
                <div className="da-stat">
                  <span className="da-stat-label">Decision</span>
                  <span className="da-stat-value" style={{ color: decisionColor(evalResult.decision) }}>
                    {evalResult.decision}
                  </span>
                </div>
              </div>

              <p className="da-reasoning">{evalResult.reasoning}</p>
            </div>
          )}
        </div>

        {/* ── Right: Chat Panel ── */}
        <div className="da-panel da-chat-panel">
          <div className="da-chat-header">
            <h2 className="da-panel-title">GM Chat</h2>
            <button className="da-reset-btn" onClick={handleReset}>New Chat</button>
          </div>

          <div className="da-chat-messages">
            {messages.map((msg, i) => (
              <div key={i} className={`da-msg da-msg-${msg.role}`}>
                <div className="da-msg-label">{msg.role === 'gm' ? '🏈 GM' : 'You'}</div>
                <div className="da-msg-text">{msg.text}</div>
                {msg.agentData && msg.agentData.length > 0 && (
                  <div className="da-inline-cards">
                    {msg.agentData.map((d, j) => (
                      <div key={j} className="da-inline-card">
                        <span className="da-ic-name">{d.player}</span>
                        <span className="da-ic-grade">{d.predicted_grade}</span>
                        <span className="da-ic-tier" style={{ color: tierColor(d.predicted_tier) }}>{d.predicted_tier}</span>
                        <span className="da-ic-decision" style={{ color: decisionColor(d.decision) }}>{d.decision}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
            {chatLoading && (
              <div className="da-msg da-msg-gm">
                <div className="da-msg-label">🏈 GM</div>
                <div className="da-msg-text da-typing">Analyzing<span>.</span><span>.</span><span>.</span></div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          <div className="da-chat-input-row">
            <textarea
              className="da-chat-input"
              placeholder={`Ask about any ${posLabel.toLowerCase()}…`}
              value={chatInput}
              onChange={e => setChatInput(e.target.value)}
              onKeyDown={handleChatKey}
              rows={2}
            />
            <button
              className="da-btn da-send-btn"
              style={{ backgroundColor: accentColor }}
              onClick={handleChat}
              disabled={chatLoading}
            >
              Send
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DefenseAnalyst;
