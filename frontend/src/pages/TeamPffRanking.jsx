import React, { useState, useEffect } from "react";
import "./PlayerRanking.css";          // reuse the same stylesheet

const TeamPffRanking = () => {
  /* ---------------- state ---------------- */
  const [team, setTeam] = useState("Seahawks");     // «CHANGED»
  const [year, setYear] = useState("2021");
  const [snapCounts, setSnapCounts] = useState("0");
  const [players, setPlayers] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  /* -------------- data fetch ------------- */
  const fetchPlayers = async () => {
    setLoading(true);
    setError("");
    try {
      // «CHANGED»  call the NEW endpoint
      const url = `http://127.0.0.1:5000/team_pff?team=${team}&year=${year}&snap_counts=${snapCounts}`;
      const res = await fetch(url);

      if (!res.ok) {
        const err = await res.json();
        setError(err.error || "Failed to fetch data.");
        setPlayers([]);
      } else {
        setPlayers(await res.json());
      }
    } catch (e) {
      setError("An error occurred while fetching data.");
      setPlayers([]);
    }
    setLoading(false);
  };

  useEffect(() => { fetchPlayers(); }, []); // optional auto‑load

  /* -------------- render ------------------ */
  return (
    <div style={{ padding: "1rem" }}>
      <h2>Team Roster – sorted by PFF score</h2>

      {/* ---- controls ---- */}
      <div style={{ marginBottom: "1rem" }}>
        <label style={{ marginRight: "1rem" }}>
          Team&nbsp;
          <input
            type="text"
            value={team}
            onChange={(e) => setTeam(e.target.value)}
            style={{ marginLeft: "0.5rem", width: "60px" }}
          />
        </label>

        <label style={{ marginRight: "1rem" }}>
          Year&nbsp;
          <input
            type="number"
            value={year}
            onChange={(e) => setYear(e.target.value)}
            style={{ marginLeft: "0.5rem", width: "80px" }}
          />
        </label>

        <label style={{ marginRight: "1rem" }}>
          Min Snaps&nbsp;
          <input
            type="number"
            value={snapCounts}
            onChange={(e) => setSnapCounts(e.target.value)}
            style={{ marginLeft: "0.5rem", width: "80px" }}
          />
        </label>

        <button onClick={fetchPlayers}>Fetch Roster</button>
      </div>

      {/* ---- feedback ---- */}
      {loading && <p>Loading…</p>}
      {error && <p style={{ color: "red" }}>Error: {error}</p>}

      {/* ---- table ---- */}
      {!loading && !error && players.length > 0 && (
        <table border="1" cellPadding="8" cellSpacing="0">
          <thead>
            <tr>
              <th>Player</th>
              <th>Position</th>
              <th>PFF Score</th>   {/* «CHANGED» header */}
              <th>Snap Counts</th>
            </tr>
          </thead>
          <tbody>
            {players.map((p) => (
              <tr key={p._id}>
                <td>{p.player}</td>
                <td>{p.position}</td>
                <td>{p.pff_score}</td> {/* «CHANGED» field */}
                <td>{p.snap_counts}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {!loading && !error && players.length === 0 && (
        <p>No players found. Adjust your criteria and try again.</p>
      )}
    </div>
  );
};

export default TeamPffRanking;