import React, { useState, useEffect } from 'react';
import './PlayerRanking.css';


const PlayerRanking = () => {
  const [position, setPosition] = useState("QB");
  const [year, setYear] = useState("2021");
  const [snapCounts, setSnapCounts] = useState("0");
    const [minGrade, setMinGrade] = useState("0");      // ⬅️ new state
  const [players, setPlayers] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const fetchPlayers = async () => {
    setLoading(true);
    setError("");
    try {
      const url = `http://127.0.0.1:5000/player_ranking?position=${position}&year=${year}&snap_counts=${snapCounts}&min_grade=${minGrade}`
      const response = await fetch(url);
      if (!response.ok) {
        const errorData = await response.json();
        setError(errorData.error || "Failed to fetch data.");
        setPlayers([]);
      } else {
        const data = await response.json();
        setPlayers(data);
      }
    // eslint-disable-next-line no-unused-vars
    } catch (err) {
      setError("An error occurred while fetching data.");
      setPlayers([]);
    }
    setLoading(false);
  };

  // Optionally, fetch the rankings when the component mounts
  useEffect(() => {
    fetchPlayers();
  }, []);

  return (
    <div style={{ padding: '1rem' }}>
      <h2>Player Rankings</h2>

      <div style={{ marginBottom: '1rem' }}>
        <label style={{ marginRight: '1rem' }}>
          Position:
          <input
            type="text"
            value={position}
            onChange={(e) => setPosition(e.target.value)}
            style={{ marginLeft: '0.5rem' }}
          />
        </label>

        <label style={{ marginRight: '1rem' }}>
          Year:
          <input
            type="number"
            value={year}
            onChange={(e) => setYear(e.target.value)}
            style={{ marginLeft: '0.5rem', width: '80px' }}
          />
        </label>

        <label style={{ marginRight: '1rem' }}>
          Minimum Snap Counts:
          <input
            type="number"
            value={snapCounts}
            onChange={(e) => setSnapCounts(e.target.value)}
            style={{ marginLeft: '0.5rem', width: '80px' }}
          />
        </label>

        <label style={{ marginRight: '1rem' }}>
          Minimum PFF Grade:
          <input
            type="number"
            value={minGrade}
            onChange={(e) => setMinGrade(e.target.value)}   // ⬅️ update state
            style={{ marginLeft: '0.5rem', width: '80px' }}
          />
        </label>

        <button onClick={fetchPlayers}>Fetch Rankings</button>
      </div>

      {loading && <p>Loading...</p>}
      {error && <p style={{ color: 'red' }}>Error: {error}</p>}

      {!loading && !error && players.length > 0 && (
        <table border="1" cellPadding="8" cellSpacing="0">
          <thead>
            <tr>
              <th>Player</th>
              <th>Team</th>
              <th>Year</th>
              <th>Ranking Grade</th>
              <th>Snap Counts</th>
            </tr>
          </thead>
          <tbody>
            {players.map((player) => {
              // Determine which ranking grade field is available.
              const rankingGrade =
                player.grades_offense !== undefined
                  ? player.grades_offense
                  : player.grades_defense !== undefined
                  ? player.grades_defense
                  : "N/A";
              return (
                <tr key={player._id}>
                  <td>{player.player}</td>
                  <td>{player.team}</td>
                  <td>{player.Year}</td>
                  <td>{rankingGrade}</td>
                  <td>{player.snap_counts}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}

      {!loading && !error && players.length === 0 && (
        <p>No players found. Adjust your criteria and try again.</p>
      )}
    </div>
  );
};

export default PlayerRanking;
