import React, { useState, useEffect } from 'react';
import './PlayerRanking.css';

const PlayerRanking = () => {
  const [position, setPosition] = useState("QB");
  const [year, setYear] = useState("2021");
  const [snapCounts, setSnapCounts] = useState("0");
  const [players, setPlayers] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const fetchPlayers = async () => {
    setLoading(true);
    setError("");
    try {
      const url = `http://127.0.0.1:5000/player_ranking?position=${position}&year=${year}&snap_counts=${snapCounts}`;
      const response = await fetch(url);
      if (!response.ok) {
        const errorData = await response.json();
        setError(errorData.error || "Failed to fetch data.");
        setPlayers([]);
      } else {
        const data = await response.json();
        setPlayers(data);
      }
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
    <div className="player-ranking">
      <h2>Player Rankings</h2>

      <div className="filters">
        <label>
          Position:
          <input
            type="text"
            value={position}
            onChange={(e) => setPosition(e.target.value)}
          />
        </label>

        <label>
          Year:
          <input
            type="number"
            value={year}
            onChange={(e) => setYear(e.target.value)}
          />
        </label>

        <label>
          Minimum Snap Counts:
          <input
            type="number"
            value={snapCounts}
            onChange={(e) => setSnapCounts(e.target.value)}
          />
        </label>

        <button onClick={fetchPlayers}>Fetch Rankings</button>
      </div>

      {loading && <p className="loading-message">Loading...</p>}
      {error && <p className="error-message">Error: {error}</p>}

      {!loading && !error && players.length > 0 && (
        <table className="ranking-table">
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
