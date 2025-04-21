import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import './Results.css';

const Results = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { players } = location.state || {}; // Added selected filters
  console.log(players);

  // List of offensive positions
  const offensivePositions = ['QB', 'HB', 'WR', 'TE', 'T', 'G', 'C'];

  const handlePlayerClick = (player) => {
    navigate(`/player/${encodeURIComponent(player.player)}`, { state: { player } });
  };

  return (
    <div className="results-container">
      <header>
        <h2>Results Page</h2>
      </header>

      <div className="results-content">
        {players && players.length > 0 ? (
          <table border="1" cellPadding="8" cellSpacing="0">
            <thead>
              <tr>
                <th>Player</th>
                <th>Team</th>
                <th>Position</th>
                <th>Grade</th>
              </tr>
            </thead>
            <tbody>
              {players.map((player, index) => (
                <tr key={index}>
                  <td>
                    <button
                      className="player-button"
                      onClick={() => handlePlayerClick(player)}
                    >
                      {player.player}
                    </button>
                  </td>
                  <td>{player.Team}</td>
                  <td>{player.position}</td>
                  <td>
                    {offensivePositions.includes(player.position)
                      ? player.grades_offense
                      : player.grades_defense}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p>No player data available. Please go back and try again.</p>
        )}
      </div>
    </div>
  );
};

export default Results;
