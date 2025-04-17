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
          players.map((player, index) => (
            <div key={index} className="player-card">
              <p>
                Player:{' '}
                <button
                  className="player-button"
                  onClick={() => handlePlayerClick(player)}
                >
                  {player.player}
                </button>
              </p>
              <p>Team: {player.Team}</p>
              <p>Position: {player.position}</p>
              <p>
                Grade: {
                  offensivePositions.includes(player.position)
                    ? player.grades_offense
                    : player.grades_defense
                }
              </p>
            </div>
          ))
        ) : (
          <p>No player data available. Please go back and try again.</p>
        )}
      </div>
    </div>
  );
};

export default Results;
