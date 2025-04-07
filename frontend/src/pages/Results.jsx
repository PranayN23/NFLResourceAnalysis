import React from 'react';
import { useNavigate } from 'react-router-dom';
import './Results.css';

const Results = () => {
  const navigate = useNavigate();

  const handlePlayerClick = (playerName) => {
    navigate(`/player/${encodeURIComponent(playerName)}`);
  };

  return (
    <div className="results-container">
      <header>
        <h2>Results Page</h2>
      </header>

      <div className="results-content">
        <p>This is the results page. Display relevant player data here after fetching it from the API.</p>
        <p>Data will appear here once the form is submitted successfully!</p>
      </div>

      <div className="dummy-results">
        <h3>Dummy Player Data</h3>
        <ul>
          <li>
            Player:{' '}
            <button
              className="player-button"
              onClick={() => handlePlayerClick('Player 1')}
            >
              Player 1
            </button>
          </li>
          <li>Team: Seahawks</li>
          <li>Position: QB</li>
          <li>Statistics: Placeholder data for now.</li>
        </ul>
      </div>
    </div>
  );
};

export default Results;
