import React, { useEffect, useState } from 'react';
import { useLocation } from 'react-router-dom';
import './PlayerPages.css';

const positionFields = {
  "S": [
    "grades_defense", "grades_coverage_defense", "tackles"
  ],
  "CB": [
    "grades_defense", "interceptions", "pass_break_ups"
  ],
  "DI": [
    "grades_defense", "sacks", "total_pressures"
  ],
  "ED": [
    "grades_defense", "sacks", "total_pressures"
  ],
  "LB": [
    "grades_defense", "interceptions", "sacks"
  ],
  "QB": [
    "qb_rating", "touchdowns", "yards"
  ],
  "T": [
    "grades_pass_block", "pressures_allowed", "sacks_allowed"
  ],
  "G": [
    "grades_pass_block", "pressures_allowed", "sacks_allowed"
  ],
  "C": [
    "grades_pass_block", "pressures_allowed", "sacks_allowed"
  ],
  "TE": [
    "receptions", "touchdowns", "yards_after_catch"
  ],
  "WR": [
    "receptions", "touchdowns", "yards"
  ],
  "HB": [
    "attempts", "touchdowns", "yards"
  ]
};

const PlayerPage = () => {
  const location = useLocation();
  const { player } = location.state || {}; // Get player object passed from Results page

  const [playerData, setPlayerData] = useState(null);
  const [capSpace, setCapSpace] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!player) return;

    // Simulate an API request based on player data
    const fetchPlayerData = async () => {
      const params = new URLSearchParams({
        player_name: player.player,
        team: player.Team,
        position: player.position,
        year: "2024", // Default year, can be dynamic as needed
      });

      try {
        const res = await fetch(`http://127.0.0.1:5000/get_player_data?${params.toString()}`);
        const data = await res.json();

        if (data.error) {
          console.error(data.error);
          return;
        }

        setPlayerData(data[0] || null);
      } catch (err) {
        console.error("Fetch error:", err);
      }
    };

    fetchPlayerData();
  }, [player]); // Re-fetch if the player changes

  const handlePredict = async () => {
    if (!capSpace || isNaN(capSpace)) {
      setError('Please enter a valid cap space percentage');
      return;
    }

    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const res = await fetch('http://127.0.0.1:5000/predict_player_group', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          player_name: player.player,
          projected_cap: parseFloat(capSpace),
        }),
      });

      const data = await res.json();

      if (data.error) {
        setError(data.error);
        return;
      }

      setPrediction(data);
    } catch (err) {
      console.error('Prediction error:', err);
      setError('Failed to fetch prediction');
    } finally {
      setLoading(false);
    }
  };

  if (!playerData) return <p>Loading...</p>;

  // Get position specific fields (3 stats per position)
  const positionStats = positionFields[player.position] || [];

  return (
    <div className="player-page">
      <h2>{player.player}</h2>
      <div className="player-info">
        <p><strong>Team:</strong> {player.Team}</p>
        <p><strong>Position:</strong> {player.position}</p>
      </div>
      
      <div className="general-stats">
        <h4>General Stats</h4>
        <ul>
          {positionStats.map(field => (
            <li key={field}>
              <strong>
                {field
                  .replace(/_/g, ' ') // Replace underscores with spaces
                  .split(' ') // Split the string into words
                  .map((word) =>
                    ['QB', 'HB', 'WR', 'TE', 'S', 'CB', 'DI', 'ED', 'LB', 'T', 'G', 'C'].includes(word.toUpperCase()) // Check if it's a position abbreviation
                      ? word.toUpperCase() // Keep it uppercase
                      : word.charAt(0).toUpperCase() + word.slice(1).toLowerCase() // Capitalize other words
                  )
                  .join(' ')}:
              </strong> {playerData[field] || 0}            
            </li>
          ))}
        </ul>
      </div>

      {player.position === 'QB' && (
        <div className="prediction-section">
          <h4>Performance Prediction</h4>
          <div className="cap-space-input">
            <label htmlFor="capSpace">Cap Space %: </label>
            <input
              id="capSpace"
              type="number"
              step="0.01"
              value={capSpace}
              onChange={(e) => setCapSpace(e.target.value)}
              placeholder="e.g., 15.5"
            />
            <button onClick={handlePredict} disabled={loading}>
              {loading ? 'Predicting...' : 'Predict'}
            </button>
          </div>

          {error && <p className="error-message">{error}</p>}

          {prediction && (
            <div className="prediction-results">
              <h5>Prediction Results:</h5>
              <p><strong>Predicted PFF Grade:</strong> {prediction.predicted_pff.toFixed(2)}</p>
              <p><strong>Performance Tier:</strong> {prediction.group}</p>
              <p><strong>Projected Cap Space:</strong> {prediction.projected_cap}%</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default PlayerPage;