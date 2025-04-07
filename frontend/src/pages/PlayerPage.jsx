import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import './PlayerPages.css';

const PlayerPage = () => {
  const { playerName } = useParams();
  const [playerData, setPlayerData] = useState(null);

  useEffect(() => {
    // Simulate an API request based on playerName
    const fetchPlayerData = async () => {
      // Replace this with actual API call
      const dummyData = {
        name: decodeURIComponent(playerName),
        team: 'Seahawks',
        position: 'QB',
        advancedStats: {
          EPA: '+0.25',
          SuccessRate: '52%',
          AirYardsPerAttempt: '8.1',
        },
      };
      setPlayerData(dummyData);
    };

    fetchPlayerData();
  }, [playerName]);

  if (!playerData) return <p>Loading...</p>;

  return (
    <div className="player-page">
      <h2>{playerData.name} - Advanced Stats</h2>
      <div className="player-info">
        <p><strong>Team:</strong> {playerData.team}</p>
        <p><strong>Position:</strong> {playerData.position}</p>
      </div>
      <div className="advanced-stats">
        <h4>Advanced Stats:</h4>
        <ul>
          <li><strong>EPA:</strong> {playerData.advancedStats.EPA}</li>
          <li><strong>Success Rate:</strong> {playerData.advancedStats.SuccessRate}</li>
          <li><strong>Air Yards/Attempt:</strong> {playerData.advancedStats.AirYardsPerAttempt}</li>
        </ul>
      </div>
    </div>
  );
};

export default PlayerPage;
