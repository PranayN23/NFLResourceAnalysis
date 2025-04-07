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
      <p>Team: {playerData.team}</p>
      <p>Position: {playerData.position}</p>
      <h4>Advanced Stats:</h4>
      <ul>
        <li>EPA: {playerData.advancedStats.EPA}</li>
        <li>Success Rate: {playerData.advancedStats.SuccessRate}</li>
        <li>Air Yards/Attempt: {playerData.advancedStats.AirYardsPerAttempt}</li>
      </ul>
    </div>
  );
};

export default PlayerPage;
