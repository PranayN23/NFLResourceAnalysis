import React, { useEffect, useState } from 'react';
import { useLocation } from 'react-router-dom';
import './PlayerPages.css';

const PlayerPage = () => {
  const location = useLocation();
  const { player } = location.state || {}; // Get player object passed from Results page

  const [playerData, setPlayerData] = useState(null);

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

  if (!playerData) return <p>Loading...</p>;

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
          <li><strong>PFF:</strong> {playerData.grades_offense || 0}</li>
        </ul>
      </div>
      <div className="position-stats">
        <h4>Position Specific Stats</h4>
        <ul>
          <li><strong>Touchdowns:</strong> {playerData.touchdowns}</li>
          <li><strong>Completion Percent:</strong> {playerData.completion_percent}</li>
          <li><strong>QB Rating:</strong> {playerData.qb_rating}</li>
        </ul>
      </div>
    </div>
  );
};

export default PlayerPage;
