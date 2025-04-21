import React, { useEffect, useState } from 'react';
import { useLocation } from 'react-router-dom';
import './PlayerPages.css';

//install recharts
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

//dummy data rn
const data = [
    { yearIndex: 1, playerPFF: 68, avgPFF: 72 },
    { yearIndex: 2, playerPFF: 70, avgPFF: 74 },
    { yearIndex: 3, playerPFF: 73, avgPFF: 75 },
    { yearIndex: 4, playerPFF: 76, avgPFF: 76 },
    { yearIndex: 5, playerPFF: 79, avgPFF: 77 },
  ];

const YearLineChart = () => (
  <ResponsiveContainer width="50%" height={300}>
    <LineChart data={data}>
      <CartesianGrid strokeDasharray="3 3" />
      <XAxis 
        label={{ value: "Year of Experience",  position: "insideBottom", dy: 6.8 }}
      />
      <YAxis
        label={{ value: "PFF Score", angle: -90, position: "insideLeft" }}
      />
      <Tooltip />
      <Legend verticalAlign="bottom" />
      <Line
        type="monotone"
        dataKey="playerPFF"
        stroke="#8884d8"
        name="Player PFF"
      />
      <Line
        type="monotone"
        dataKey="avgPFF"
        stroke="#82ca9d"
        name="Average PFF"
      />
    </LineChart>
  </ResponsiveContainer>
);

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

  // Get position specific fields (3 stats per position)
  const positionStats = positionFields[player.position] || [];

  return (
    <div className="player-page">
      <h2>{player.player}</h2>
      <div className="player-info">
        <p><strong>Team:</strong> {player.Team}</p>
        <p><strong>Position:</strong> {player.position}</p>
      </div>

      <YearLineChart />
      
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
    </div>
  );
};

export default PlayerPage;