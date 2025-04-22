import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './ComparePlayers.css';

const positions = ['QB', 'HB', 'WR', 'TE', 'T', 'G', 'C', 'ED', 'DI', 'LB', 'CB', 'S'];
const teams = [
  'Seahawks', 'Cardinals', '49ers', 'Rams', 'Cowboys', 'Giants', 'Eagles', 'Commanders',
  'Bears', 'Lions', 'Packers', 'Vikings', 'Falcons', 'Panthers', 'Saints', 'Buccaneers',
  'Bills', 'Dolphins', 'Patriots', 'Jets', 'Ravens', 'Bengals', 'Browns', 'Steelers',
  'Texans', 'Colts', 'Jaguars', 'Titans', 'Broncos', 'Chiefs', 'Raiders', 'Chargers'
];

const ComparePlayers = () => {
  const [player1, setPlayer1] = useState({ position: 'All', team: 'All', name: 'All', options: [] });
  const [player2, setPlayer2] = useState({ position: 'All', team: 'All', name: 'All', options: [] });

  const [data1, setData1] = useState(null);
  const [data2, setData2] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  // Generic function to fetch options from API
  const fetchPlayerOptions = async (position, team, setPlayer) => {
    if (position === 'All' || team === 'All') {
      setPlayer(prev => ({ ...prev, options: [], name: 'All' }));
      return;
    }
    try {
      const res = await axios.get('http://127.0.0.1:5000/get_pos_team_name', {
        params: { pos: position, team }
      });
      setPlayer(prev => ({ ...prev, options: res.data, name: 'All' }));
    } catch (err) {
      console.error('Failed to fetch players:', err);
      setPlayer(prev => ({ ...prev, options: [], name: 'All' }));
    }
  };

  // Triggers when player1 position/team change
  useEffect(() => {
    fetchPlayerOptions(player1.position, player1.team, setPlayer1);
  }, [player1.position, player1.team]);

  // Triggers when player2 position/team change
  useEffect(() => {
    fetchPlayerOptions(player2.position, player2.team, setPlayer2);
  }, [player2.position, player2.team]);

  const fetchPlayerData = async (player) => {
    const res = await axios.get('http://127.0.0.1:5000/get_player_data', {
      params: {
        player_name: player.name,
        position: player.position,
        team: player.team,
        year: 2024
      }
    });
    return res.data[0]; // Assuming single result
  };

  const handleCompare = async () => {
    if (player1.name === 'All' || player2.name === 'All') {
      setError("Please select both players.");
      return;
    }

    try {
      setLoading(true);
      setError('');
      const [dataOne, dataTwo] = await Promise.all([
        fetchPlayerData(player1),
        fetchPlayerData(player2)
      ]);
      setData1(dataOne);
      setData2(dataTwo);
    } catch (err) {
      console.error('Failed to fetch player data:', err);
      setError("Failed to fetch player data.");
    } finally {
      setLoading(false);
    }
  };

  const renderDropdowns = (player, setPlayer, label) => (
    <div className="player-form">
      <h4>{label}</h4>
      <select value={player.position} onChange={(e) => setPlayer(p => ({ ...p, position: e.target.value }))}>
        <option value="All">Position</option>
        {positions.map(pos => <option key={pos} value={pos}>{pos}</option>)}
      </select>

      <select value={player.team} onChange={(e) => setPlayer(p => ({ ...p, team: e.target.value }))}>
        <option value="All">Team</option>
        {teams.map(team => <option key={team} value={team}>{team}</option>)}
      </select>

      <select value={player.name} onChange={(e) => setPlayer(p => ({ ...p, name: e.target.value }))}>
        <option value="All">Player</option>
        {player.options.map(name => <option key={name} value={name}>{name}</option>)}
      </select>
    </div>
  );

  const renderStats = (data) => (
    <div className="player-box">
      <h3>{data.player}</h3>
      <ul>
        {Object.entries(data).map(([key, value]) =>
          (typeof value === 'string' || typeof value === 'number') && (
            <li key={key}><strong>{key.replace(/_/g, ' ')}:</strong> {value}</li>
          )
        )}
      </ul>
    </div>
  );

  return (
    <div className="compare-container">
      <h2>Compare Two Players</h2>

      <div className="compare-form-row">
        {renderDropdowns(player1, setPlayer1, 'Player 1')}
        {renderDropdowns(player2, setPlayer2, 'Player 2')}
      </div>

      <button className="compare-button" onClick={handleCompare}>Compare</button>

      {error && <p className="error">{error}</p>}
      {loading && <p>Loading...</p>}

      {(data1 && data2) && (
        <div className="comparison-result">
          {renderStats(data1)}
          {renderStats(data2)}
        </div>
      )}
    </div>
  );
};

export default ComparePlayers;
