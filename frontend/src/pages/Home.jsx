import { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import './Home.css';

function Home() {
  const [position, setPosition] = useState('All');
  const [team, setTeam] = useState('All');
  const [player, setPlayer] = useState('All');
  const [playerOptions, setPlayerOptions] = useState([]); // New state for players

  const positions = ['QB', 'HB', 'WR', 'TE', 'T', 'G', 'C', 'ED', 'DI', 'LB', 'CB', 'S'];
  const teams = [
    'Seahawks', 'Cardinals', '49ers', 'Rams', 'Cowboys', 'Giants', 'Eagles', 'Commanders',
    'Bears', 'Lions', 'Packers', 'Vikings', 'Falcons', 'Panthers', 'Saints', 'Buccaneers',
    'Bills', 'Dolphins', 'Patriots', 'Jets', 'Ravens', 'Bengals', 'Browns', 'Steelers',
    'Texans', 'Colts', 'Jaguars', 'Titans', 'Broncos', 'Chiefs', 'Raiders', 'Chargers'
  ];

  const navigate = useNavigate();

  // 🆕 Fetch player names when team or position changes
  useEffect(() => {
    const fetchPlayers = async () => {
      if (position === 'All' || team === 'All') {
        setPlayerOptions([]);
        return;
      }

      try {
        const response = await axios.get('http://127.0.0.1:5000/get_pos_team_name', {
          params: {
            pos: position,
            team: team
          }
        });
        setPlayerOptions(response.data);  // Array of names
        setPlayer('All'); // Reset player selection
      } catch (error) {
        console.error('Failed to fetch players:', error);
        setPlayerOptions([]);
      }
    };

    fetchPlayers();
  }, [position, team]);

  const handleSubmit = async () => {
    try {
      let response;
      let val = false
      if (player !== 'All') {
        // Fetch specific player with name, pos, team, and year
        response = await axios.get('http://127.0.0.1:5000/get_player_data', {
          params: {
            player_name: player,
            position: position,
            team: team,
            year: 2024
          }
        });
      } else if (position !== 'All' && team !== 'All') {
        response = await axios.get('http://127.0.0.1:5000/get_pos_team', {
          params: {
            pos: position,
            team: team,
          }
        });
      } else if (position !== 'All') {
        response = await axios.get('http://127.0.0.1:5000/get_pos', {
          params: { pos: position }
        });
      } else if (team !== 'All') {
        response = await axios.get('http://127.0.0.1:5000/get_team', {
          params: { team }
        });
      } else {
        alert("Select one fielld")
        val = true
      }
      if (!val) {
        console.log("Players:", response.data.map(player => player.player));
        navigate('/results', {
          state: {
            players: response.data,
          }
        });
      }
  
    } catch (error) {
      console.error('Error calling the API:', error);
      alert('Error calling the API.');
    }
  };
  
  

  return (
    <div className="container">
      <header>
        <h2>Player Selection</h2>
      </header>

      <div className="form-container">
        <div className="form-group">
          <label>Position:</label>
          <select value={position} onChange={(e) => setPosition(e.target.value)}>
            <option value="All">All</option>
            {positions.map(pos => <option key={pos} value={pos}>{pos}</option>)}
          </select>
        </div>

        <div className="form-group">
          <label>Team:</label>
          <select value={team} onChange={(e) => setTeam(e.target.value)}>
            <option value="All">All</option>
            {teams.map(t => <option key={t} value={t}>{t}</option>)}
          </select>
        </div>

        <div className="form-group">
          <label>Player:</label>
          <select value={player} onChange={(e) => setPlayer(e.target.value)}>
            <option value="All">All</option>
            {playerOptions.map(p => <option key={p} value={p}>{p}</option>)}
          </select>
        </div>

        <button className="submit-button" onClick={handleSubmit}>Submit</button>
      </div>
    </div>
  );
}

export default Home;
