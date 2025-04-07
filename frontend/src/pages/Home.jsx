import { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom'; // Import useNavigate
import './Home.css';

function App() {
  const [position, setPosition] = useState('QB');
  const [team, setTeam] = useState('Seahawks');
  const [player, setPlayer] = useState('Player 1');

  const positions = ['QB', 'RB', 'WR', 'TE', 'T', 'G', 'C', 'ED', 'DI', 'LB', 'CB', 'S'];
  const teams = [
    'Seahawks', 'Cardinals', '49ers', 'Rams', 'Cowboys', 'Giants', 'Eagles', 'Commanders',
    'Bears', 'Lions', 'Packers', 'Vikings', 'Falcons', 'Panthers', 'Saints', 'Buccaneers',
    'Bills', 'Dolphins', 'Patriots', 'Jets', 'Ravens', 'Bengals', 'Browns', 'Steelers',
    'Texans', 'Colts', 'Jaguars', 'Titans', 'Broncos', 'Chiefs', 'Raiders', 'Chargers'
  ];
  const players = teams.reduce((acc, team) => {
    acc[team] = ['Player 1', 'Player 2', 'Player 3'];
    return acc;
  }, {});

  const navigate = useNavigate(); // Initialize navigate function

  // Function to call the API with selected values
  const handleSubmit = async () => {
    try {
      const response = await axios.post('http://127.0.0.1:5000/get_player_data', {
        position,
        team,
        player
      });
      console.log('Response from server:', response.data);
      alert('API called successfully!');

      // Navigate to the results page after successful submission
      navigate('/results'); // Adjust the path as necessary
    } catch (error) {
      console.error('Error calling the API:', error);
      alert('Error calling the API.');
      navigate('/results'); // Adjust the path as necessary

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
            {positions.map(pos => <option key={pos} value={pos}>{pos}</option>)}
          </select>
        </div>

        <div className="form-group">
          <label>Team:</label>
          <select value={team} onChange={(e) => setTeam(e.target.value)}>
            {teams.map(t => <option key={t} value={t}>{t}</option>)}
          </select>
        </div>

        <div className="form-group">
          <label>Player:</label>
          <select value={player} onChange={(e) => setPlayer(e.target.value)}>
            {players[team].map(p => <option key={p} value={p}>{p}</option>)}
          </select>
        </div>

        <button className="submit-button" onClick={handleSubmit}>Submit</button>
      </div>
    </div>
  );
}

export default App;
