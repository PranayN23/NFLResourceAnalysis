import { useState } from 'react'
import axios from 'axios'
import './App.css'

function App() {
  const [position, setPosition] = useState('QB')
  const [team, setTeam] = useState('Seahawks')
  const [player, setPlayer] = useState('Player 1')

  const positions = ['QB', 'RB', 'WR', 'TE', 'T', 'G', 'C', 'ED', 'DI', 'LB', 'CB', 'S']
  const teams = [
    'Seahawks', 'Cardinals', '49ers', 'Rams', 'Cowboys', 'Giants', 'Eagles', 'Commanders',
    'Bears', 'Lions', 'Packers', 'Vikings', 'Falcons', 'Panthers', 'Saints', 'Buccaneers',
    'Bills', 'Dolphins', 'Patriots', 'Jets', 'Ravens', 'Bengals', 'Browns', 'Steelers',
    'Texans', 'Colts', 'Jaguars', 'Titans', 'Broncos', 'Chiefs', 'Raiders', 'Chargers'
  ]
  const players = teams.reduce((acc, team) => {
    acc[team] = ['Player 1', 'Player 2', 'Player 3']
    return acc
  }, {})

  // Function to call the API with selected values
  const handleSubmit = async () => {
    try {
      const response = await axios.post('http://127.0.0.1:5000/get_player_data', {
        position,
        team,
        player
      })
      console.log('Response from server:', response.data)
      alert('API called successfully!')
    } catch (error) {
      console.error('Error calling the API:', error)
      alert('Error calling the API.')
    }
  }

  return (
    <div className="container">
      <nav className="navbar">
        <h1 className="title">ML@Purdue: NFL Player Analysis</h1>
        <a href="/results" className="nav-link">Results</a>
      </nav>
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
  )
}

export default App
