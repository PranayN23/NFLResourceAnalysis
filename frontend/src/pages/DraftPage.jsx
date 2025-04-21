import React, { useState } from 'react';
import './DraftPage.css';

const dummyProspects = [
  { name: 'John Doe', team: 'Tigers', position: 'QB', experience: 0 },
  { name: 'Alex Smith', team: 'Seahawks', position: 'WR', experience: 0 },
  { name: 'Marcus Jones', team: 'Tigers', position: 'QB', experience: 0 },
  { name: 'Liam Carter', team: 'Panthers', position: 'RB', experience: 0 },
];

const DraftPage = () => {
  const [team, setTeam] = useState('');
  const [position, setPosition] = useState('');
  const [prospects, setProspects] = useState([]);
  const [showDropdown, setShowDropdown] = useState(false);

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      const filtered = dummyProspects.filter(p =>
        p.team.toLowerCase() === team.toLowerCase() &&
        p.position.toLowerCase() === position.toLowerCase() &&
        p.experience === 0
      );
      setProspects(filtered);
      setShowDropdown(true);
    }
  };

  return (
    <div className="draft-page">
      <h2>Draft Prospect Finder</h2>
      <div className="input-group">
        <label>Team:</label>
        <input
          type="text"
          value={team}
          onChange={(e) => setTeam(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Enter team"
        />
      </div>
      <div className="input-group">
        <label>Position:</label>
        <input
          type="text"
          value={position}
          onChange={(e) => setPosition(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Enter position (e.g. QB)"
        />
      </div>
      <div className="input-group">
        <label>Years of Experience:</label>
        <input type="number" value="0" disabled />
      </div>

      {showDropdown && (
        <div className="dropdown">
          <h4>Matching Draft Prospects</h4>
          {prospects.length > 0 ? (
            <ul>
              {prospects.map((prospect, index) => (
                <li key={index}>{prospect.name} ({prospect.position}, {prospect.team})</li>
              ))}
            </ul>
          ) : (
            <p>No matching prospects found.</p>
          )}
        </div>
      )}
    </div>
  );
};

export default DraftPage;