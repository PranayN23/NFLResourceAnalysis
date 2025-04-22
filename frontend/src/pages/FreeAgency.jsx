import React from 'react';
import './FreeAgency.css'; // Make sure the path is correct relative to this file

function FreeAgency() {
  return (
    <div className="free-agency-container">
      <h1 className="free-agency-title">Free Agency</h1>
      <form className="free-agency-form">
        <label htmlFor="teamName" className="input-label">
          Team Name
        </label>
        <input
          type="text"
          id="teamName"
          name="teamName"
          className="input-field"
          placeholder="Enter Team Name"
        />

        <label htmlFor="playerName" className="input-label">
          Player Name
        </label>
        <input
          type="text"
          id="playerName"
          name="playerName"
          className="input-field"
          placeholder="Enter Player Name"
        />

        <label htmlFor="compensation" className="input-label">
          Compensation:
        </label>
        <input
          type="text"
          id="compensation"
          name="compensation"
          className="input-field"
          placeholder="Enter Compensation"
        />
      </form>
    </div>
  );
}

export default FreeAgency;
