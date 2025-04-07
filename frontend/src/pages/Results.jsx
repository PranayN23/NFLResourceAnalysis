import React from 'react';
import './Results.css'; // Add styles if needed

const Results = () => {
  return (
    <div className="results-container">
      <header>
        <h2>Results Page</h2>
      </header>
      
      <div className="results-content">
        <p>This is the results page. Display relevant player data here after fetching it from the API.</p>
        <p>Data will appear here once the form is submitted successfully!</p>
      </div>
      
      <div className="dummy-results">
        <h3>Dummy Player Data</h3>
        <ul>
          <li>Player: Player 1</li>
          <li>Team: Seahawks</li>
          <li>Position: QB</li>
          <li>Statistics: Placeholder data for now.</li>
        </ul>
      </div>
    </div>
  );
};

export default Results;
