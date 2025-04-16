import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import './PlayerPages.css';

const PlayerPage = () => {
  //const { playerName } = useParams(); //use this for actual web
  const playerName = 'James Washington' //use this for testing

  const [playerData, setPlayerData] = useState(null);
  const [team, setTeam] = useState('');
  const [position, setPosition] = useState('');
  const [year, setYear] = useState(' ');

  useEffect(() => {
    // Simulate an API request based on playerName
    const fetchPlayerData = async () => {
      const res1 = await fetch(`http://127.0.0.1:5000/get_player_year_pos_team?player_name=${playerName}`);
      const yearData = await res1.json();

      const years = Object.keys(yearData.years).sort((a, b) => b - a); // descending order
        for (let year of years) {
          const entry = yearData.years[year];
          const team = entry.teams?.[0];
          const position = entry.positions?.[0];

          setTeam(team);
          setPosition(position);
          setYear(year);
          
          const params = new URLSearchParams({
            player_name: playerName,
            team,
            position,
            year
          });

        fetch(`http://127.0.0.1:5000/get_player_data?${params.toString()}`)
          .then((res) => res.json())
          .then((data) => setPlayerData(data[0]))
          .catch((err) => console.error('Fetch error:', err));
        break;
      };
  }

    fetchPlayerData();
  }, [playerName]);

  if (!playerData) return <p>Loading...</p>;

  return (
    <div className="player-page">
      <h2>{playerName}</h2>
      <div className="player-info">
        <p><strong>Team:</strong> {team}</p>
        <p><strong>Position:</strong> {position}</p>
        <p><strong>Year:</strong> {year}</p>
      </div>
      <div className="general-stats">
        <h4>General Stats</h4>
        <ul>
          <li><strong>PFF:</strong> 0</li>
          <li><strong>Cap Space:</strong> 0</li>
          <li><strong>Draft Capital:</strong> 0</li>
        </ul>
      </div>
      <div className="position-stats">
        <h4>Position Specific Stats</h4>
        {position == 'C' && (
          <ul>
            <li><strong>Grades Pass Block:</strong> {playerData.grades_pass_block}</li>
            <li><strong>Grades Run Block:</strong> {playerData.grades_run_block}</li>
            <li><strong>Sacks Allowed:</strong> {playerData.sacks_allowed}</li>
            <li><strong>Pressures Allowed:</strong> {playerData.pressures_allowed}</li>
          </ul>
        )}

        {position == 'CB' && (
          <ul>
            <li><strong>QB Rating Against:</strong> {playerData.qb_rating_against}</li>
            <li><strong>Coverage Grade:</strong> {playerData.grades_coverage_defense}</li>
            <li><strong>Pass Breakups:</strong> {playerData.pass_break_ups}</li>
            <li><strong>Targets:</strong> {playerData.targets}</li>
          </ul>
        )}

        {position == 'DI' && (
          <ul>
            <li><strong>Overall Defense Grade:</strong> {playerData.grades_defense}</li>
            <li><strong>Sacks:</strong> {playerData.sacks}</li>
            <li><strong>Total Pressures:</strong> {playerData.total_pressures}</li>
            <li><strong>Tackles for Loss:</strong> {playerData.tackles_for_loss}</li>
          </ul>
        )}

        {position == 'ED' && (
          <ul>
            <li><strong>Sacks:</strong> {playerData.sacks}</li>
            <li><strong>Total Pressures:</strong> {playerData.total_pressures}</li>
            <li><strong>Run Defense Grade:</strong> {playerData.grades_run_defense}</li>
            <li><strong>Stops:</strong> {playerData.stops}</li>
          </ul>
        )}

        {position == 'LB' && (
          <ul>
            <li><strong>Overall Defense Grade:</strong> {playerData.grades_defense}</li>
            <li><strong>Sacks:</strong> {playerData.sacks}</li>
            <li><strong>Tackles for Loss:</strong> {playerData.tackles_for_loss}</li>
            <li><strong>Interceptions:</strong> {playerData.interception}</li>
          </ul>
        )}

        {position == 'QB' && (
          <ul>
            <li><strong>Touchdowns:</strong> {playerData.touchdowns}</li>
            <li><strong>Completion Percent:</strong> {playerData.completion_percent}</li>
            <li><strong>Sack Percent:</strong> {playerData.sack_percent}</li>
            <li><strong>Yards per Attempt:</strong> {playerData.ypa}</li>
            <li><strong>QB Rating:</strong> {playerData.qb_rating}</li>
          </ul>
        )}

        {position == 'T' && (
          <ul>
            <li><strong>Pass Block Grade:</strong> {playerData.grades_pass_block}</li>
            <li><strong>Run Block Grade:</strong> {playerData.grades_run_block}</li>
            <li><strong>Sacks Allowed:</strong> {playerData.sacks_allowed}</li>
            <li><strong>Pressures Allowed:</strong> {playerData.pressures_allowed}</li>
          </ul>
        )}

        {position == 'G' && (
          <ul>
            <li><strong>Pass Block Grade:</strong> {playerData.grades_pass_block}</li>
            <li><strong>Run Block Grade:</strong> {playerData.grades_run_block}</li>
            <li><strong>Sacks Allowed:</strong> {playerData.sacks_allowed}</li>
            <li><strong>Pressures Allowed:</strong> {playerData.pressures_allowed}</li>
          </ul>
        )}

        {position == 'S' && (
          <ul>
            <li><strong>Missed Tackle Rate:</strong> {playerData.missed_tackle_rate}</li>
            <li><strong>Fumble Recoveries:</strong> {playerData.fumble_recoveries}</li>
            <li><strong>Interception Touchdowns:</strong> {playerData.interception_touchdown}</li>
            <li><strong>Receptions Allowed:</strong> {playerData.receptions}</li>
            <li><strong>Forced Fumbles:</strong> {playerData.forced_fumbles}</li>
          </ul>
        )}

        {position == 'TE' && (
          <ul>
            <li><strong>Receptions:</strong> {playerData.receptions}</li>
            <li><strong>Touchdowns:</strong> {playerData.touchdowns}</li>
            <li><strong>Yards After Catch:</strong> {playerData.yards_after_catch}</li>
            <li><strong>Yards Per Reception:</strong> {playerData.yards_per_reception}</li>
          </ul>
        )}

        {position == 'WR' && (
          <ul>
            <li><strong>Receptions:</strong> {playerData.receptions}</li>
            <li><strong>Touchdowns:</strong> {playerData.touchdowns}</li>
            <li><strong>Yards:</strong> {playerData.yards}</li>
            <li><strong>Yards After Catch per Reception:</strong> {playerData.yards_after_catch_per_reception}</li>
          </ul>
        )}

        {position == 'HB' && (
          <ul>
            <li><strong>Attempts:</strong> {playerData.attempts}</li>
            <li><strong>Yards:</strong> {playerData.yards}</li>
            <li><strong>Touchdowns:</strong> {playerData.touchdowns}</li>
            <li><strong>Yards After Contact:</strong> {playerData.yards_after_contact}</li>
          </ul>
        )}
      </div>
    </div>
  );
};

export default PlayerPage;
