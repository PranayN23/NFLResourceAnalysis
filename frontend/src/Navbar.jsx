import { Link } from "react-router-dom";
import './Navbar.css';
import mlLogo from './assets/mlpurdue-logo.png';
import pffLogo from './assets/pff-logo.jpeg';

const Navbar = () => {
    return (
        <div id="navbar-container">
            <div id="navbar-left">
                <img src={mlLogo} alt="ML Purdue Logo" className="navbar-logo" />
                <img src={pffLogo} alt="PFF Logo" className="navbar-logo" />
                <div id="navbar-title">ML @Purdue: NFL Resource Allocation</div>
            </div>

            <div id="navbar-links">
                <Link to="/">Home</Link>
                <Link to="/login">Login</Link>
                <Link to="/signup">Signup</Link>
                <Link to="/results">Results</Link>
                <Link to="/player-rankings">Player Rankings</Link>
<<<<<<< HEAD
                <Link to="/draft">Draft/Rookies</Link>
=======
                <Link to="/free-agency">Free Agency</Link>
>>>>>>> 9453fd46515d09d91ef3f4c0445758af9dc3b37d
            </div>
        </div>
    );
}; 

export default Navbar;