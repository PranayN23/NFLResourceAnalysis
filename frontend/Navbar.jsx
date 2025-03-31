// src/Navbar.js
import React from "react";
import { Link, useNavigate } from "react-router-dom";
import { useContext } from "react";
import { GlobalContext } from './contexts/GlobalContext';
import "./Navbar.css";
import logo from "./assets/logo.png";

const Navbar = () => {
  const { user, setUser } = useContext(GlobalContext);
  const navigate = useNavigate();

  const handleLogout = () => {
    setUser(null);
    navigate('/login');
  };

  return (
    <nav className="navbar-container">
      <div className="navbar-logo">
        <Link to="/" className="logo-link">
          <img src={logo} alt="Logo" className="logo-img" />
        </Link>
      </div>

      <div className="navbar-links">
        <Link to="/" className="nav-link">Home</Link>

        {user ? (
          <>
            <Link to="/dashboard" className="nav-link">Dashboard</Link>
            <button onClick={handleLogout} className="nav-link">Logout</button>
          </>
        ) : (
          <>
            <Link to="/login" className="nav-link">Login</Link>
            <Link to="/signup" className="nav-link">Signup</Link>
          </>
        )}
      </div>
    </nav>
  );
};

export default Navbar;
