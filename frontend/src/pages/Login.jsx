import React, { useState, useContext } from "react";
import "./Login.css"; // Import our custom CSS
import { GlobalContext } from '../contexts/GlobalContext'; // Adjust path as needed
import { useNavigate } from 'react-router-dom';


// We define the Login component
const Login = () => {
  // We use React state to store form values
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const { setUser } = useContext(GlobalContext);
  const navigate = useNavigate();

  // This function handles the form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    const formData = { username, password };
    try {
    const response = await fetch('http://127.0.0.1:5000/login', {
        method: 'POST',
        headers: {
        'Content-Type': 'application/json',
        },
        
        body: JSON.stringify(formData),
    });

    if (response.ok) {
        const data = await response.json();
        console.log(data)
        setUser(data);
        navigate('/');
    } else {
        const error = await response.json();
        console.log(error)
        alert(error.message || 'Login failed.');
    }
    } catch (err) {
      console.log(err)
    alert('Login failed');
    }
};

  return (
    <div className="login-container">
      {/* Title */}
      <h1 className="login-title">Sign in to NFL Resource Allocation</h1>

      {/* Form */}
      <form className="login-form" onSubmit={handleSubmit}>
        {/* Username label & input */}
        <label htmlFor="username" className="login-label">
          Username
        </label>
        <input
          id="username"
          type="text"
          className="login-input"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          placeholder="Enter your username"
        />

        {/* Password label & input row (with "Forgot Password?") */}
        <div className="password-label-row">
          <label htmlFor="password" className="login-label">
            Password
          </label>
          {/* The "Forgot Password?" link */}
          <a href="/forgot-password" className="forgot-password-link">
            Forgot Password?
          </a>
        </div>
        <input
          id="password"
          type="password"
          className="login-input"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="Enter your password"
        />

        {/* Sign In button */}
        <button type="submit" className="login-button">
          Sign In
        </button>
      </form>
    </div>
  );
};

export default Login;
