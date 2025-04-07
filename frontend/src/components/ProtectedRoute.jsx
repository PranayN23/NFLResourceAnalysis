// src/components/ProtectedRoute.js
import React from 'react';
import { Navigate } from 'react-router-dom';
import { useContext } from 'react';
import { GlobalContext } from '../contexts/GlobalContext';

const ProtectedRoute = ({ element }) => {
  const { user } = useContext(GlobalContext);
  
  // If no user is found, redirect to login
  return user ? element : <Navigate to="/login" />;
};

export default ProtectedRoute;
