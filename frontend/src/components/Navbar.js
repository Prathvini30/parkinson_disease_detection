import React from 'react';
import { Link } from 'react-router-dom';
import './Navbar.css';

const Navbar = () => {
  return (
    <nav className="navbar">
      <div className="navbar-logo"><Link to="/">Parkinson Detect</Link></div>
      <div className="navbar-links">
        <Link to="/">About</Link>
        <Link to="/assessment">Assessment</Link>
      </div>
      <Link to="/assessment">
        <button className="navbar-button">Start Assessment</button>
      </Link>
    </nav>
  );
};

export default Navbar;
