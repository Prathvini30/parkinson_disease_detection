import React from 'react';
import { Link } from 'react-router-dom';
import './LandingPage.css';

const LandingPage = () => {
  return (
    <div className="landing-page">
      <div className="hero-section">
        <h1>Early Detection of</h1>
        <h2>Advanced AI-powered assessment combining motor skills analysis, voice patterns, and symptom evaluation for early Parkinson's detection.</h2>
        <Link to="/assessment">
          <button className="hero-button">Start Free Assessment</button>
        </Link>
        <p className="hero-subtext">Takes 10-15 minutes • Completely confidential • No medical expertise required</p>
      </div>
      <div className="features-section">
        <div className="feature">
          <h3>AI-Powered Analysis</h3>
          <p>Advanced machine learning models analyze your motor skills and voice patterns</p>
        </div>
        <div className="feature">
          <h3>Multi-Modal Assessment</h3>
          <p>Combines spiral drawing, voice recording, and symptom questionnaire</p>
        </div>
        <div className="feature">
          <h3>Privacy Protected</h3>
          <p>Your data is encrypted and processed securely with complete privacy</p>
        </div>
      </div>
    </div>
  );
};

export default LandingPage;
