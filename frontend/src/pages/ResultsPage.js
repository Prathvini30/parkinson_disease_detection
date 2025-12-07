import React from 'react';
import { useLocation } from 'react-router-dom';
import './ResultsPage.css';

const ResultsPage = () => {
  const location = useLocation();
  const { prediction, confidence } = location.state || {};

  return (
    <div className="results-page">
      <div className="results-container">
        <h1>Assessment Results</h1>
        {prediction ? (
          <div className="results-summary">
            <h2>Your personalized analysis is ready.</h2>
            <div className={`risk-level ${prediction === 'Healthy' ? 'low-risk' : 'high-risk'}`}>
              <h3>{prediction === 'Healthy' ? 'Low Risk Level' : 'High Risk Level'}</h3>
              <p className="confidence-score">{confidence.toFixed(2)}%</p>
              <p>Overall risk assessment</p>
            </div>
            <p className="prediction-sentence">Based on the assessment, the person is **{prediction}**.</p>
            <div className="recommendations">
              <h3>Recommendations</h3>
              <p>This is a preliminary assessment. Please consult a healthcare professional for a comprehensive evaluation.</p>
            </div>
          </div>
        ) : (
          <p>No results to display. Please complete the assessment first.</p>
        )}
      </div>
    </div>
  );
};

export default ResultsPage;
