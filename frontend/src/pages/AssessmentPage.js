import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import './AssessmentPage.css';

const AssessmentPage = () => {
  const [step, setStep] = useState(1);
  const [imageFile, setImageFile] = useState(null);
  const [audioBlob, setAudioBlob] = useState(null);
  const [audioFile, setAudioFile] = useState(null); // New state for uploaded audio file
  const [isRecording, setIsRecording] = useState(false);
  const [audioInputMethod, setAudioInputMethod] = useState('record'); // 'record' or 'upload'
  const [questionnaireScores, setQuestionnaireScores] = useState({
    balance: 5,
    sleep: 5,
    muscle_stiffness: 5,
    tremor: 5,
    speech_difficulty: 5,
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const navigate = useNavigate();
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const symptoms = [
    { name: 'balance', label: 'Balance', description: 'Difficulty maintaining balance' },
    { name: 'sleep', label: 'Sleep', description: 'Sleep disturbances' },
    { name: 'muscle_stiffness', label: 'Muscle Stiffness', description: 'Rigidity or resistance when moving joints' },
    { name: 'tremor', label: 'Tremor', description: 'Shaking when your muscles are relaxed' },
    { name: 'speech_difficulty', label: 'Speech Difficulty', description: 'Difficulty speaking clearly' },
  ];

  const handleImageChange = (e) => {
    setImageFile(e.target.files[0]);
  };

  const handleAudioFileChange = (e) => {
    setAudioFile(e.target.files[0]);
    setAudioBlob(null); // Clear recorded audio if user chooses to upload
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };

      mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        setAudioBlob(blob);
        setAudioFile(null); // Clear uploaded file if user chooses to record
        stream.getTracks().forEach(track => track.stop()); // Stop microphone access
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
    } catch (err) {
      setError("Error accessing microphone: " + err.message);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const handleQuestionnaireChange = (symptomName, value) => {
    setQuestionnaireScores(prevScores => ({
      ...prevScores,
      [symptomName]: parseInt(value),
    }));
  };

  const nextStep = () => {
    setStep(step + 1);
  };

  const prevStep = () => {
    setStep(step - 1);
  };

  const handleSubmit = async () => {
    if (!imageFile) {
      setError("Please upload a spiral image.");
      return;
    }

    let finalAudio = null;
    if (audioInputMethod === 'record' && audioBlob) {
      finalAudio = new File([audioBlob], 'recorded_audio.webm', { type: 'audio/webm' });
    } else if (audioInputMethod === 'upload' && audioFile) {
      finalAudio = audioFile;
    } else {
      setError("Please provide an audio input (record or upload).");
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('image', imageFile);
    formData.append('audio', finalAudio);
    formData.append('questionnaire_scores', JSON.stringify(questionnaireScores));

    try {
      const { data } = await axios.post('http://127.0.0.1:5000/predict', formData);
      navigate('/results', { state: { prediction: data.prediction, confidence: data.confidence } });
    } catch (err) {
      setError(err.response?.data?.error || err.message);
    } finally {
      setLoading(false);
    }
  };

  const renderStep = () => {
    switch (step) {
      case 1:
        return (
          <div>
            <h2>Step 1: Spiral Drawing Test</h2>
            <p>Upload a clear image of a spiral you've drawn on paper.</p>
            <div className="upload-area">
              <input type="file" accept="image/*" onChange={handleImageChange} />
              <p>Drop your spiral image here or click to browse</p>
            </div>
            <button onClick={nextStep} disabled={!imageFile}>Next</button>
          </div>
        );
      case 2:
        return (
          <div>
            <h2>Step 2: Voice Analysis</h2>
            <p>Please record yourself saying the following text clearly and naturally:</p>
            <div className="quote">
              <p>"The quick brown fox jumps over the lazy dog. Peter Piper picked a peck of pickled peppers. She sells seashells by the seashore."</p>
            </div>

            <div className="audio-input-options">
              <label>
                <input
                  type="radio"
                  value="record"
                  checked={audioInputMethod === 'record'}
                  onChange={() => setAudioInputMethod('record')}
                />
                Record Voice
              </label>
              <label>
                <input
                  type="radio"
                  value="upload"
                  checked={audioInputMethod === 'upload'}
                  onChange={() => setAudioInputMethod('upload')}
                />
                Upload Audio File
              </label>
            </div>

            {audioInputMethod === 'record' ? (
              <div className="voice-recorder-controls">
                <button onClick={startRecording} disabled={isRecording}>Record</button>
                <button onClick={stopRecording} disabled={!isRecording}>Stop</button>
                {audioBlob && <p>Audio recorded!</p>}
              </div>
            ) : (
              <div className="upload-area">
                <input type="file" accept="audio/*" onChange={handleAudioFileChange} />
                <p>Drop your audio file here or click to browse</p>
              </div>
            )}

            <button onClick={prevStep}>Previous</button>
            <button onClick={nextStep} disabled={!(audioBlob || audioFile)}>Next</button>
          </div>
        );
      case 3:
        return (
          <div>
            <h2>Step 3: Symptom Assessment</h2>
            <p>Please rate each symptom from 1 (none) to 10 (severe) based on your current experience.</p>
            <div className="questionnaire">
              {symptoms.map((symptom) => (
                <div key={symptom.name} className="symptom-item">
                  <h3>{symptom.label}</h3>
                  <p>{symptom.description}</p>
                  <input
                    type="range"
                    min="1"
                    max="10"
                    value={questionnaireScores[symptom.name]}
                    onChange={(e) => handleQuestionnaireChange(symptom.name, e.target.value)}
                  />
                  <span>{questionnaireScores[symptom.name]}</span>
                </div>
              ))}
            </div>
            <button onClick={prevStep}>Previous</button>
            <button onClick={handleSubmit} disabled={loading}>{loading ? 'Analyzing...' : 'Submit'}</button>
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div className="assessment-page">
      <div className="assessment-container">
        <h1>Parkinson's Assessment</h1>
        <div className="progress-bar">
          <div className={`progress-step ${step >= 1 ? 'active' : ''}`}></div>
          <div className={`progress-step ${step >= 2 ? 'active' : ''}`}></div>
          <div className={`progress-step ${step >= 3 ? 'active' : ''}`}></div>
        </div>
        {error && <p className="error-message">{error}</p>}
        {renderStep()}
      </div>
    </div>
  );
};

export default AssessmentPage;
