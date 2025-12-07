# Parkinson's Disease Detection

This project is a full-stack web application designed to predict the likelihood of Parkinson's disease based on a combination of user-submitted data: spiral drawings, voice recordings, and a questionnaire about symptoms.

## 🌟 Features

- **Multi-modal Prediction:** Utilizes three different data points for a more comprehensive assessment:
    - **Image Analysis:** Analyzes spiral drawings for motor impairment.
    - **Voice Analysis:** Processes voice recordings to detect vocal tremors and instability.
    - **Symptom Questionnaire:** Incorporates self-reported symptoms for a holistic view.
- **Simple Web Interface:** An easy-to-use frontend built with React to guide the user through the data submission process.
- **Machine Learning Backend:** A Python backend powered by a Random Forest model to process the data and provide a prediction.

## 🛠️ Tech Stack

- **Frontend:** React, JavaScript
- **Backend:** Python, Flask
- **Machine Learning:** Scikit-learn
- **Data Handling:** Pandas, NumPy, Librosa (for audio), Pillow (for images)

## 📂 Project Structure

```
.
├── backend/            # Contains the Flask server, ML model, and prediction logic
│   ├── app.py          # Main Flask application
│   ├── model.py        # (If you have a separate model definition)
│   ├── train_combined_model.py # Script to train the model
│   └── ...
├── frontend/           # Contains the React user interface
│   ├── src/
│   ├── public/
│   └── ...
├── spiral_images/      # Dataset for spiral images
├── voice_samples/      # Dataset for voice recordings
└── .gitignore          # Files and folders to ignore
```

## ⚙️ Setup and Installation

### Prerequisites

- Python 3.8+
- Node.js and npm

### Backend Setup

1.  **Navigate to the backend directory:**
    ```bash
    cd backend
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv .venv
    .venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```
3.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

### Frontend Setup

1.  **Navigate to the frontend directory:**
    ```bash
    cd frontend
    ```
2.  **Install the required npm packages:**
    ```bash
    npm install
    ```

## 🚀 Usage

1.  **Run the Backend Server:**
    From the `backend` directory:
    ```bash
    flask run
    ```
    The server will start on `http://127.0.0.1:5000`.

2.  **Run the Frontend Application:**
    From the `frontend` directory, in a new terminal:
    ```bash
    npm start
    ```
    The application will open in your browser at `http://localhost:3000`.

3.  **Use the Application:**
    -   Upload a spiral drawing.
    -   Record or upload a voice sample.
    -   Fill out the symptom questionnaire.
    -   Click 'Predict' to see the result.

## 📈 Model Performance

The combined model was trained using a Random Forest Classifier and evaluated on a test set. The following metrics were achieved:

- **Accuracy:** 83.33%
- **Precision:** 100.00%
- **Recall:** 66.67%
- **F1-score:** 80.00%

*Note: These metrics are based on the provided dataset and should be interpreted as a proof-of-concept. For a real-world medical application, a much larger and more diverse dataset would be required.*

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improving the model or the application, please feel free to fork the repository and submit a pull request.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details (if you choose to add one).
