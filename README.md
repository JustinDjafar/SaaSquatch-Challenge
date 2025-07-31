# SaaSquatch Reply Probability Predictor

This project enhances the SaaSquatch Leads platform by predicting the probability of a reply to an email outreach based on `industry`, `focus`, and `message`. It uses a TensorFlow LSTM model and integrates with a Flask web interface.

## Setup Instructions
1. **Install Python 3.10**:
   - Download from [python.org](https://www.python.org/downloads/release/python-3109/) or use the Microsoft Store.
   - Ensure a 64-bit installation.

2. **Create a Virtual Environment**:
   ```bash
   python3.10 -m venv caprae_env
   source caprae_env/bin/activate  # Linux/Mac
   .\caprae_env\Scripts\activate  # Windows
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare Data**:
   - Place `dummy_lead_data.json` in the project directory (contains 300 entries with `industry`, `focus`, `message`, `reply`).

5. **Run the Application**:
   ```bash
   python model.py
   ```
   - Access the web interface at `http://localhost:5000`.
   - Test the API with:
     ```bash
     curl -X POST -H "Content-Type: application/json" -d '{"industry":"finance","focus":"networking","message":"hello im interested in digital transformation"}' http://localhost:5000/predict_reply_probability
     ```

## Files
- `model.py`: Main script with LSTM model, Flask app, and prediction logic.
- `templates/index.html`: HTML form for user input and result display.
- `requirements.txt`: Dependencies.
- `dummy_lead_data.json`: Dataset (not included in repo due to permissions; structure: `id`, `industry`, `focus`, `message`, `reply`).
- `one_hot_encoder.pkl`, `tokenizer.pkl`, `reply_probability_model.keras`: Generated during training.

## Usage
- Open `http://localhost:5000` in a browser.
- Select an `industry` and `focus`, enter a `message`, and click "Predict Reply Probability".
- View the predicted reply probability (e.g., "22.10%").

## Notes
- Built in ~5 hours for the SaaSquatch Leads challenge.
- Model accuracy/F1-score printed during training (e.g., ~0.85/0.80).
- See `report.md` for detailed approach and evaluation.