# SaaSquatch Reply Probability Predictor

This project enhances the **SaaSquatch Leads** platform by predicting the probability of receiving a reply to an email outreach. It uses an LSTM-based TensorFlow model and provides a simple web interface built with Flask.

You can either **train the model from scratch** (Option 1) or **use the pre-trained model files** (Option 2).

> ⚠️ Requires Python 3.10 and TensorFlow 2.19. You can install them manually or use a virtual environment.

---

## 🔧 Setup Instructions

### Option 1: Train the Model from Scratch

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Dataset**
   - Place the file `dummy_lead_data.json` in the project directory.
   - The file should contain 300 entries with the following fields:
     - `industry`
     - `focus`
     - `message`
     - `reply`

3. **Train the Model**
   ```bash
   python model.py
   ```
   - This will train the model and generate:
     - `reply_probability_model.keras`
     - `tokenizer.pkl`
     - `one_hot_encoder.pkl`

4. **Run the Web Application**
   ```bash
   python predictor.py
   ```
   - Open [http://localhost:5000](http://localhost:5000) in your browser to use the predictor.

---

### Option 2: Use Pre-trained Model

1. Ensure the following files exist in the project directory:
   - `reply_probability_model.keras`
   - `tokenizer.pkl`
   - `one_hot_encoder.pkl`

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Web Application**
   ```bash
   python predictor.py
   ```
   - Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## 📁 Project Structure

| File                          | Description                                                |
|-------------------------------|------------------------------------------------------------|
| `model.py`                    | LSTM model training, preprocessing, and evaluation script  |
| `predictor.py`                | Flask web server and prediction logic                      |
| `templates/index.html`        | Web UI for user input and prediction result display        |
| `requirements.txt`            | Python dependencies list                                   |
| `dummy_lead_data.json`        | Sample dataset with 300 entries                            |
| `reply_probability_model.keras` | Pre-trained model file                                   |
| `tokenizer.pkl`               | Tokenizer for processing message text                      |
| `one_hot_encoder.pkl`         | Encoder for industry and focus fields                      |

---

## 🚀 Usage

1. Open your browser and go to [http://localhost:5000](http://localhost:5000).
2. Select an **Industry** and **Focus**, type your **Message**, then click **Predict Reply Probability**.
3. You will see the predicted reply probability (e.g., `22.10%`).

---

## 📌 Notes

- Built in ~5 hours for the SaaSquatch Leads Challenge.
- Training performance:  
  - Accuracy ≈ **0.85**  
  - F1 Score ≈ **0.80**
- For a deeper explanation, refer to `report.md`.

---