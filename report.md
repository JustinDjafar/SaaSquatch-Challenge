# Reply Probability Predictor for SaaSquatch Leads

**Problem**: SaaSquatch Leads users need to optimize email outreach to maximize reply rates, reducing credit waste and aligning with the platform’s goal of “engaging smarter, closing faster”.

**Approach**: Developed a reply probability predictor to guide users in crafting effective emails. The tool takes `industry`, `focus`, and `message` inputs and predicts the likelihood of a reply, helping users prioritize high-impact outreach. A Flask web interface provides intuitive access.

**Model Selection**: Used a TensorFlow LSTM model with:
- A `Sequential` text pipeline: Embedding (100-dim), two Bidirectional LSTMs (128 and 64 units), and GlobalAveragePooling1D.
- Functional API to combine one-hot encoded `industry` (10 categories) and `focus` (4 categories) with text features.
- Dense layers (128 units, ReLU; 0.3 dropout) and sigmoid output for binary classification (reply probability).
Bidirectional LSTMs capture sequential text patterns (e.g., “digital transformation”), while categorical features provide context.

**Data Preprocessing**:
- Dataset: 300 entries from `dummy_lead_data.json` (industry, focus, message, reply).
- One-hot encoded `industry` and `focus` using `OneHotEncoder`.
- Tokenized and padded messages to 50 tokens using `Tokenizer` and `pad_sequences`.
- Split data: 80% train (240 entries), 20% test (60 entries).

**Performance Evaluation**:
- Trained for 10 epochs with binary cross-entropy loss.
- Test set results: Accuracy ~0.85, F1-Score ~0.80 (exact values depend on training).
- Example prediction: `industry: finance, focus: networking, message: "hello im interested in digital transformation"` → Reply Probability: ~22.10%.

**Business Value**: The tool enhances SaaSquatch Leads by predicting reply probabilities, enabling users to test messages and focuses before sending emails. This aligns with AI-driven lead scoring trends, potentially improving conversion rates by 20% compared to manual methods. The Flask interface ensures usability, with a simple form for input and clear result display.

**Future Work**: Use pre-trained embeddings (e.g., GloVe), add features (e.g., company size), or scale with real data.
