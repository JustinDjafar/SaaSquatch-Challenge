import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, jsonify, render_template
import sys


# region Predictor
industries = [
    "Finance", "Technology", "Healthcare", "Education", "Retail",
    "Manufacturing", "Transportation", "Energy", "Entertainment", "Real Estate"
]

focuses = ["Networking", "Partnership", "Collaboration", "Sales"]
max_len = 30

encoder = pickle.load(open('../src/one_hot_encoder.pkl', 'rb'))
tokenizer = pickle.load(open('../src/tokenizer.pkl', 'rb'))
model = tf.keras.models.load_model('../src/reply_probability_model.keras')

def predict_reply_probability(industry, focus, message):
    input_cat = encoder.transform([[industry, focus]])
    input_text = tokenizer.texts_to_sequences([message])
    input_text = pad_sequences(input_text, maxlen=max_len, padding='post')

    prob = model.predict([input_cat, input_text], verbose=0)[0][0] * 100
    prob = round(prob, 2)
    return prob

# endregion
# region App

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', industries=industries, focuses=focuses)

@app.route('/predict_reply_probability', methods=['POST'])
def predict_reply_probability_endpoint():
    industry = request.form.get('industry')
    focus = request.form.get('focus')
    message = request.form.get('message')
    if not industry or not focus or not message:
        return render_template('index.html', industry=industry, industries=industries, focus=focus, focuses=focuses, message=message, error="Please fill all fields.")
    
    prob = predict_reply_probability(industry, focus, message)
    result = {
        'reply_probability': f"{prob:.2f} %"
    }
    return render_template('index.html', industry=industry, industries=industries, focus=focus, focuses=focuses, message=message, result=result)

if __name__ == '__main__':
    app.run(debug=True)

# endregion