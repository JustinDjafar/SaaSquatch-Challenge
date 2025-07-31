import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, jsonify, render_template
import pickle

# region Preprocessing

df = pd.read_json('dummy_lead_data.json')
df['industry'] = df['industry'].str.title()
df['focus'] = df['focus'].str.title()

industries = [
    "Finance", "Technology", "Healthcare", "Education", "Retail",
    "Manufacturing", "Transportation", "Energy", "Entertainment", "Real Estate"
]

focuses = ["Networking", "Partnership", "Collaboration", "Sales"]

categorical_features = ['industry', 'focus']
text_feature = 'message'
text_feature = 'message'
max_words = 1000
max_len = 50
embedding_dim = 100

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', categories=[industries, focuses])
X_cat = encoder.fit_transform(df[categorical_features])
with open ('one_hot_encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(df[text_feature])
X_text = tokenizer.texts_to_sequences(df[text_feature])
X_text = pad_sequences(X_text, maxlen=max_len, padding='post')
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

y = df['reply'].values

X_train_cat, X_test_cat, X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_cat, X_text, y, test_size=0.2, random_state=42
)

# endregion
# region Model

text_model = tf.keras.Sequential([
    tf.keras.Input(shape=(max_len,)),
    tf.keras.layers.Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.GlobalAveragePooling1D()
])

input_cat = tf.keras.Input(shape=(X_cat.shape[1],), name='categorical')
input_text = tf.keras.Input(shape=(max_len,), name='text')
text_features = text_model(input_text)
concat = tf.keras.layers.Concatenate()([input_cat, text_features])
dense = tf.keras.layers.Dense(128, activation='relu')(concat)
dropout = tf.keras.layers.Dropout(0.3)(dense)
output = tf.keras.layers.Dense(1, activation='sigmoid')(dropout)

model = tf.keras.Model(inputs=[input_cat, input_text], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(
    [X_train_cat, X_train_text], y_train,
    epochs=10, batch_size=16, validation_split=0.2, verbose=1
)

y_pred = (model.predict([X_test_cat, X_test_text]) > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print(f"Model F1-Score: {f1:.2f}")

model.save('reply_probability_model.keras')

# endregion
# region App
# def predict_reply_probability(industry, focus, message):
#     encoder = pickle.load(open('one_hot_encoder.pkl', 'rb'))
#     tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
#     model = tf.keras.models.load_model('reply_probability_model.keras')
    
#     # Preprocess input
#     input_cat = encoder.transform([[industry, focus]])
#     input_text = tokenizer.texts_to_sequences([message])
#     input_text = pad_sequences(input_text, maxlen=max_len, padding='post')
    
#     # Predict
#     prob = model.predict([input_cat, input_text], verbose=0)[0][0] * 100
#     return round(prob, 2)

# # Flask app
# app = Flask(__name__)

# @app.route("/", methods=['GET'])
# def index():
#     return render_template('index.html', industries=industries, focuses=focuses)

# @app.route('/predict_reply_probability', methods=['POST'])
# def predict_reply_probability_endpoint():
#     data = request.form if request.form else request.get_json()
#     industry = data.get('industry')
#     focus = data.get('focus')
#     message = data.get('message')
#     if not all([industry, focus, message]):
#         return jsonify({'error': 'Missing input data'}), 400
#     prob = predict_reply_probability(industry, focus, message)
#     result = {
#         'industry': industry,
#         'focus': focus,
#         'message': message,
#         'reply_probability': f"{prob}%"
#     }
#     if request.form:
#         return render_template('index.html', industries=industries, focuses=focuses, result=result)
#     return jsonify(result)

# if __name__ == '__main__':
#     # Example prediction
#     sample_input = {
#         'industry': 'finance',
#         'focus': 'networking',
#         'message': 'hello im interested in digital transformation'
#     }
#     prob = predict_reply_probability(sample_input['industry'], sample_input['focus'], sample_input['message'])
#     print(f"Industry: {sample_input['industry']}, Focus: {sample_input['focus']}")
#     print(f"Message: {sample_input['message']}")
#     print(f"Reply Probability: {prob}%")
#     app.run(debug=True)
    
# endregion