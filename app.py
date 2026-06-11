from flask import Flask, request, jsonify
import numpy as np
import joblib
import re
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)

app = Flask(__name__)

# =========================
# LOAD ARTIFACTS
# =========================

tokenizer = joblib.load("artifacts/tokenizer.joblib")
max_len = joblib.load("artifacts/max_len.joblib")

# =========================
# REBUILD MODEL (IMPORTANT)
# =========================

def build_model():
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, Conv1D, LSTM, Dense

    model = Sequential([
        Embedding(input_dim=10000, output_dim=128, input_length=max_len),
        Conv1D(128, 5, activation='relu'),
        LSTM(64),
        Dense(1, activation='sigmoid')
    ])

    return model


model = build_model()

# ✅ FORCE BUILD (IMPORTANT FIX)
model.build(input_shape=(None, max_len))

# now load weights
model.load_weights("models/model.weights.h5")


# =========================
# TEXT PREPROCESSING
# =========================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(text):
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(
        seq,
        maxlen=max_len,
        padding='post',
        truncating='post'
    )
    return padded


# =========================
# ROUTES
# =========================

@app.route("/")
def home():
    return jsonify({
        "message": "SMS Phishing Detection API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"error": "No message provided"}), 400

    message = data["message"]

    processed = preprocess(message)

    pred = model.predict(processed)[0][0]

    label = "Phishing" if pred > 0.5 else "Legitimate"

    return jsonify({
        "message": message,
        "prediction_score": float(pred),
        "result": label
    })


# =========================
# RUN SERVER
# =========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
