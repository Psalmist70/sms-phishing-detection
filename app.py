
from flask import Flask, request, jsonify
import numpy as np
import joblib
import re
from tensorflow.keras.models import load_model

app = Flask(__name__)

# =========================
# LOAD MODEL & ARTIFACTS
# =========================

model = load_model("models/sms_phishing_cnn_lstm_model.h5")
tokenizer = joblib.load("artifacts/tokenizer.joblib")
max_len = joblib.load("artifacts/max_len.joblib")


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
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
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

    if "message" not in data:
        return jsonify({"error": "No message provided"}), 400

    message = data["message"]

    processed = preprocess(message)

    prediction = model.predict(processed)[0][0]

    label = "Phishing" if prediction > 0.5 else "Legitimate"

    return jsonify({
        "message": message,
        "prediction_score": float(prediction),
        "result": label
    })


# =========================
# RUN SERVER
# =========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
