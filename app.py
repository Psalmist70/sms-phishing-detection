import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

print("STEP 1 - Starting app")

from flask import Flask, request, jsonify
print("STEP 2 - Flask imported")

import numpy as np
print("STEP 3 - NumPy imported")

import joblib
print("STEP 4 - Joblib imported")

import re

import tensorflow as tf
print("STEP 5 - TensorFlow imported")

import keras
print("STEP 6 - Keras imported")

from tensorflow.keras.preprocessing.sequence import pad_sequences
print("STEP 7 - pad_sequences imported")

print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)

app = Flask(__name__)

# =========================
# LOAD ARTIFACTS
# =========================

print("STEP 8 - Loading tokenizer")
tokenizer = joblib.load("artifacts/tokenizer.joblib")

print("STEP 9 - Loading max_len")
max_len = joblib.load("artifacts/max_len.joblib")

print("max_len =", max_len)

# =========================
# REBUILD MODEL
# =========================

def build_model():
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Embedding,
        Conv1D,
        MaxPooling1D,
        Bidirectional,
        LSTM,
        Dropout,
        Dense
    )

    model = Sequential([
        Embedding(
            input_dim=10000,
            output_dim=128,
            input_length=100
        ),

        Conv1D(
            filters=128,
            kernel_size=5,
            activation="relu"
        ),

        MaxPooling1D(pool_size=2),

        Conv1D(
            filters=64,
            kernel_size=3,
            activation="relu"
        ),

        Bidirectional(
            LSTM(
                units=64,
                return_sequences=False,
                dropout=0.0,
                recurrent_dropout=0.0
            )
        ),

        Dropout(0.5),

        Dense(64, activation="relu"),

        Dense(1, activation="sigmoid")
    ])

    return model


print("STEP 10 - Building architecture")
model = build_model()

print("STEP 11 - Forcing model build")
model.build(input_shape=(None, 100))

print("STEP 12 - Loading weights")
model.load_weights("models/model.weights.h5")

print("STEP 13 - Weights loaded successfully")

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
        padding="post",
        truncating="post"
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
        return jsonify({
            "error": "No message provided"
        }), 400

    message = data["message"]

    processed = preprocess(message)

    pred = float(model.predict(processed, verbose=0)[0][0])

    label = "Phishing" if pred > 0.5 else "Legitimate"

    return jsonify({
        "message": message,
        "prediction_score": pred,
        "result": label
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)        "result": label
    })
