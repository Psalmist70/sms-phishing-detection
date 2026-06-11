from flask import Flask, request, jsonify
import numpy as np
import re
import onnxruntime as ort

app = Flask(__name__)

# =========================
# CONFIG
# =========================

MAX_LEN = 100  # hardcoded (from training)

# =========================
# LOAD ONNX MODEL
# =========================

session = ort.InferenceSession("models/sms_phishing.onnx")

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# =========================
# TEXT PREPROCESSING
# =========================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# SIMPLE TOKENIZATION (NO KERAS, NO JOBLIB)
def encode_text(text):
    text = clean_text(text)

    # simple character-level numeric encoding
    seq = [ord(c) % 10000 for c in text]

    padded = np.zeros((1, MAX_LEN), dtype=np.float32)

    for i, val in enumerate(seq[:MAX_LEN]):
        padded[0, i] = val

    return padded.astype(np.float32)


# =========================
# ROUTES
# =========================

@app.route("/")
def home():
    return jsonify({
        "message": "SMS Phishing Detection API (ONNX) is running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data or "message" not in data:
            return jsonify({"error": "No message provided"}), 400

        message = data["message"]

        processed = encode_text(message)

        pred = session.run(
            [output_name],
            {input_name: processed}
        )[0][0][0]

        label = "Phishing" if pred > 0.5 else "Legitimate"

        return jsonify({
            "message": message,
            "prediction_score": float(pred),
            "result": label
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# =========================
# RUN SERVER
# =========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
