from flask import Flask, request, jsonify
import numpy as np
import re
import json
import onnxruntime as ort

app = Flask(__name__)

print("STEP 1 - Flask initialized")

# =========================
# CONFIG
# =========================
MAX_LEN = 100
MAX_WORDS = 10000

print("STEP 2 - Config set")

# =========================
# LOAD WORD INDEX
# =========================
print("STEP 3 - Loading word_index.json")

with open("artifacts/word_index.json", "r") as f:
    word_index = json.load(f)

OOV_INDEX = word_index.get("<OOV>", 1)

print("STEP 4 - word_index loaded:", len(word_index))

# =========================
# LOAD ONNX MODEL
# =========================
print("STEP 5 - Loading ONNX model")

session = ort.InferenceSession("models/sms_phishing.onnx")

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print("STEP 6 - ONNX model loaded successfully")

# =========================
# TEXT CLEANING
# =========================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =========================
# TOKENIZER REBUILD (TRAINING MATCHED)
# =========================
def texts_to_sequence(text):
    words = clean_text(text).split()

    seq = []
    for w in words:
        idx = word_index.get(w, OOV_INDEX)

        if idx >= MAX_WORDS:
            idx = OOV_INDEX

        seq.append(idx)

    return seq


# =========================
# PREPROCESSING
# =========================
def preprocess(text):
    seq = texts_to_sequence(text)

    padded = np.zeros((1, MAX_LEN), dtype=np.float32)

    for i, val in enumerate(seq[:MAX_LEN]):
        padded[0, i] = val

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
    try:
        data = request.get_json()

        if not data or "message" not in data:
            return jsonify({"error": "No message provided"}), 400

        message = data["message"]

        print("Incoming message:", message)

        processed = preprocess(message)

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
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("STEP 7 - Starting Flask server")
    app.run(host="0.0.0.0", port=5000)
