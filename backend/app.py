from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import load_model, predict

app = Flask(__name__)
CORS(app)  # allows frontend to call this API

# Load model once when server starts
print("Loading model...")
tokenizer, model = load_model()
print("✅ Model ready!")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "AI Threat Detection API is running"})

@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Please provide 'text' in request body"}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Text cannot be empty"}), 400

    result = predict(text, tokenizer, model)
    return jsonify({
        "text":       text,
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "scores":     result["scores"]
    })

@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    data = request.get_json()

    if not data or "texts" not in data:
        return jsonify({"error": "Please provide 'texts' array in request body"}), 400

    texts = data["texts"]
    if not isinstance(texts, list) or len(texts) == 0:
        return jsonify({"error": "texts must be a non-empty array"}), 400

    results = [predict(t, tokenizer, model) for t in texts]
    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(debug=True, port=5000)