from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import load_model, predict
import anthropic

app = Flask(__name__)
CORS(app)

print("Loading RoBERTa model...")
tokenizer, model = load_model()
print("Model ready!")

claude = anthropic.Anthropic()

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "SentinelAI API running"})

@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Provide 'text' in request body"}), 400
    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Text cannot be empty"}), 400
    result = predict(text, tokenizer, model)
    return jsonify({"text": text, "prediction": result["prediction"],
                    "confidence": result["confidence"], "scores": result["scores"]})

@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    data = request.get_json()
    if not data or "texts" not in data:
        return jsonify({"error": "Provide 'texts' array"}), 400
    results = [predict(t, tokenizer, model) for t in data["texts"]]
    return jsonify({"results": results})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Provide 'message' in request body"}), 400

    user_message = data["message"]
    history      = data.get("history", [])
    system       = data.get("system", "You are ARIA, a helpful AI safety agent.")

    messages = []
    for h in history:
        if h.get("role") in ("user", "assistant"):
            messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": user_message})

    try:
        response = claude.messages.create(
            model="claude-opus-4-6",
            max_tokens=512,
            system=system,
            messages=messages
        )
        reply = response.content[0].text
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e), "reply": "I'm here for you. Could you tell me more?"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)