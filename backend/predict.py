import os
import torch
import json
from transformers import RobertaTokenizer, RobertaForSequenceClassification

MODEL_DIR = r"C:\Users\Lenovo\Desktop\threat detection mp\AI-Threat-Detection-on-Social-Media\models\roberta_threat_model"
MAX_LEN   = 64
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ID2LABEL = {0: "hate_speech", 1: "suicide", 2: "extremism", 3: "fake_news"}

def load_model():
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_DIR)
    model = RobertaForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    model.eval()
    return tokenizer, model

def predict(text, tokenizer, model):
    encoding = tokenizer(
        text, truncation=True, padding="max_length",
        max_length=MAX_LEN, return_tensors="pt"
    )
    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs   = torch.softmax(outputs.logits, dim=1).squeeze().cpu().numpy()
    pred_id    = int(probs.argmax())
    pred_label = ID2LABEL[pred_id]
    confidence = float(probs[pred_id])
    return {
        "prediction": pred_label,
        "confidence": round(confidence * 100, 2),
        "scores":     {ID2LABEL[i]: round(float(p) * 100, 2) for i, p in enumerate(probs)}
    }

if __name__ == "__main__":
    print("Loading model...")
    tokenizer, model = load_model()
    print("Model loaded! Type text to classify (Ctrl+C to exit)\n")
    while True:
        try:
            text = input("Enter text: ").strip()
            if not text:
                continue
            result = predict(text, tokenizer, model)
            print(f"\n  Prediction : {result['prediction'].upper()}")
            print(f"  Confidence : {result['confidence']}%")
            print(f"  All scores : {result['scores']}\n")
        except KeyboardInterrupt:
            print("\nDone.")
            break