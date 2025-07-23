
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import json
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("./")
tokenizer = AutoTokenizer.from_pretrained("./")

# Load emotion columns
with open("./emotion_columns.json", "r") as f:
    emotion_columns = json.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", 
                      truncation=True, max_length=256)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits.detach().cpu().numpy()
    
    # Convert logits to probabilities using sigmoid
    probs = 1 / (1 + np.exp(-logits))[0]
    
    # Convert to binary predictions (0 or 1)
    predictions = (probs > 0.5).astype(int)
    
    # Create response with emotions and their predictions
    result = {emotion_columns[i]: int(predictions[i]) for i in range(len(emotion_columns))}
    emotions_detected = [emotion for emotion, val in result.items() if val == 1]
    
    return jsonify({
        'predictions': result,
        'emotions_detected': emotions_detected,
        'probabilities': {emotion_columns[i]: float(probs[i]) for i in range(len(emotion_columns))}
    })

if __name__ == '__main__':
    print("Starting Flask server on port 5000...")
    app.run(host='0.0.0.0', port=5000)
