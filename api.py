import os
import mlflow.pyfunc
from flask import Flask, request, jsonify

app = Flask(__name__)

# Point directly to the folder you just copied
MODEL_PATH = "model" 

try:
    model = mlflow.pyfunc.load_model(MODEL_PATH)
    print("Model loaded successfully from local folder!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    try:
        data = request.get_json()
        import pandas as pd
        df = pd.DataFrame([data])
        prediction = model.predict(df)
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)