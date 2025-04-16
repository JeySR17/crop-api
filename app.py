from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the crop recommendation model
model = pickle.load(open('RandomForest.pkl', 'rb'))

@app.route('/')
def home():
    return 'Crop Recommendation API is running!'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = [
        data['N'],
        data['P'],
        data['K'],
        data['temperature'],
        data['humidity'],
        data['ph'],
        data['rainfall']
    ]
    prediction = model.predict([features])[0]
    return jsonify(prediction)

if __name__ == '__main__':
    # Get the port from the environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
