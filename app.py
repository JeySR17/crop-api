from flask import Flask, request, jsonify
import pickle
import numpy as np

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
