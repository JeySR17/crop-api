from flask import Flask, request, jsonify, render_template_string
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the crop recommendation model
model = pickle.load(open('RandomForest.pkl', 'rb'))

# Variable to store latest sensor data
latest_data = {}

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

@app.route('/logdata', methods=['POST'])
def log_data():
    global latest_data
    latest_data = request.get_json()
    return {'status': 'data received'}

@app.route('/live')
def live_dashboard():
    html = """
    <html>
    <head>
        <title>Live Sensor Data</title>
        <meta http-equiv="refresh" content="5">
        <style>
            body { font-family: Arial, sans-serif; padding: 20px; }
            .box { background: #f4f4f4; padding: 20px; border: 1px solid #ccc; width: fit-content; }
        </style>
    </head>
    <body>
        <h2>🌾 Live Sensor Data</h2>
        <div class="box">
        {% if data %}
            {% for k, v in data.items() %}
                <p><strong>{{ k }}</strong>: {{ v }}</p>
            {% endfor %}
        {% else %}
            <p>No data received yet.</p>
        {% endif %}
        </div>
    </body>
    </html>
    """
    return render_template_string(html, data=latest_data)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
