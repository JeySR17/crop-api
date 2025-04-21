from flask import Flask, request, jsonify, render_template
import pickle
import os

app = Flask(__name__)

# Load the crop recommendation model
model = pickle.load(open('RandomForest.pkl', 'rb'))

# Variable to store the latest sensor data
latest_data = {}

@app.route('/')
def home():
    return 'Crop Recommendation API is running!'

@app.route('/predict', methods=['POST'])
def predict():
    global latest_data
    data = request.get_json()

    # Store the incoming data for later use (to display on /viewdata route)
    latest_data = {
        'N': data['N'],
        'P': data['P'],
        'K': data['K'],
        'temperature': data['temperature'],
        'humidity': data['humidity'],
        'ph': data['ph'],
        'rainfall': data['rainfall']
    }
    
    # Extract features and make prediction
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

    # Return the prediction
    return jsonify({'prediction': prediction})

@app.route('/viewdata', methods=['GET'])
def view_data():
    global latest_data
    if latest_data:
        # Display the latest sensor data and predicted crop
        return render_template('viewdata.html', data=latest_data)
    else:
        return "No data available. Please send data to /predict first."

if __name__ == '__main__':
    # Get the port from the environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
