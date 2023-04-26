
from flask import Flask, request, jsonify
import joblib
import pandas as pd
# from sklearn.feature_extraction.text import TfidfVesctorizer

# Initialize Flask app
application = Flask(__name__)

# Load pre-trained model and vectorizer
model = joblib.load('fake_news_classification_model.pkl')
vectorizer = joblib.load('fake_news_classification_vectorizer.pkl')

@application.route("/",methods=["GET"])
def home():
    return "Fake news detector api"

# Define API endpoint for fake news classification
@application.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from request
        data = request.json

        # Preprocess the input data
        title = data['title']
        text = data['text']

        # Vectorize the input data
        input_text = title + ' ' + text
        input_vector = vectorizer.transform([input_text])

        # Make prediction
        prediction = model.predict(input_vector)[0]
        prediction_str = str(prediction)

        # Create response
        response = {
            'prediction': prediction_str
        }

        # Return response as JSON
        return jsonify(response)
    except Exception as e:
        # Return error message if there's an exception
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Run the Flask app
    application.run(debug=True)
