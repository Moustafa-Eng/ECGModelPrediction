import joblib
from flask import Flask, request, jsonify
import numpy as np
from model_loader import model
def load_model(model_path):
    model = joblib.load(model_path)
    return model

# Load the model once, so it can be reused across requests
model = load_model('trained_model5_inceptionv3old.joblib')




app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.get_json(force=True)
        
        # Convert the JSON data to a numpy array (adjust as needed for your model)
        input_data = np.array(data['input']).reshape(1, -1)
        
        # Make a prediction using the pre-loaded model
        prediction = model.predict(input_data)
        
        # Return the prediction as a JSON response
        return jsonify({'prediction': prediction.tolist()})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
