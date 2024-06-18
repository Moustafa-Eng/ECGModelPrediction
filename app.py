from flask import Flask, request, jsonify
import numpy as np
import cv2
import tensorflow as tf
app = Flask(__name__)

model_path = "trained_model5_inceptionv3old.keras"

def load_model(model_path):
    try:
        # Attempt to load as a Keras model
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

loaded_model = load_model(model_path)

# Define cropping dimensions
top_crop = 250
bottom_crop = 100
image_size = (299, 299)
lower_white = np.array([50, 50, 50], dtype=np.uint8)
upper_white = np.array([255, 255, 255], dtype=np.uint8)
class_labels = [
    'Patient that have History of MI',
    'Myocardial Infarction Patients',
    'Normal Person',
    'Patient that have abnormal heart beats',
    'COVID-19 Patients'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if loaded_model is None:
            return jsonify({"error": "Model is not loaded properly."}), 500

        img_file = request.files.get('image')
        if img_file is None:
            return jsonify({"error": "No image file found in the request."}), 400
        
        img = cv2.imdecode(np.frombuffer(img_file.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Failed to decode the image."}), 400
        
        cropped_img = img[top_crop:-bottom_crop, :]
        img_filtered = cv2.bilateralFilter(cropped_img, d=9, sigmaColor=75, sigmaSpace=75)
        mask = cv2.inRange(img_filtered, lower_white, upper_white)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        mask = cv2.bitwise_not(mask)
        bk = np.full(cropped_img.shape, 255, dtype=np.uint8)
        fg_masked = cv2.bitwise_and(cropped_img, cropped_img, mask=mask)
        mask = cv2.bitwise_not(mask)
        bk_masked = cv2.bitwise_and(bk, bk, mask=mask)
        final = cv2.bitwise_or(fg_masked, bk_masked)
        resized_photo = cv2.resize(final, image_size)
        normalized_photo = resized_photo / 255.0
        expanded_photo = np.expand_dims(normalized_photo, axis=0)

        prediction = loaded_model.predict(expanded_photo)
        predicted_category = class_labels[np.argmax(prediction)]
        return jsonify({"predicted_category": predicted_category})
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/')
def index():
    return "Hello, World!"

if __name__ == '__main__':
    app.run(debug=True)
