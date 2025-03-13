from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import json

# Initialize Flask app
app = Flask(__name__)

# Ensure the 'uploads' folder exists
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the trained model
MODEL_PATH = os.path.abspath("C:\Projects\plant-disease-detection\plant-disease-detection\model\inceptionv3_plant_disease_multi_v2.h5")  # Updated path
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}! Please check the file path.")
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels (Updated for 30 classes)
class_labels = [
    "Healthy", "Powdery Mildew", "Rust", "Leaf Spot", "Blight",
    "Bacterial Wilt", "Early Blight", "Late Blight", "Downy Mildew",
    "Anthracnose", "Cercospora Leaf Spot", "Mosaic Virus", "Fusarium Wilt",
    "Verticillium Wilt", "Black Rot", "Alternaria Leaf Spot", "Charcoal Rot",
    "Damping-Off", "Gray Mold", "Septoria Leaf Spot", "Phytophthora Rot",
    "Sooty Mold", "Yellow Leaf Curl Virus", "Stem Canker", "Bacterial Leaf Streak",
    "Root Knot Nematode", "Pythium Root Rot", "Powdery Scab", "Rhizoctonia Root Rot"
]

# Load disease information from JSON file
DISEASE_INFO_PATH = "C:\Projects\plant-disease-detection1\plant-disease-detection\plant-disease-detection\disease_info.json"
if os.path.exists(DISEASE_INFO_PATH):
    with open(DISEASE_INFO_PATH, "r") as f:
        disease_info = json.load(f)
else:
    disease_info = {}

# Function to preprocess image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(299, 299))  # Resize for InceptionV3
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    return img_array

# Route for Home Page
@app.route('/')
def home():
    return render_template('index.html')

# Route for Prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Save uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        # Preprocess the image
        img_array = preprocess_image(file_path)

        # Make prediction
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        predicted_class = class_labels[predicted_index]
        confidence = round(float(np.max(prediction) * 100), 2)  # Convert to percentage

        # Retrieve disease information (if available)
        disease_details = disease_info.get(predicted_class, {
            "cure": "No information available.",
            "growth_tips": "No information available."
        })

        return render_template('result.html',
                               prediction=predicted_class,
                               confidence=f"{confidence}%",
                               cure=disease_details["cure"],
                               growth_tips=disease_details["growth_tips"])

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
