from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import json

# Initialize Flask app
app = Flask(__name__)

# Uploads folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "plant_disease_model.h5")
CLASS_INDEX_PATH = os.path.join(BASE_DIR, "class_indices.json")

# Load the model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"⚠️ Model not found at {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

# Load class index map
if not os.path.exists(CLASS_INDEX_PATH):
    raise FileNotFoundError(f"⚠️ class_indices.json not found at {CLASS_INDEX_PATH}")
with open(CLASS_INDEX_PATH, 'r', encoding='utf-8') as f:
    class_indices = json.load(f)

# Build index-to-label mapping
index_to_label = {int(k): v["name"] for k, v in class_indices.items()}
disease_info = {
    v["name"].lower(): {
        "cure": v.get("cure", "No information available."),
        "growth_tips": v.get("growth_tips", "No information available.")
    }
    for v in class_indices.values()
}

# Image preprocessing function
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(64, 64))  # Match training size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        img_array = preprocess_image(file_path)
        prediction = model.predict(img_array)
        predicted_index = int(np.argmax(prediction))
        predicted_class = index_to_label.get(predicted_index, "Unknown")
        confidence = round(float(np.max(prediction)) * 100, 2)

        info = disease_info.get(predicted_class.lower(), {
            "cure": "No information available.",
            "growth_tips": "No information available."
        })

        if info["cure"] == "No information available.":
            print(f"⚠️ Missing info for: {predicted_class}")

        return render_template(
            'result.html',
            prediction=predicted_class,
            confidence=f"{confidence}%",
            cure=info["cure"],
            growth_tips=info["growth_tips"]
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
