from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from soil_properties import get_soil_properties_and_crops

app = Flask(__name__)

# ----------------------------------
# MODEL PATH
# ----------------------------------
MODEL_PATH = "soil_mobilenet_model.h5"

# ----------------------------------
# CLASS LABELS (final & only list)
# MUST MATCH EXACT TRAINING FOLDER NAMES
# ----------------------------------
SOIL_CLASSES = [
    "Alluvial soil",
    "Black Soil",
    "Clay soil",
    "Red soil"
]

# Load model
model = tf.keras.models.load_model(MODEL_PATH)


# ----------------------------------
# HOME ROUTE
# ----------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# ----------------------------------
# PREDICT ROUTE
# ----------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("unknown.html",
                               reason="No file selected",
                               confidence=0)

    file = request.files["file"]

    if file.filename == "":
        return render_template("unknown.html",
                               reason="Empty file uploaded",
                               confidence=0)

    # Save uploaded file
    save_path = os.path.join("static", "uploaded_img.jpg")
    file.save(save_path)

    # Preprocess
    img = load_img(save_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)
    confidence = float(np.max(preds) * 100)
    predicted_idx = np.argmax(preds)
    soil_type = SOIL_CLASSES[predicted_idx]

    # Unknown condition
    if confidence < 60:
        return render_template("unknown.html",
                               reason="Low prediction confidence",
                               confidence=round(confidence, 2))

    # Get properties from CSV
    info = get_soil_properties_and_crops(soil_type)

    if info is None:
        return render_template("unknown.html",
                               reason="Soil type not found in CSV",
                               confidence=round(confidence, 2))

    # Final result
    return render_template("result.html",
                           soil_type=soil_type,
                           confidence=round(confidence, 2),
                           avg=info["avg"],
                           crops=info["crops"],
                           image_path="static/uploaded_img.jpg")


# ----------------------------------
# RUN FLASK
# ----------------------------------
if __name__ == "__main__":
    app.run(debug=True)
