from flask import Flask, render_template, request
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from soil_properties import get_soil_properties_and_crops

app = Flask(__name__)

# ------------------------------
# LOAD TFLITE MODEL
# ------------------------------
interpreter = tf.lite.Interpreter(model_path="soil_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ------------------------------
# SOIL CLASSES (ORDER MUST MATCH TRAINING)
# ------------------------------
SOIL_CLASSES = [
    "Alluvial soil",
    "Black Soil",
    "Clay soil",
    "Red soil"
]


# ------------------------------
# FUNCTION: Predict using TFLite
# ------------------------------
def predict_tflite(img_array):
    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]["index"])
    return preds


# ------------------------------
# HOME PAGE
# ------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# ------------------------------
# PREDICT ROUTE
# ------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    # No file uploaded
    if "file" not in request.files:
        return render_template("unknown.html",
                               reason="No file selected",
                               confidence=0)

    file = request.files["file"]

    if file.filename == "":
        return render_template("unknown.html",
                               reason="Empty file uploaded",
                               confidence=0)

    # Save uploaded image
    save_path = os.path.join("static", "uploaded_img.jpg")
    file.save(save_path)

    # Preprocess the image
    img = load_img(save_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # TFLite Prediction
    preds = predict_tflite(img_array)[0]
    confidence = float(np.max(preds) * 100)
    soil_type = SOIL_CLASSES[np.argmax(preds)]

    # UNKNOWN soil handling
    if confidence < 60:  # threshold
        return render_template(
            "unknown.html",
            reason="Low prediction confidence",
            confidence=round(confidence, 2)
        )

    # Read soil properties from CSV
    info = get_soil_properties_and_crops(soil_type)

    if info is None:
        return render_template(
            "unknown.html",
            reason="Soil type not found in CSV",
            confidence=round(confidence, 2)
        )

    # Final Result Page
    return render_template(
        "result.html",
        soil_type=soil_type,
        confidence=round(confidence, 2),
        avg=info["avg"],
        crops=info["crops"],
        image_path="/static/uploaded_img.jpg"
    )


# ------------------------------
# RUN APP
# ------------------------------
if __name__ == "__main__":
    app.run(debug=True)
