# from flask import Flask, render_template, request
# import numpy as np
# import os
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from soil_properties import get_soil_properties_and_crops

# app = Flask(__name__)

# # ------------------------------
# # CONFIGURATION FOR RENDER
# # ------------------------------
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# # Create uploads directory if it doesn't exist
# UPLOAD_FOLDER = 'static/uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # ------------------------------
# # LOAD TFLITE MODEL
# # ------------------------------
# try:
#     interpreter = tf.lite.Interpreter(model_path="soil_model.tflite")
#     interpreter.allocate_tensors()
#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()
#     model_loaded = True
# except Exception as e:
#     print(f"Error loading model: {e}")
#     model_loaded = False

# # ------------------------------
# # SOIL CLASSES (ORDER MUST MATCH TRAINING)
# # ------------------------------
# SOIL_CLASSES = [
#     "Alluvial soil",
#     "Black Soil",
#     "Clay soil",
#     "Red soil"
# ]

# # ------------------------------
# # FUNCTION: Predict using TFLite
# # ------------------------------
# def predict_tflite(img_array):
#     interpreter.set_tensor(input_details[0]["index"], img_array)
#     interpreter.invoke()
#     preds = interpreter.get_tensor(output_details[0]["index"])
#     return preds

# # ------------------------------
# # HEALTH CHECK ROUTE (IMPORTANT FOR RENDER)
# # ------------------------------
# @app.route("/health")
# def health_check():
#     return {"status": "healthy", "model_loaded": model_loaded}, 200

# # ------------------------------
# # HOME PAGE
# # ------------------------------
# @app.route("/")
# def home():
#     return render_template("index.html")

# # ------------------------------
# # PREDICT ROUTE
# # ------------------------------
# @app.route("/predict", methods=["POST"])
# def predict():
#     # Check if model is loaded
#     if not model_loaded:
#         return render_template("unknown.html",
#                                reason="Model not loaded",
#                                confidence=0)

#     # No file uploaded
#     if "file" not in request.files:
#         return render_template("unknown.html",
#                                reason="No file selected",
#                                confidence=0)

#     file = request.files["file"]

#     if file.filename == "":
#         return render_template("unknown.html",
#                                reason="Empty file uploaded",
#                                confidence=0)

#     try:
#         # Save uploaded image with unique name
#         filename = f"uploaded_{os.urandom(8).hex()}.jpg"
#         save_path = os.path.join(UPLOAD_FOLDER, filename)
#         file.save(save_path)

#         # Preprocess the image
#         img = load_img(save_path, target_size=(224, 224))
#         img_array = img_to_array(img) / 255.0
#         img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

#         # TFLite Prediction
#         preds = predict_tflite(img_array)[0]
#         confidence = float(np.max(preds) * 100)
#         soil_type = SOIL_CLASSES[np.argmax(preds)]

#         # UNKNOWN soil handling
#         if confidence < 60:  # threshold
#             return render_template(
#                 "unknown.html",
#                 reason="Low prediction confidence",
#                 confidence=round(confidence, 2)
#             )

#         # Read soil properties from CSV
#         info = get_soil_properties_and_crops(soil_type)

#         if info is None:
#             return render_template(
#                 "unknown.html",
#                 reason="Soil type not found in CSV",
#                 confidence=round(confidence, 2)
#             )

#         # Final Result Page
#         return render_template(
#             "result.html",
#             soil_type=soil_type,
#             confidence=round(confidence, 2),
#             avg=info["avg"],
#             crops=info["crops"],
#             image_path=f"/static/uploads/{filename}"
#         )

#     except Exception as e:
#         print(f"Prediction error: {e}")
#         return render_template("unknown.html",
#                                reason="Error processing image",
#                                confidence=0)

# # ------------------------------
# # RUN APP (PRODUCTION SETTINGS)
# # ------------------------------
# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 5000))
#     app.run(host='0.0.0.0', port=port, debug=False)


from flask import Flask, render_template, request
import numpy as np
import os
import tflite_runtime.interpreter as tflite
from PIL import Image
from soil_properties import get_soil_properties_and_crops

app = Flask(__name__)

# ------------------------------
# CONFIGURATION FOR RENDER
# ------------------------------
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------------------
# LOAD TFLITE MODEL
# ------------------------------
try:
    interpreter = tflite.Interpreter(model_path="soil_model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    model_loaded = True
    print("✅ TFLite model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model_loaded = False

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
# PIL IMAGE PREPROCESSING FUNCTION
# ------------------------------
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0).astype(np.float32)

# ------------------------------
# FUNCTION: Predict using TFLite
# ------------------------------
def predict_tflite(img_array):
    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]["index"])
    return preds

# ------------------------------
# HEALTH CHECK ROUTE (IMPORTANT FOR RENDER)
# ------------------------------
@app.route("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model_loaded}, 200

@app.route("/test")
def test():
    return "Server is working! ✅"

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
    # Check if model is loaded
    if not model_loaded:
        return render_template("unknown.html",
                               reason="Model not loaded",
                               confidence=0)

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

    try:
        # Save uploaded image with unique name
        filename = f"uploaded_{os.urandom(8).hex()}.jpg"
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)

        # Preprocess the image using PIL
        img_array = load_and_preprocess_image(save_path, target_size=(224, 224))

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
            image_path=f"/static/uploads/{filename}"
        )

    except Exception as e:
        print(f"Prediction error: {e}")
        return render_template("unknown.html",
                               reason="Error processing image",
                               confidence=0)

# ------------------------------
# RUN APP (PRODUCTION SETTINGS)
# ------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)