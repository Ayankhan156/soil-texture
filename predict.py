import tensorflow as tf
import numpy as np
import cv2
import sys
from soil_properties import get_soil_properties_and_crops

# Load model
MODEL_PATH = "soil_mobilenet_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels (same order as training)
class_names = ["Alluvial soil", "Black Soil", "Clay soil", "Red soil"]

IMG_SIZE = (224, 224)

def predict_image(image_path):
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Error: Could not read image.")
        return

    # Preprocess
    img_resized = cv2.resize(img, IMG_SIZE)
    img_normalized = img_resized / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0)

    # Predict
    predictions = model.predict(img_expanded)
    class_index = np.argmax(predictions)
    confidence = float(np.max(predictions)) * 100
    soil_type = class_names[class_index]

    # --- Print Result ---
    print("\n==============================")
    print(f"Predicted Soil Type: {soil_type}")
    print(f"Confidence: {confidence:.2f}%")
    print("==============================\n")

    # --- Get CSV-based info ---
    info = get_soil_properties_and_crops(soil_type)
    
    print("---- Average Soil Properties (From CSV) ----")
    for key, val in info["average_properties"].items():
        print(f"{key}: {val}")

    print("\n---- Recommended Crops ----")
    for crop in info["recommended_crops"]:
        print(f"• {crop}")

    print("\n")

# ---- RUN ----
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
    else:
        predict_image(sys.argv[1])
