import cv2
import numpy as np
import tensorflow as tf
from soil_properties import get_soil_properties_and_crops
from tkinter import Tk, filedialog

IMG_SIZE = (224, 224)
CLASS_NAMES = ["Alluvial soil", "Black Soil", "Clay soil", "Red soil"]

model = tf.keras.models.load_model("soil_mobilenet_model.h5")

def choose_file():
    Tk().withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Soil Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )
    return file_path

def predict_image():
    file_path = choose_file()
    if not file_path:
        print("No file selected.")
        return

    image = cv2.imread(file_path)
    if image is None:
        print("Error reading image")
        return

    # Prepare image for model
    img_resized = cv2.resize(image, IMG_SIZE)
    img_array = np.expand_dims(img_resized / 255.0, axis=0)

    pred = model.predict(img_array)
    idx = np.argmax(pred)
    soil_type = CLASS_NAMES[idx]
    print(">>> Model Predicted Soil =", soil_type)

    confidence = float(pred[0][idx] * 100)

    info = get_soil_properties_and_crops(soil_type)

    # Create neat display canvas
    h, w = 700, 900
    display = np.zeros((h, w, 3), dtype=np.uint8)

    # Text positions
    x, y = 20, 40

    # Soil Type
    cv2.putText(display, f"Soil: {soil_type}", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
    y += 60

    # Confidence
    cv2.putText(display, f"Confidence: {confidence:.2f}%", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    y += 50

    ###################################
    #       PROPERTIES SECTION
    ###################################
    cv2.putText(display, "Properties:", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)
    y += 50

    for key, val in info["avg"].items():
        cv2.putText(display, f"{key}: {val:.2f}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        y += 35

    y += 20

    ###################################
    #       CROPS SECTION
    ###################################
    cv2.putText(display, "Recommended Crops:", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    y += 45

    for crop in info["crops"]:
        cv2.putText(display, f"- {crop}", (x + 30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        y += 35

    ###################################
    # Show final output
    ###################################
    cv2.imshow("Soil Prediction", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    predict_image()
