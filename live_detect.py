
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load model
model = tf.keras.models.load_model("soil_cnn_model.h5")

# Class labels
class_labels = [
    "Alluvial soil",
    "Black Soil",
    "Clay soil",
    "Red soil"
]

# Preprocessing
def preprocess(frame):
    img = cv2.resize(frame, (224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

# Start webcam
cap = cv2.VideoCapture(0)

print("📸 Starting camera... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Predict
    img_input = preprocess(frame)
    preds = model.predict(img_input)
    idx = np.argmax(preds)
    soil_type = class_labels[idx]
    confidence = preds[0][idx] * 100

    # Draw prediction on frame
    text = f"{soil_type} ({confidence:.2f}%)"
    cv2.putText(frame, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0,255,0), 2)

    # Display video
    cv2.imshow("Soil Live Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
