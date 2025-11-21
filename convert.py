import tensorflow as tf

model = tf.keras.models.load_model("soil_mobilenet_model.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("soil_model.tflite", "wb") as f:
    f.write(tflite_model)
