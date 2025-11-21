import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# Path to your dataset
TRAIN_PATH = "Dataset/Train"
TEST_PATH = "Dataset/test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25

# -------------------- DATA GENERATOR --------------------
train_datagen = ImageDataGenerator(
    rescale=1/255.,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1/255.)

train_gen = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_gen = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

num_classes = train_gen.num_classes
print("Detected Classes:", train_gen.class_indices)

# -------------------- MOBILE NET V2 BASE --------------------
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False   # Freeze pretrained layers

# -------------------- CUSTOM MODEL --------------------
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------- TRAIN --------------------
history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=EPOCHS
)

# -------------------- SAVE MODEL --------------------
model.save("soil_mobilenet_model.h5")
print("Model saved as soil_mobilenet_model.h5")
