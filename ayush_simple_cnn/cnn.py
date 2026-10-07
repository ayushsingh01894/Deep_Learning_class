import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from PIL import Image
from pathlib import Path

# =========================
# SETTINGS
# =========================
TRAIN_FOLDER = "dataset/Ayush"
TEST_IMAGE = "test/test.jpg"

IMG_SIZE = (128, 128)
BATCH_SIZE = 8
EPOCHS = 10

# =========================
# LOAD YOUR AYUSH IMAGES
# =========================
if not Path(TRAIN_FOLDER).exists():
    raise SystemExit("dataset/Ayush folder nahi mila.")

dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    labels="inferred",
    label_mode="int",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = dataset.class_names
print("Classes:", class_names)

# =========================
# CNN MODEL
# =========================
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(128, 128, 3)),

    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# =========================
# TRAIN
# =========================
print("\nTraining started...\n")
model.fit(dataset, epochs=EPOCHS)

print("\nTraining completed!")

# =========================
# PREDICT NEW IMAGE
# =========================
if not Path(TEST_IMAGE).exists():
    print("\nTest image nahi mili.")
    print("Apni new image 'test' folder me test.jpg naam se rakho.")
    print("Example: test/test.jpg")
    raise SystemExit

img = Image.open(TEST_IMAGE).convert("RGB")
img = img.resize(IMG_SIZE)

img = np.array(img, dtype=np.float32) / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img, verbose=0)[0]
index = np.argmax(prediction)

print("\n======================")
print("Prediction :", class_names[index])
print("Confidence :", round(float(prediction[index]) * 100, 2), "%")
print("======================")
