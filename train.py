import tensorflow as tf  # type: ignore
from tensorflow.keras import layers, models  # type: ignore
from tensorflow.keras.applications import MobileNetV2  # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # type: ignore
import matplotlib.pyplot as plt

DATASET_PATH = "dataset"

IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 10

# Load training data
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Load validation data
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_dataset.class_names

print("Classes found:", class_names)

if class_names != ["cat", "dog"]:
    raise ValueError(
        f"Expected folders ['cat', 'dog'], but found {class_names}"
    )

# Improve performance
AUTOTUNE = tf.data.AUTOTUNE

# Apply preprocessing to datasets
def preprocess_batch(images, labels):
    return preprocess_input(images), labels

train_dataset = train_dataset.map(preprocess_batch, num_parallel_calls=AUTOTUNE)
validation_dataset = validation_dataset.map(preprocess_batch, num_parallel_calls=AUTOTUNE)

train_dataset = train_dataset.prefetch(AUTOTUNE)
validation_dataset = validation_dataset.prefetch(AUTOTUNE)

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

# Pre-trained MobileNetV2
base_model = MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

# Build model (without Lambda layer for serialization compatibility)
model = models.Sequential([
    data_augmentation,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)

# Save model
model.save("cat_dog_model.keras")

print("===================================")
print("MODEL SAVED SUCCESSFULLY!")
print("Classes:", class_names)
print("===================================")

# Accuracy graph
plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()

plt.show()