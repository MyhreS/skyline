import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50

# Create a CNN model
# Load ResNet50 with pre-trained ImageNet weights
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(128, 87, 3))

# Freeze the base model
base_model.trainable = False
# Create the model
model = tf.keras.Sequential(
    [
        layers.Input(shape=(128, 87, 1)),
        layers.Conv2D(
            3, (3, 3), padding="same"
        ),  # This layer converts the 1 channel input to 3 channels
        base_model,
        layers.Conv2D(256, kernel_size=(3, 3), activation="relu"),
        # Add a normal conv2d layer
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dense(6, activation="softmax"),
    ]
)

model.summary()

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
