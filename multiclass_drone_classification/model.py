PATH_TO_SKYLINE = "/workspace/skyline"  # "/cluster/datastore/simonmy/skyline"
PATH_TO_INPUT_DATA = (
    "/workspace/data/datav3"  # "/cluster/datastore/simonmy/data/datav2"
)
PATH_TO_OUTPUT_DATA = (
    "/workspace/skyline/cache/data"  # "/cluster/datastore/simonmy/skyline/cache/data"
)
# PATH_TO_SKYLINE = "/cluster/datastore/simonmy/skyline"  # "/workspace/skyline"
# PATH_TO_INPUT_DATA = "/cluster/datastore/simonmy/data/datav2"  # "/workspace/data/data"
# PATH_TO_OUTPUT_DATA = (
#     "/cluster/datastore/simonmy/skyline/cache/data"  # "/workspace/skyline/cache/data"
# )
RUN_NAME = "run_9"
import sys

sys.path.append(PATH_TO_SKYLINE)
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    LeakyReLU,
    Flatten,
    Dense,
    Dropout,
)
import logging
from cirrus import Data
from cumulus import Logger
from stratus import Evaluater


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M",
)
logging.info(
    "Num GPUs Available: %d", len(tf.config.experimental.list_physical_devices("GPU"))
)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

"""
------------------------------------
MODEL
"""

data = Data(PATH_TO_INPUT_DATA, PATH_TO_OUTPUT_DATA)
data.set_window_size(1, load_cached_windowing=True)
data.set_val_of_train_split(0.2)
data.set_label_class_map(
    {
        "drone": [
            "electric_quad_drone",
            "racing_drone",
            "electric_fixedwing_drone",
            "petrol_fixedwing_drone",
        ],
        "non-drone": [
            "dvc_non_drone",
            "animal",
            "speech",
            "TUT_dcase",
            "nature_chernobyl",
        ],
    }
)
data.set_audio_format("log_mel")
data.set_augmentations(["mix_1", "mix_2", "mix_3", "mix_4"], only_drone=True)
data.set_limit(100_000)
data.describe_it()
data.make_it()

train_ds, shape, class_weights = data.load_it(split="train", label_encoding="integer")
val_ds, shape = data.load_it(split="val", label_encoding="integer")

print(class_weights)
print(shape)
logger = Logger(RUN_NAME, clean=True)


# Create a CNN model
# Load ResNet50 with pre-trained ImageNet weights
# base_model = ResNet50(
#     weights="imagenet", include_top=False, input_shape=(shape[0], shape[1], 3)
# )

# # Freeze the base model
# base_model.trainable = False
# Create the model
model = Sequential(
    [
        Conv2D(
            64,
            (3, 3),
            padding="same",
            input_shape=(shape[0], shape[1], 1),
        ),
        BatchNormalization(),
        LeakyReLU(),
        # Second Conv2D layer
        Conv2D(64, (3, 3), padding="same"),
        BatchNormalization(),
        LeakyReLU(),
        Dropout(0.5),
        # Third Conv2D layer
        Conv2D(128, (3, 3), padding="same"),
        BatchNormalization(),
        LeakyReLU(),
        # Fourth Conv2D layer
        Conv2D(256, (3, 3), padding="same"),
        BatchNormalization(),
        LeakyReLU(),
        # Fifth Conv2D layer
        Conv2D(256, (3, 3), padding="same"),
        BatchNormalization(),
        LeakyReLU(),
        Dropout(0.5),
        # Sixth Conv2D layer
        Conv2D(256, (3, 3), padding="same"),
        BatchNormalization(),
        LeakyReLU(),
        Dropout(0.5),
        # Seventh Conv2D layer
        Conv2D(256, (3, 3), padding="same"),
        BatchNormalization(),
        LeakyReLU(),
        # Flatten the output
        Flatten(),
        # Dense layer
        Dense(256, activation="relu"),
        Dense(1, activation="sigmoid"),
    ]
)

model.summary()
logger.log_model_info(model)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

callbacks = []
callbacks.append(EarlyStopping(monitor="val_loss", patience=5))
callbacks.append(TensorBoard(log_dir=logger.get_tensorboard_path(), histogram_freq=1))


history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=callbacks,
    class_weight=class_weights,
)
logger.log_model(model)
logger.log_train_history(history.history)


evaluater = Evaluater(model, data, logger, "integer")
