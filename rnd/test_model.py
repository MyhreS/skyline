PATH_TO_SKYLINE = "/Users/simonmyhre/workdir/gitdir/skyline"

import sys

sys.path.append(PATH_TO_SKYLINE)
from cumulus import Evaluater

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam


shape = (63, 512)
model = tf.keras.Sequential(
    [
        layers.Input(shape=(shape[0], shape[1], 1)),
        layers.Conv2D(3, (3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(3, (3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(3, (3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(3, (3, 3), activation="relu"),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)

model.summary()

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)


class_label_map = {
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

Evaluater(model, class_label_map, "../cache/image_from_directory")
