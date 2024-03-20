PATH_TO_SKYLINE = "/cluster/datastore/simonmy/skyline"
import os
import sys

sys.path.append(PATH_TO_SKYLINE)

from cumulus import (
    Evaluater,
    calculate_class_weights,
    log_model,
    log_model_summary,
    log_train_history,
)
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
len_gpus = len(tf.config.experimental.list_physical_devices("GPU"))
print(f"Num GPUs Available: {len_gpus}")


"""
Loading the data
"""

RUN_ID = "Run-4-electric_fixedwing_drone-other_drone-non_drone"
output_data = os.path.join("cache", RUN_ID, "data")

label_map = {
    "electric_fixedwing_drone": ["electric_fixedwing_drone"],
    "other-drones": [
        "electric_quad_drone",
        "racing_drone",
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

training_dataset = image_dataset_from_directory(
    os.path.join(output_data, "train"),
    seed=123,
    image_size=(63, 512),
    batch_size=32,
    color_mode="grayscale",
    label_mode="categorical",
)

validation_dataset = image_dataset_from_directory(
    os.path.join(output_data, "val"),
    seed=123,
    image_size=(63, 512),
    batch_size=32,
    color_mode="grayscale",
    label_mode="categorical",
)


"""
Building the model
"""

shape = (63, 512)
base_model = ResNet50(
    weights="imagenet", include_top=False, input_shape=(shape[0], shape[1], 3)
)

base_model.trainable = False
model = tf.keras.Sequential(
    [
        layers.Input(shape=(shape[0], shape[1], 1)),
        layers.Conv2D(3, (3, 3), padding="same"),
        base_model,
        layers.Conv2D(256, 3, padding="same", activation="relu"),
        layers.Dropout(0.5),
        layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dense(3, activation="softmax"),
    ]
)

model.summary()
log_model_summary(model, RUN_ID)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

"""
Fitting the model
"""

callbacks = []
callbacks.append(EarlyStopping(monitor="val_loss", patience=7))
callbacks.append(TensorBoard(log_dir=os.path.join("cache", RUN_ID), histogram_freq=1))


history = model.fit(
    training_dataset,
    validation_data=validation_dataset,
    epochs=10,
    callbacks=callbacks,
    class_weight=calculate_class_weights(training_dataset),
)
log_train_history(history.history, RUN_ID)
log_model(model, RUN_ID)


"""
Evaluating the model
"""

Evaluater(
    model,
    label_map,
    output_data,
    label_mode="categorical",
    run_id=RUN_ID,
)
