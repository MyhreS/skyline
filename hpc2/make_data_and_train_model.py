PATH_TO_SKYLINE = "/cluster/datastore/simonmy/skyline"
PATH_TO_INPUT_DATA = "/cluster/datastore/simonmy/data/datav3"
# PATH_TO_OUTPUT_DATA = "cache/data"
# RUN_ID = "run-multiclass_electic-quad_other-drone_non-drone"
import os
import sys

sys.path.append(PATH_TO_SKYLINE)

from cirrus import Data
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    LeakyReLU,
    Flatten,
    Dense,
    MaxPooling2D,
    Dropout,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
len_gpus = len(tf.config.experimental.list_physical_devices("GPU"))
print(f"Num GPUs Available: {len_gpus}")


"""
Making the data
"""

RUN_ID = "Run-1-CNN-binary_drone_non_drone"
output_data = os.path.join("cache", RUN_ID, "data")
data = Data(PATH_TO_INPUT_DATA, output_data, RUN_ID)
data.set_window_size(2, load_cached_windowing=True)
data.set_val_of_train_split(0.2)
data.set_label_class_map(
    {
        "drone": [
            "electric_quad_drone",
            "racing_drone",
            "electric_fixedwing_drone",
            "petrol_fixedwing_drone",
        ],
        "other": [
            "dvc_non_drone",
            "animal",
            "speech",
            "TUT_dcase",
            "nature_chernobyl",
        ],
    }
)
# data.set_augmentations(["mix_1", "mix_2"], only_drone=True)
data.set_limit(150_000)
data.set_audio_format("log_mel")
data.save_format("image")
data.describe_it()
data.make_it(clean=False)

"""
Loading the data
"""

training_dataset = image_dataset_from_directory(
    os.path.join(output_data, "train"),
    seed=123,
    image_size=(63, 512),
    batch_size=32,
    # color_mode="grayscale",
    # label_mode="categorical",
)

validation_dataset = image_dataset_from_directory(
    os.path.join(output_data, "val"),
    seed=123,
    image_size=(63, 512),
    batch_size=32,
    color_mode="grayscale",
    # label_mode="categorical",
)


"""
Building the model
"""

# shape = (63, 512)
# base_model = ResNet50(
#     weights="imagenet", include_top=False, input_shape=(shape[0], shape[1], 3)
# )

# base_model.trainable = False
# model = tf.keras.Sequential(
#     [
#         layers.Input(shape=(shape[0], shape[1], 1)),
#         layers.Conv2D(3, (3, 3), padding="same"),
#         base_model,
#         layers.Conv2D(256, 3, padding="same", activation="relu"),
#         layers.Dropout(0.5),
#         layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
#         layers.Flatten(),
#         layers.Dense(256, activation="relu"),
#         layers.Dropout(0.5),
#         layers.Dense(128, activation="relu"),
#         layers.Dense(1, activation="sigmoid"),
#     ]
# )

shape = (63, 512)

# Define the sequential model
model = Sequential(
    [
        # Layer 1
        Conv2D(
            64,
            5,
            input_shape=(shape[0], shape[1], 1),
            activation="relu",
        ),
        Dropout(0.2),
        # Layer 2
        Conv2D(128, 3, activation="relu"),
        MaxPooling2D(2),
        # Layer 3
        Conv2D(256, 3, activation="relu"),
        Dropout(0.2),
        # Layer 4
        Conv2D(256, 3, activation="relu"),
        BatchNormalization(),
        # Layer 5
        Conv2D(256, 3, activation="relu"),
        MaxPooling2D(2),
        # Layer 6
        Conv2D(256, 3, activation="relu"),
        Dropout(0.2),
        # Layer 7
        Conv2D(256, 3, activation="relu"),
        MaxPooling2D(2),
        BatchNormalization(),
        # Layer 8
        Conv2D(256, 3, activation="relu"),
        Dropout(0.2),
        Flatten(),
        Dense(256, activation="relu"),
        # Dense layer
        Dense(128, activation="relu"),
        Dense(1, activation="sigmoid"),
    ]
)

model.summary()
log_model_summary(model, RUN_ID)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
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
    data.datamaker.label_map,
    output_data,
    label_mode="binary",
    run_id=RUN_ID,
)
