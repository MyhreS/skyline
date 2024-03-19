PATH_TO_SKYLINE = "/workspace/skyline"
PATH_TO_INPUT_DATA = "/workspace/data/datav3"
PATH_TO_OUTPUT_DATA = "cache/data"
RUN_ID = "Test_run"
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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
len_gpus = len(tf.config.experimental.list_physical_devices("GPU"))
print(f"Num GPUs Available: {len_gpus}")


"""
Making the data
"""

data = Data(PATH_TO_INPUT_DATA, PATH_TO_OUTPUT_DATA, RUN_ID)
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
        "non-drone": [
            "dvc_non_drone",
            "animal",
            "speech",
            "TUT_dcase",
            "nature_chernobyl",
        ],
    }
)
data.set_augmentations(["mix_1", "mix_2", "mix_3"], only_drone=True)
data.set_limit(300_000)
data.set_audio_format("log_mel")
data.save_format("image")
data.describe_it()
data.make_it(clean=True)

"""
Loading the data
"""

training_dataset = image_dataset_from_directory(
    os.path.join(PATH_TO_OUTPUT_DATA, "train"),
    seed=123,
    image_size=(63, 512),
    batch_size=32,
    color_mode="grayscale",
)

validation_dataset = image_dataset_from_directory(
    os.path.join(PATH_TO_OUTPUT_DATA, "val"),
    seed=123,
    image_size=(63, 512),
    batch_size=32,
    color_mode="grayscale",
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
        layers.Conv2D(3, (3, 3), padding="same", activation="relu"),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
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
callbacks.append(EarlyStopping(monitor="val_loss", patience=10))
callbacks.append(TensorBoard(log_dir=os.path.join("cache", RUN_ID), histogram_freq=1))


history = model.fit(
    training_dataset,
    validation_data=validation_dataset,
    epochs=10,
    callbacks=callbacks,
    class_weight=calculate_class_weights(training_dataset),
)
log_train_history(history, RUN_ID)
log_model(model, RUN_ID)

"""
Evaluating the model
"""

Evaluater(
    model,
    data.datamaker.label_map,
    PATH_TO_OUTPUT_DATA,
    label_mode="binary",
    run_id=RUN_ID,
)
