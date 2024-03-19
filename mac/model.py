PATH_TO_SKYLINE = "/Users/simonmyhre/workdir/gitdir/skyline"
PATH_TO_INPUT_DATA = "/Users/simonmyhre/workdir/gitdir/sqml/projects/sm_multiclass_masters_project/pull_data/cache/datav3"
PATH_TO_OUTPUT_DATA = "cache/data"
RUN_ID = "Test_run"
import os
import sys

sys.path.append(PATH_TO_SKYLINE)

from cirrus import Data
from cumulus import Evaluater, calculate_class_weights, log_train_history
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

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
# data.set_augmentations(["mix_1", "mix_2"], only_drone=True)
data.set_limit(500)
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
Building and fitting the model
"""

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

history = model.fit(
    training_dataset,
    validation_data=validation_dataset,
    epochs=1,
    class_weights=calculate_class_weights(training_dataset),
)
log_train_history(history.history, RUN_ID)

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
