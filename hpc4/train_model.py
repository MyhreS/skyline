PATH_TO_SKYLINE = "/workspace/skyline"
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
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    LeakyReLU,
    Flatten,
    Dense,
    MaxPooling2D,
    Dropout,
    Input,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
len_gpus = len(tf.config.experimental.list_physical_devices("GPU"))
print(f"Num GPUs Available: {len_gpus}")


"""
Loading the data
"""
RUN_ID = "Run-5-petrol_fixedwing_drone-other_drone-other"
output_data = os.path.join("cache", RUN_ID, "data")

label_map = {
    "petrol_fixedwing_drone": ["petrol_fixedwing_drone"],
    "other-drones": [
        "electric_quad_drone",
        "racing_drone",
        "electric_fixedwing_drone",
    ],
    "other": [
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

base_model.trainable = True

layer_name = base_model.layers[80].name
intermediate_model = Model(
    inputs=base_model.input, outputs=base_model.get_layer(layer_name).output
)

model = Sequential(
    [
        Input(shape=(shape[0], shape[1], 1)),
        Conv2D(3, (3, 3), padding="same"),
        intermediate_model,
        Conv2D(256, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(1, 2)),
        Dropout(0.5),
        Conv2D(256, (3, 3), padding="same", activation="relu"),
        MaxPooling2D(pool_size=(1, 2)),
        Conv2D(256, (3, 3), padding="same", activation="relu"),
        Dropout(0.5),
        MaxPooling2D(pool_size=(1, 2)),
        Conv2D(256, (3, 3), padding="same", activation="relu"),
        Dropout(0.5),
        MaxPooling2D(pool_size=(1, 2)),
        Conv2D(256, (3, 3), padding="same", activation="relu"),
        MaxPooling2D(pool_size=(1, 2)),
        Conv2D(256, (3, 3), padding="same", activation="relu"),
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(128, activation="relu"),
        Dense(3, activation="softmax"),
    ]
)

model.summary()
log_model_summary(model, RUN_ID)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0000005),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

"""
Fitting the model
"""

callbacks = []
callbacks.append(EarlyStopping(monitor="val_loss", patience=5))
callbacks.append(TensorBoard(log_dir=os.path.join("cache", RUN_ID), histogram_freq=1))


history = model.fit(
    training_dataset,
    validation_data=validation_dataset,
    epochs=30,
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
