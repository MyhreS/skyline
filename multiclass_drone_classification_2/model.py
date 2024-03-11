# PATH_TO_SKYLINE = "/workspace/skyline"  # "/cluster/datastore/simonmy/skyline"
# PATH_TO_INPUT_DATA = (
#     "/workspace/data/datav2"  # "/cluster/datastore/simonmy/data/datav2"
# )
# PATH_TO_OUTPUT_DATA = (
#     "/workspace/skyline/cache/data"  # "/cluster/datastore/simonmy/skyline/cache/data"
# )
PATH_TO_SKYLINE = "/cluster/datastore/simonmy/skyline"  # "/workspace/skyline"
PATH_TO_INPUT_DATA = "/cluster/datastore/simonmy/data/datav2"  # "/workspace/data/data"
PATH_TO_OUTPUT_DATA = (
    "/cluster/datastore/simonmy/skyline/cache/data"  # "/workspace/skyline/cache/data"
)
RUN_NAME = "run_1_resnet"
import sys

sys.path.append(PATH_TO_SKYLINE)
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
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

data = Data(PATH_TO_INPUT_DATA, PATH_TO_OUTPUT_DATA)
data.set_window_size(1)
data.set_split_configuration(train_percent=50, test_percent=35, val_percent=15)
data.set_label_class_map(
    {
        "normal_drone": ["normal_drone"],
        "normal_fixedwing": ["normal_fixedwing"],
        "petrol_fixedwing": ["petrol_fixedwing"],
        "racing_drone": ["racing_drone"],
        "non-drone": ["nature_chernobyl", "false_positives_drone"],
        "speech": ["speech"],
    }
)
data.remove_label("false_positives_drone")
# data.remove_label("speech")
data.set_sample_rate(44100)
# data.set_augmentations(
#     ["low_pass", "pitch_shift", "add_noise", "high_pass", "band_pass"]
# )
data.set_audio_format("log_mel")
data.set_limit(150_000)
data.describe_it()
data.make_it()

train_ds, shape, class_weights = data.load_it(split="train", label_encoding="one_hot")
val_ds, shape = data.load_it(split="val", label_encoding="one_hot")

print(class_weights)
print(shape)

logger = Logger(RUN_NAME, clean=True)
length_train = sum(1 for _ in train_ds)
length_val = sum(1 for _ in val_ds)
data_config = {
    "Train length": length_train,
    "Val length": length_val,
    "Tensor shape": str(shape),
}
logger.save_data_config(data_config)

# Create a CNN model
# Load ResNet50 with pre-trained ImageNet weights
base_model = ResNet50(
    weights="imagenet", include_top=False, input_shape=(shape[0], shape[1], 3)
)

# Freeze the base model
base_model.trainable = False
# Create the model
model = tf.keras.Sequential(
    [
        layers.Input(shape=(shape[0], shape[1], 1)),
        layers.Conv2D(
            3, (3, 3), padding="same"
        ),  # This layer converts the 1 channel input to 3 channels
        base_model,
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dense(6, activation="softmax"),
    ]
)

model.summary()
logger.save_model_info(model)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks = []
callbacks.append(EarlyStopping(monitor="val_loss", patience=5))
callbacks.append(TensorBoard(log_dir=logger.get_tensorboard_path(), histogram_freq=1))


history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=callbacks,
    class_weight=class_weights,
)
logger.save_model(model)
logger.save_model_train_history(history.history)


evaluater = Evaluater(model, data, logger, "one_hot")