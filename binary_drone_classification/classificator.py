import sys
import platform

current_os = platform.system()
if current_os == "Darwin":  # macOS
    sys.path.append("/Users/simonmyhre/workdir/gitdir/skyline")
elif current_os == "Linux":  # Linux
    sys.path.append("/cluster/datastore/simonmy/skyline")
from dotenv import load_dotenv
import os

load_dotenv()
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

from cirrus import Data
from cumulus import Logger

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M",
)

# Check if GPU is available
logging.info(
    "Num GPUs Available: %d", len(tf.config.experimental.list_physical_devices("GPU"))
)

data = Data(os.getenv("DATA_INPUT_PATH"), os.getenv("DATA_OUTPUT_PATH"))
data.set_window_size(1)
data.set_split_configuration(train_percent=65, test_percent=20, val_percent=15)
data.set_label_to_class_map(
    {
        "drone": [
            "normal_drone",
            "normal_fixedwing",
            "petrol_fixedwing",
            "racing_drone",
        ],
        "non-drone": ["nature_chernobyl", "false_positives_drone"],
    }
)
data.set_sample_rate(44100)
# data.set_augmentations(['low_pass'])
data.set_audio_format("stft")
data.set_file_type("tfrecord")
data.set_limit(100)
# data.describe_it()

data.make_it()

train_ds, shape, class_weights = data.load_it(split="train", label_encoding="integer")
val_ds, shape = data.load_it(split="val", label_encoding="integer")

print(class_weights)
print(shape)

logger = Logger("run_1", clean=True)
length_train = sum(1 for _ in train_ds)
length_val = sum(1 for _ in val_ds)
data_config = {
    "Train length": length_train,
    "Val length": length_val,
    "Tensor shape": str(shape),
}
logger.save_data_config(data_config)

# Create a CNN model
model = tf.keras.Sequential(
    [
        layers.Input(
            shape=(shape[0], shape[1], 1)
        ),  # # 1 sec window: 128, 87. 2 sec window: 1025, 173.
        layers.Conv2D(32, 7, padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Dropout(0.5),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 4, padding="same"),
        layers.LeakyReLU(),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Dropout(0.5),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding="same"),
        layers.LeakyReLU(),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding="same"),
        layers.LeakyReLU(),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding="same"),
        layers.LeakyReLU(),
        layers.Flatten(),
        layers.Dense(128),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Dense(64),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Dense(1, activation="sigmoid"),
    ]
)

model.summary()
logger.save_model_info(model)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.000005),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

callbacks = []
callbacks.append(EarlyStopping(monitor="val_loss", patience=10))
callbacks.append(TensorBoard(log_dir=logger.get_tensorboard_path(), histogram_freq=1))


history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=1,
    callbacks=callbacks,
    class_weight=class_weights,
)
logger.save_model(model)
logger.save_model_train_history(history.history)


"""
Test the model
"""
test_normal_drone_ds, shape = data.load_it(
    split="test_normal_drone", label_encoding="integer"
)
test_normal_fixedwing_ds, shape = data.load_it(
    split="test_normal_fixedwing", label_encoding="integer"
)
test_petrol_fixedwing_ds, shape = data.load_it(
    split="test_petrol_fixedwing", label_encoding="integer"
)
test_racing_drone_ds, shape = data.load_it(
    split="test_racing_drone", label_encoding="integer"
)
test_nature_chernobyl_ds, shape = data.load_it(
    split="test_nature_chernobyl", label_encoding="integer"
)
test_false_positive_ds, shape = data.load_it(
    split="test_false_positives_drone", label_encoding="integer"
)

# Test evaluations
normal_drone_test_loss, normal_drone_test_acc = model.evaluate(test_normal_drone_ds)
normal_fixedwing_test_loss, normal_fixedwing_test_acc = model.evaluate(
    test_normal_fixedwing_ds
)
petrol_fixedwing_test_loss, petrol_fixedwing_test_acc = model.evaluate(
    test_petrol_fixedwing_ds
)
racing_drone_test_loss, racing_drone_test_acc = model.evaluate(test_racing_drone_ds)
nature_chernobyl_test_loss, nature_chernobyl_test_acc = model.evaluate(
    test_nature_chernobyl_ds
)
false_positive_test_loss, false_positive_test_acc = model.evaluate(
    test_false_positive_ds
)

test_results = {
    "normal_drone": normal_drone_test_acc,
    "normal_fixedwing": normal_fixedwing_test_acc,
    "petrol_fixedwing": petrol_fixedwing_test_acc,
    "racing_drone": racing_drone_test_acc,
    "nature_chernobyl": nature_chernobyl_test_acc,
    "false_positive": false_positive_test_acc,
}
logger.save_model_test_results(test_results)

# Get the average accuracy
average_accuracy = sum(test_results.values()) / len(test_results)
print(f"Average accuracy: {average_accuracy}")
