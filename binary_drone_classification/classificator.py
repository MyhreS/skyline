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
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

from cirrus import Data

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

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.000005),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

callbacks = []
callbacks.append(EarlyStopping(monitor="val_loss", patience=10))


history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=1,
    callbacks=callbacks,
    class_weight=class_weights,
)


logging.info("History:")
logging.info("Train accuracy: %s", history.history["accuracy"])
logging.info("Train loss: %s", history.history["loss"])

test_normal_drone_ds, shape = data.load_it(split="test_normal_drone", label_encoding="integer")
test_normal_fixedwing_ds, shape = data.load_it(split="test_normal_fixedwing", label_encoding="integer")
test_petrol_fixedwing_ds, shape = data.load_it(split="test_petrol_fixedwing", label_encoding="integer")
test_racing_drone_ds, shape = data.load_it(split="test_racing_drone", label_encoding="integer")
test_nature_chernobyl_ds, shape = data.load_it(split="test_nature_chernobyl", label_encoding="integer")
test_false_positive_ds, shape = data.load_it(split="test_false_positives_drone", label_encoding="integer")

# Test evaluations
normal_drone_test_loss, normal_drone_test_acc = model.evaluate(test_normal_drone_ds)
normal_fixedwing_test_loss, normal_fixedwing_test_acc = model.evaluate(test_normal_fixedwing_ds)
petrol_fixedwing_test_loss, petrol_fixedwing_test_acc = model.evaluate(test_petrol_fixedwing_ds)
racing_drone_test_loss, racing_drone_test_acc = model.evaluate(test_racing_drone_ds)
nature_chernobyl_test_loss, nature_chernobyl_test_acc = model.evaluate(test_nature_chernobyl_ds)
false_positive_test_loss, false_positive_test_acc = model.evaluate(test_false_positive_ds)
average_test_acc = (normal_drone_test_acc + normal_fixedwing_test_acc + petrol_fixedwing_test_acc + racing_drone_test_acc + nature_chernobyl_test_acc + false_positive_test_acc) / 6
average_test_loss = (normal_drone_test_loss + normal_fixedwing_test_loss + petrol_fixedwing_test_loss + racing_drone_test_loss + nature_chernobyl_test_loss + false_positive_test_loss) / 6
print("Average test accuracy: ", average_test_acc)
print("Average test loss: ", average_test_loss)


