# PATH_TO_SKYLINE = "/cluster/datastore/simonmy/skyline"  # "/workspace/skyline"
PATH_TO_SKYLINE = "/Users/simonmyhre/workdir/gitdir/skyline"

import sys

sys.path.append(PATH_TO_SKYLINE)
from cumulus import calculate_class_weights

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras import layers

# import resnet
from tensorflow.keras.applications import ResNet50

training_dataset = image_dataset_from_directory(
    "../cache/image_from_directory/train/",
    seed=123,
    image_size=(63, 512),
    batch_size=32,
    color_mode="grayscale",
)

validation_dataset = image_dataset_from_directory(
    "../cache/image_from_directory/val/",
    seed=123,
    image_size=(63, 512),
    batch_size=32,
    color_mode="grayscale",
)


class_weights = calculate_class_weights(training_dataset)

shape = (63, 512)

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

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)


history = model.fit(
    training_dataset,
    validation_data=validation_dataset,
    epochs=7,
    class_weight=class_weights,
)

""" Drone testing """

print("Testing on electric_quad_drone")
testing_dataset_1 = image_dataset_from_directory(
    "../cache/image_from_directory/test_electric_quad_drone/",
    seed=123,
    image_size=(63, 512),
    batch_size=32,
    color_mode="grayscale",
)
model.evaluate(testing_dataset_1)


print("Testing on electric_fixedwing_drone")
testing_dataset_2 = image_dataset_from_directory(
    "../cache/image_from_directory/test_electric_fixedwing_drone/",
    seed=123,
    image_size=(63, 512),
    batch_size=32,
    color_mode="grayscale",
)
model.evaluate(testing_dataset_2)

print("Testing on racing_drone")
testing_dataset_3 = image_dataset_from_directory(
    "../cache/image_from_directory/test_racing_drone/",
    seed=123,
    image_size=(63, 512),
    batch_size=32,
    color_mode="grayscale",
)
model.evaluate(testing_dataset_3)

print("Testing on petrol_fixedwing_drone")
testing_dataset_4 = image_dataset_from_directory(
    "../cache/image_from_directory/test_petrol_fixedwing_drone/",
    seed=123,
    image_size=(63, 512),
    batch_size=32,
    color_mode="grayscale",
)
model.evaluate(testing_dataset_4)



""" Testing on non-drone """

print("Testing on speech")
testing_dataset_5 = image_dataset_from_directory(
    "../cache/image_from_directory/test_speech/",
    seed=123,
    image_size=(63, 512),
    batch_size=32,
    color_mode="grayscale",
)
model.evaluate(testing_dataset_5)

print("Testing on chernobyl")
testing_dataset_6 = image_dataset_from_directory(
    "../cache/image_from_directory/test_nature_chernobyl/",
    seed=123,
    image_size=(63, 512),
    batch_size=32,
    color_mode="grayscale",
)
model.evaluate(testing_dataset_6)

print("Testing on dvc_non_drone")
testing_dataset_7 = image_dataset_from_directory(
    "../cache/image_from_directory/test_dvc_non_drone/",
    seed=123,
    image_size=(63, 512),
    batch_size=32,
    color_mode="grayscale",
)
model.evaluate(testing_dataset_7)

print("Testing on animal")
testing_dataset_8 = image_dataset_from_directory(
    "../cache/image_from_directory/test_animal/",
    seed=123,
    image_size=(63, 512),
    batch_size=32,
    color_mode="grayscale",
)
model.evaluate(testing_dataset_8)

print("Testing on TUT_dcase")
testing_dataset_9 = image_dataset_from_directory(
    "../cache/image_from_directory/test_TUT_dcase/",
    seed=123,
    image_size=(63, 512),
    batch_size=32,
    color_mode="grayscale",
)
model.evaluate(testing_dataset_9)

