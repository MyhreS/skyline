# PATH_TO_SKYLINE = "/cluster/datastore/simonmy/skyline"  # "/workspace/skyline"
# PATH_TO_INPUT_DATA = "/cluster/datastore/simonmy/data/datav3"  # "/workspace/data/data"
# PATH_TO_OUTPUT_DATA = (
#     "/cluster/datastore/simonmy/skyline/cache/data"  # "/workspace/skyline/cache/data"
# )
PATH_TO_SKYLINE = "/Users/simonmyhre/workdir/gitdir/skyline"
PATH_TO_INPUT_DATA = "/Users/simonmyhre/workdir/gitdir/sqml/projects/sm_multiclass_masters_project/pull_data/cache/datav3"
PATH_TO_OUTPUT_DATA = "/Users/simonmyhre/workdir/gitdir/skyline/cache/data"
import sys

sys.path.append(PATH_TO_SKYLINE)
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from cirrus import Data
from cumulus import Logger
from stratus import Evaluater


data = Data(PATH_TO_INPUT_DATA, PATH_TO_OUTPUT_DATA)
train_ds, shape, class_weights = data.load_it(split="train", label_encoding="one_hot")
val_ds, shape = data.load_it(split="val", label_encoding="one_hot")

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
        layers.Dense(2, activation="softmax"),
    ]
)

model.summary()

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)


history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=1,
    class_weight=class_weights,
)

logger = Logger("Run_test", clean=True)
evaluater = Evaluater(model, data, logger, "one_hot")
