import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras import layers

# import resnet
from tensorflow.keras.applications import ResNet50

training_dataset = image_dataset_from_directory(
    "../cache/image_from_directory/train/",
    # validation_split=0.2,
    # subset="training",
    seed=123,
    image_size=(63, 512),
    batch_size=32,
    color_mode="grayscale",
)

validation_dataset = image_dataset_from_directory(
    "../cache/image_from_directory/val/",
    # validation_split=0.2,
    # subset="training",
    seed=123,
    image_size=(63, 512),
    batch_size=32,
    color_mode="grayscale",
)


# Freeze the base_model

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
)

print("Testing on electric_quad_drone")
testing_dataset_1 = image_dataset_from_directory(
    "../cache/image_from_directory/test_electric_quad_drone/",
    # subset="testing",
    seed=123,
    image_size=(63, 512),
    batch_size=32,
    color_mode="grayscale",
)
model.evaluate(testing_dataset_1)

print("Testing on speech")
testing_dataset_2 = image_dataset_from_directory(
    "../cache/image_from_directory/test_speech/",
    # subset="testing",
    seed=123,
    image_size=(63, 512),
    batch_size=32,
    color_mode="grayscale",
)
model.evaluate(testing_dataset_2)

print("Testing on chernobyl")
testing_dataset_3 = image_dataset_from_directory(
    "../cache/image_from_directory/test_nature_chernobyl/",
    # subset="testing",
    seed=123,
    image_size=(63, 512),
    batch_size=32,
    color_mode="grayscale",
)
model.evaluate(testing_dataset_3)

print("Testing on animal")
testing_dataset_4 = image_dataset_from_directory(
    "../cache/image_from_directory/test_animal/",
    # subset="testing",
    seed=123,
    image_size=(63, 512),
    batch_size=32,
    color_mode="grayscale",
)
model.evaluate(testing_dataset_4)
